#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill memory 对外唯一 HTTP 入口：启动时预加载嵌入模型，提供 /retrieve、/add、/manage。
训练进程仅通过本服务与 memory_manager 交互。
"""
from __future__ import annotations

import argparse
import sys
import threading
from typing import Any

try:
    from flask import Flask, jsonify, request
except ImportError as e:
    raise ImportError(
        "memory_server 依赖 Flask，请先安装: pip install flask"
    ) from e

from .skill_item import SkillItem
from .skill_manager import DEFAULT_RETRIEVER_URL, SkillManager
from .skill_memory import SkillMemoryDuplicateIdError, SkillMemoryFullError
from .retrieval import RetrieveMode

app = Flask(__name__)

_manager: SkillManager | None = None
_manager_lock = threading.Lock()


def _get_manager() -> SkillManager:
    if _manager is None:
        raise RuntimeError("manager not initialized")
    return _manager


def _parse_skill_payload(data: dict[str, Any], *, assigned_id: str = "") -> SkillItem:
    """
    支持两种结构：
    1) 整份即为 jsonl 风格字段（含 \"skill name\" 等）；
    2) {\"skill\": { ... }} 内层同上。

    ``assigned_id`` 非空时会覆盖 payload 中的 id 字段。
    ``reward`` 字段会被自动映射为 ``utility``（在 SkillItem.from_json_dict 中处理）。
    """
    if "skill" in data and isinstance(data["skill"], dict):
        raw = data["skill"]
    else:
        raw = data
    return SkillItem.from_json_dict(raw, assigned_id=assigned_id)


@app.get("/health")
def health():
    m = _get_manager()
    # 查询 retriever_server 状态（不报错，仅标记）
    retriever_ok = False
    try:
        rh = m.retriever_health()
        retriever_ok = bool(rh.get("model_loaded"))
    except Exception:
        pass
    return jsonify(
        {
            "ok": True,
            "retriever_ready": retriever_ok,
            "retriever_url": m.retriever_url,
            "current_size": m.current_size(),
            "max_capacity": m.get_max_capacity(),
        }
    )


@app.post("/retrieve")
def retrieve():
    """JSON: {\"question\": str, \"top_k\"?: int | null}。返回按 retrieve_mode 排序的 skills。"""
    body = request.get_json(silent=True) or {}
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"ok": False, "error": "missing or empty 'question'"}), 400
    top_k = body.get("top_k")
    if top_k is not None:
        try:
            top_k = int(top_k)
            if top_k < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "top_k must be a non-negative int"}), 400

    m = _get_manager()
    try:
        with _manager_lock:
            items = m.retrieve(question.strip(), top_k=top_k)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify(
        {
            "ok": True,
            "skills": [
                {
                    "id": it.id,
                    "skill name": it.skill_name,
                    "problem type": it.problem_type,
                    "key insight": it.key_insight,
                    "method": it.method,
                }
                for it in items
            ],
            "count": len(items),
        }
    )


@app.post("/add")
def add():
    """添加一条 skill 到内存并持久化到默认 jsonl 文件。

    请求体为 JSON 对象，支持以下两种结构：
    - 直接字段（推荐）：包含 ``skill name``、``problem type``、``key insight``、
      ``method``、``skill_from``、``problem``、``reward``（或 ``utility``）。
    - 嵌套结构：``{"skill": { ...上述字段... }}``。

    服务端行为：
    - ``id`` 字段由服务端自动分配（自增整数字符串），忽略请求体中的 ``id``。
    - ``reward`` 字段自动映射为内部 ``utility``；若同时存在 ``utility`` 则优先使用 ``utility``。
    - 主库已满时自动降级到警告区；警告区满时先淘汰最低 utility 的 skill。
    - add 成功后将该条记录追加写入默认持久化 jsonl 文件。

    返回：``{"ok": true, "id": "<id>", "zone": "main"|"warning", "evicted_id": str|null, "persist_path": str}``
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "JSON object required"}), 400

    raw = body.get("skill", body) if isinstance(body.get("skill"), dict) else body
    required_keys = {"skill name", "problem type", "key insight", "method", "skill_from", "problem"}
    missing = required_keys - set(raw.keys())
    if missing:
        return jsonify({"ok": False, "error": f"missing required fields: {sorted(missing)}"}), 400
    if "reward" not in raw and "utility" not in raw:
        return jsonify({"ok": False, "error": "missing required field: 'reward' or 'utility'"}), 400

    m = _get_manager()
    try:
        with _manager_lock:
            new_id = m.allocate_id()
            try:
                item = _parse_skill_payload(body, assigned_id=new_id)
            except (TypeError, KeyError, ValueError) as e:
                return jsonify({"ok": False, "error": f"invalid skill payload: {e}"}), 400

            evict_info = m.add_with_eviction(item)

            try:
                saved_path = m.append_to_jsonl(item)
            except OSError as e:
                return jsonify({
                    "ok": False,
                    "error": f"skill added to memory but failed to persist: {e}",
                    "id": item.id,
                }), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({
        "ok": True,
        "id": item.id,
        "zone": evict_info["zone"],
        "evicted_id": evict_info["evicted_id"],
        "persist_path": str(saved_path),
    })


def _compute_new_utility(
    utility: float,
    R: float,
    is_success: bool,
    lam: float = 0.9,
    tau: float = 0.2,
    u_min: float = 0.0,
    u_max: float = 1.0,
) -> tuple[float, bool]:
    """计算 EMA 更新后的 utility。

    Returns:
        (new_utility, changed) — changed 表示 effective_reward > 0 时才实际更新。
    """
    effective_reward = max(0.0, abs(R) - tau)
    if effective_reward == 0.0:
        return utility, False
    alpha = 1.0 - lam
    sign = 1.0 if is_success else -1.0
    delta = sign * effective_reward
    new_val = utility + alpha * delta
    new_val = max(u_min, min(u_max, new_val))
    return new_val, True


# utility 更新超参（可通过命令行 --update-lam 等覆盖）
_UPDATE_LAM: float = 0.9
_UPDATE_TAU: float = 0.2
_UPDATE_U_MIN: float = 0.0
_UPDATE_U_MAX: float = 1.0


@app.post("/update")
def update():
    """批量更新 skill 的 utility。

    请求体：``{"skills": [{"id": str, "is_success": bool, "reward": float}, ...]}``

    返回：每条条目的处理结果列表。
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "JSON object required"}), 400

    skills_raw = body.get("skills")
    if not isinstance(skills_raw, list) or len(skills_raw) == 0:
        return jsonify({"ok": False, "error": "'skills' must be a non-empty list"}), 400

    m = _get_manager()
    results = []

    with _manager_lock:
        for idx, entry in enumerate(skills_raw):
            if not isinstance(entry, dict):
                results.append({"index": idx, "ok": False, "error": "entry must be a JSON object"})
                continue

            skill_id = entry.get("id")
            is_success = entry.get("is_success")
            reward_raw = entry.get("reward")

            if not isinstance(skill_id, str) or not skill_id:
                results.append({"index": idx, "ok": False, "error": "missing or empty 'id'"})
                continue
            if not isinstance(is_success, bool):
                results.append({"index": idx, "id": skill_id, "ok": False, "error": "'is_success' must be bool"})
                continue
            if reward_raw is None:
                results.append({"index": idx, "id": skill_id, "ok": False, "error": "missing 'reward'"})
                continue
            try:
                reward = float(reward_raw)
            except (TypeError, ValueError):
                results.append({"index": idx, "id": skill_id, "ok": False, "error": "'reward' must be a number"})
                continue

            item, zone = m.get_by_id_any(skill_id)
            if item is None:
                results.append({"index": idx, "id": skill_id, "ok": False, "error": "skill not found"})
                continue

            new_utility, _ = _compute_new_utility(
                item.utility, reward, is_success,
                lam=_UPDATE_LAM, tau=_UPDATE_TAU, u_min=_UPDATE_U_MIN, u_max=_UPDATE_U_MAX,
            )

            try:
                zone_result = m.update_with_zone_logic(
                    skill_id, is_success, new_utility, old_utility=item.utility
                )
            except Exception as e:
                results.append({"index": idx, "id": skill_id, "ok": False, "error": str(e)})
                continue

            results.append({
                "index": idx,
                "id": skill_id,
                "ok": True,
                **zone_result,
            })

        # 批量处理完后全量持久化一次（覆盖写），确保 utility / 使用计数不丢失
        try:
            m.save_jsonl()
        except OSError as e:
            return jsonify({
                "ok": False,
                "error": f"update applied to memory but failed to persist: {e}",
                "results": results,
            }), 500

    return jsonify({"ok": True, "results": results})


@app.post("/manage")
def manage():
    """
    统一管理入口；按 action 分发（可后续扩展）。
    支持: status | get | remove | update | set_max_capacity | set_retrieve_config | list_ids
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "JSON object required"}), 400
    action = body.get("action")
    if not isinstance(action, str) or not action.strip():
        return jsonify({"ok": False, "error": "missing 'action'"}), 400
    action = action.strip().lower()
    m = _get_manager()

    try:
        with _manager_lock:
            if action == "status":
                return jsonify(
                    {
                        "ok": True,
                        "current_size": m.current_size(),
                        "max_capacity": m.get_max_capacity(),
                        "is_full": m.is_full(),
                        "warn_size": m.warn_zone_size(),
                        "warn_capacity": m.warn_zone_capacity(),
                        "retrieve_mode": m.retrieve_mode.value,
                        "retrieve_lambda": m.retrieve_lambda,
                        "retriever_url": m.retriever_url,
                    }
                )
            if action == "warn_status":
                warn_items = m.warn_zone_items()
                return jsonify(
                    {
                        "ok": True,
                        "warn_size": m.warn_zone_size(),
                        "warn_capacity": m.warn_zone_capacity(),
                        "skills": [it.to_json_dict() for it in warn_items],
                    }
                )
            if action == "get":
                sid = body.get("id")
                if not isinstance(sid, str) or not sid:
                    return jsonify({"ok": False, "error": "get requires 'id'"}), 400
                it = m.get(sid)
                if it is None:
                    return jsonify({"ok": False, "error": "not found"}), 404
                return jsonify({"ok": True, "skill": it.to_json_dict()})
            if action == "remove":
                sid = body.get("id")
                if not isinstance(sid, str) or not sid:
                    return jsonify({"ok": False, "error": "remove requires 'id'"}), 400
                ok = m.remove(sid)
                return jsonify({"ok": ok, "removed": ok})
            if action == "update":
                sid = body.get("id")
                fields = body.get("fields")
                if not isinstance(sid, str) or not sid:
                    return jsonify({"ok": False, "error": "update requires 'id'"}), 400
                if not isinstance(fields, dict):
                    return jsonify({"ok": False, "error": "update requires 'fields' object"}), 400
                ok = m.update(sid, **fields)
                if not ok:
                    return jsonify({"ok": False, "error": "not found"}), 404
                return jsonify({"ok": True, "id": sid})
            if action == "set_max_capacity":
                n_raw = body.get("value")
                if n_raw is None:
                    return jsonify({"ok": False, "error": "set_max_capacity needs int 'value'"}), 400
                try:
                    n = int(n_raw)
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": "set_max_capacity needs int 'value'"}), 400
                try:
                    m.set_max_capacity(n)
                except ValueError as e:
                    return jsonify({"ok": False, "error": str(e)}), 400
                return jsonify({"ok": True, "max_capacity": m.get_max_capacity()})
            if action == "set_retrieve_config":
                mode = body.get("retrieve_mode")
                lam = body.get("retrieve_lambda")
                if mode is not None:
                    m.retrieve_mode = mode
                if lam is not None:
                    try:
                        m.retrieve_lambda = float(lam)
                    except ValueError as e:
                        return jsonify({"ok": False, "error": str(e)}), 400
                return jsonify(
                    {
                        "ok": True,
                        "retrieve_mode": m.retrieve_mode.value,
                        "retrieve_lambda": m.retrieve_lambda,
                    }
                )
            if action == "list_ids":
                return jsonify({"ok": True, "ids": m.list_ids()})
            if action == "distill_placeholder":
                # 预留：与 SkillManager.skill_distill 对齐后可改为真实蒸馏
                out = m.skill_distill()
                return jsonify(
                    {
                        "ok": True,
                        "distilled": [it.to_json_dict() for it in out],
                        "note": "skill_distill 当前为占位实现",
                    }
                )

            return jsonify({"ok": False, "error": f"unknown action: {action}"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Skill memory HTTP server")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-capacity", type=int, default=10_000)
    p.add_argument(
        "--retrieve-mode",
        type=str,
        default=RetrieveMode.EMBEDDING.value,
        choices=[e.value for e in RetrieveMode],
    )
    p.add_argument("--retrieve-lambda", type=float, default=0.5)
    p.add_argument(
        "--retriever-url",
        type=str,
        default=DEFAULT_RETRIEVER_URL,
        help=f"retriever_server 的 HTTP 地址（默认 {DEFAULT_RETRIEVER_URL}）",
    )
    p.add_argument(
        "--retriever-timeout",
        type=float,
        default=30.0,
        help="调用 retriever_server 的超时秒数（默认 30s）",
    )
    p.add_argument(
        "--skills-jsonl",
        type=str,
        default="",
        help="启动时可选：从该 jsonl 批量导入（已存在 id 会跳过）",
    )
    p.add_argument(
        "--persist-path",
        type=str,
        default="",
        help=(
            "持久化 jsonl 路径；/add 接口成功后追加写入此文件。"
            "留空则使用默认路径 runs/skills_memory.jsonl"
        ),
    )
    p.add_argument("--update-lam", type=float, default=0.9, help="utility EMA 衰减系数 λ（默认 0.9）")
    p.add_argument("--update-tau", type=float, default=0.2, help="reward 阈值 τ，低于此值不触发更新（默认 0.2）")
    p.add_argument("--update-u-min", type=float, default=0.0, help="utility 下界（默认 0.0）")
    p.add_argument("--update-u-max", type=float, default=1.0, help="utility 上界（默认 1.0）")
    p.add_argument("--warn-capacity", type=int, default=200, help="警告区最大容量（默认 200）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global _manager, _UPDATE_LAM, _UPDATE_TAU, _UPDATE_U_MIN, _UPDATE_U_MAX
    args = _parse_args(argv)

    _UPDATE_LAM = args.update_lam
    _UPDATE_TAU = args.update_tau
    _UPDATE_U_MIN = args.update_u_min
    _UPDATE_U_MAX = args.update_u_max

    persist_path = args.persist_path.strip() or None
    _manager = SkillManager(
        max_capacity=args.max_capacity,
        warn_capacity=args.warn_capacity,
        retrieve_mode=args.retrieve_mode,
        retrieve_lambda=args.retrieve_lambda,
        retriever_url=args.retriever_url,
        retriever_timeout=args.retriever_timeout,
        persist_path=persist_path,
    )

    if args.skills_jsonl:
        from pathlib import Path as _Path

        n = _manager.load_jsonl(_Path(args.skills_jsonl))
        print(f"[memory_server] Loaded {n} new skills from {args.skills_jsonl}", flush=True)

    print(
        f"[memory_server] Listening http://{args.host}:{args.port} "
        f"(retrieve_mode={_manager.retrieve_mode.value}, "
        f"retriever_url={_manager.retriever_url}, "
        f"main_capacity={args.max_capacity}, warn_capacity={args.warn_capacity})",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
