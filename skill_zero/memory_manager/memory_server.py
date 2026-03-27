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
from .skill_manager import DEFAULT_EMBEDDING_MODEL, SkillManager
from .skill_memory import SkillMemoryDuplicateIdError, SkillMemoryFullError
from .retrieval import RetrieveMode

app = Flask(__name__)

_manager: SkillManager | None = None
_manager_lock = threading.Lock()


def _get_manager() -> SkillManager:
    if _manager is None:
        raise RuntimeError("manager not initialized")
    return _manager


def _parse_skill_payload(data: dict[str, Any]) -> SkillItem:
    """
    支持两种结构：
    1) 整份即为 jsonl 风格字段（含 \"skill name\" 等）；
    2) {\"skill\": { ... }} 内层同上。
    """
    if "skill" in data and isinstance(data["skill"], dict):
        raw = data["skill"]
    else:
        raw = data
    return SkillItem.from_json_dict(raw)


@app.get("/health")
def health():
    m = _get_manager()
    ready = m.embedding_loaded
    return jsonify(
        {
            "ok": True,
            "embedding_ready": ready,
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
            "skills": [it.to_json_dict() for it in items],
            "count": len(items),
        }
    )


@app.post("/add")
def add():
    """JSON: 与 skills.jsonl 同结构的字段，或 {\"skill\": { ... }}。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "JSON object required"}), 400
    try:
        item = _parse_skill_payload(body)
    except (TypeError, KeyError, ValueError) as e:
        return jsonify({"ok": False, "error": f"invalid skill payload: {e}"}), 400
    if not item.id:
        return jsonify({"ok": False, "error": "skill 'id' is required"}), 400

    m = _get_manager()
    try:
        with _manager_lock:
            m.add(item)
    except SkillMemoryDuplicateIdError as e:
        return jsonify({"ok": False, "error": str(e)}), 409
    except SkillMemoryFullError as e:
        return jsonify({"ok": False, "error": str(e)}), 507

    return jsonify({"ok": True, "id": item.id})


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
                        "retrieve_mode": m.retrieve_mode.value,
                        "retrieve_lambda": m.retrieve_lambda,
                        "embedding_model_name": m.embedding_model_name,
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
    p = argparse.ArgumentParser(description="Skill memory HTTP server (embedding 启动预加载)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-capacity", type=int, default=10_000)
    p.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer 模型名或路径",
    )
    p.add_argument(
        "--embedding-device",
        type=str,
        default="",
        help="留空则由 sentence-transformers 自行选择",
    )
    p.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="加载模型时不传 trust_remote_code=True",
    )
    p.add_argument(
        "--retrieve-mode",
        type=str,
        default=RetrieveMode.EMBEDDING.value,
        choices=[e.value for e in RetrieveMode],
    )
    p.add_argument("--retrieve-lambda", type=float, default=0.5)
    p.add_argument(
        "--skills-jsonl",
        type=str,
        default="",
        help="启动时可选：从该 jsonl 批量导入（已存在 id 会跳过）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global _manager
    args = _parse_args(argv)

    device = args.embedding_device.strip() or None
    _manager = SkillManager(
        max_capacity=args.max_capacity,
        retrieve_mode=args.retrieve_mode,
        retrieve_lambda=args.retrieve_lambda,
        embedding_model_name=args.embedding_model,
        embedding_device=device,
        embedding_trust_remote_code=not args.no_trust_remote_code,
    )

    print("[memory_server] Loading embedding model (warm-up)...", flush=True)
    try:
        _manager.warm_up_embedding()
    except Exception as e:
        print(f"[memory_server] FATAL: embedding warm-up failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    print("[memory_server] Embedding ready.", flush=True)

    if args.skills_jsonl:
        from pathlib import Path

        n = _manager.load_jsonl(Path(args.skills_jsonl))
        print(f"[memory_server] Loaded {n} new skills from {args.skills_jsonl}", flush=True)

    print(
        f"[memory_server] Listening http://{args.host}:{args.port} "
        f"(retrieve_mode={_manager.retrieve_mode.value})",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
