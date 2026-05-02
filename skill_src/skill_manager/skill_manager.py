#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""管理 SkillMemory：CRUD、容量、retrieve（委托 retriever_server）与 utility 更新。"""
from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .retrieval import RetrieveMode
from .skill_item import SkillItem
from .skill_memory import SkillMemory

DEFAULT_SKILLS_JSONL = Path("runs/skills_memory.jsonl")
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:8766"
# 调用独立 retriever_server 的 HTTP 超时（秒）；prepare_solver_skills 等会逐条 rank，易超过 30s
_DEFAULT_RETRIEVER_TIMEOUT = 300.0


def _resolve_retriever_timeout(explicit: float | None) -> float:
    """未显式传入时读 ``SE_RETRIEVER_TIMEOUT``，缺省为 ``_DEFAULT_RETRIEVER_TIMEOUT``。"""
    if explicit is not None:
        return float(explicit)
    raw = (os.environ.get("SE_RETRIEVER_TIMEOUT") or "").strip()
    if not raw:
        return _DEFAULT_RETRIEVER_TIMEOUT
    try:
        return max(0.1, float(raw))
    except ValueError:
        return _DEFAULT_RETRIEVER_TIMEOUT


def parse_is_success(raw: Any) -> bool:
    """支持 bool 或 0/1。"""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    raise TypeError(f"is_success must be bool or 0/1, got {type(raw).__name__}")


def _mean_group_acc(entry: dict[str, Any]) -> float | None:
    """从 ``group_infos['acc']`` 列表取标量；无效则返回 None。"""
    gi = entry.get("group_infos")
    if not isinstance(gi, dict):
        return None
    al = gi.get("acc")
    if not isinstance(al, (list, tuple)) or len(al) == 0:
        return None
    try:
        vals = [float(x) for x in al]
    except (TypeError, ValueError):
        return None
    return sum(vals) / len(vals)


def compute_next_utility(
    utility: float,
    R: float,
    acc: float,
    *,
    alpha: float,
    tau: float,
    u_min: float,
    u_max: float,
    acc_mid: float = 0.5,
) -> tuple[float, bool]:
    """U_{t+1} = clip(U_t + α * s_t * max(0, R - τ), U_min, U_max)。

    其中 s_t = 2 * (acc - acc_mid)，在 [0,1] 上 acc=acc_mid 时 s_t=0；
    纯 0/1 时与旧二值 s_t=±1 一致。

    Returns:
        (new_utility, changed) — max(0, R-τ)==0 时增量为 0，changed 为 False。
    """
    st = 2.0 * (float(acc) - float(acc_mid))
    delta = max(0.0, float(R) - float(tau))
    if delta == 0.0:
        return float(utility), False
    step = float(alpha) * st * delta
    new_val = float(utility) + step
    new_val = max(float(u_min), min(float(u_max), new_val))
    return new_val, new_val != float(utility)


class SkillManager:
    """对外封装 SkillMemory；retrieve 委托给独立的 retriever_server（HTTP）。"""

    def __init__(
        self,
        memory: SkillMemory | None = None,
        *,
        max_capacity: int = 1000,
        retrieve_mode: RetrieveMode | str = RetrieveMode.HYBRID,
        retrieve_lambda: float = 0.5,
        retriever_url: str = DEFAULT_RETRIEVER_URL,
        retriever_timeout: float | None = None,
        persist_path: str | Path | None = None,
        utility_alpha: float = 0.1,
        utility_tau: float = 0.2,
        utility_u_min: float = 0.0,
        utility_u_max: float = 1.0,
    ) -> None:
        self._memory = memory if memory is not None else SkillMemory(max_capacity=max_capacity)
        self.retrieve_mode = retrieve_mode
        self.retrieve_lambda = retrieve_lambda
        self._retriever_url = retriever_url.rstrip("/")
        self._retriever_timeout = _resolve_retriever_timeout(retriever_timeout)
        self._persist_path: Path = Path(persist_path) if persist_path else DEFAULT_SKILLS_JSONL
        self._utility_alpha = float(utility_alpha)
        self._utility_tau = float(utility_tau)
        self._utility_u_min = float(utility_u_min)
        self._utility_u_max = float(utility_u_max)
        self._next_id: int = self._init_next_id()

    @property
    def utility_alpha(self) -> float:
        return self._utility_alpha

    @utility_alpha.setter
    def utility_alpha(self, v: float) -> None:
        self._utility_alpha = float(v)

    @property
    def utility_tau(self) -> float:
        return self._utility_tau

    @utility_tau.setter
    def utility_tau(self, v: float) -> None:
        self._utility_tau = float(v)

    @property
    def utility_u_min(self) -> float:
        return self._utility_u_min

    @utility_u_min.setter
    def utility_u_min(self, v: float) -> None:
        self._utility_u_min = float(v)

    @property
    def utility_u_max(self) -> float:
        return self._utility_u_max

    @utility_u_max.setter
    def utility_u_max(self, v: float) -> None:
        self._utility_u_max = float(v)

    def _init_next_id(self) -> int:
        """扫描当前 memory 中已有的整数 id，返回 max+1（若无则从 0 开始）。"""
        max_id = -1
        for sid, _ in self._memory.items():
            try:
                v = int(sid)
                if v > max_id:
                    max_id = v
            except (ValueError, TypeError):
                pass
        return max_id + 1

    def allocate_id(self) -> str:
        """分配下一个自增 id 字符串，并将计数器递增。"""
        new_id = str(self._next_id)
        self._next_id += 1
        return new_id

    @property
    def persist_path(self) -> Path:
        return self._persist_path

    @persist_path.setter
    def persist_path(self, path: str | Path) -> None:
        self._persist_path = Path(path)

    @property
    def memory(self) -> SkillMemory:
        return self._memory

    @property
    def retrieve_mode(self) -> RetrieveMode:
        return self._retrieve_mode

    @retrieve_mode.setter
    def retrieve_mode(self, value: RetrieveMode | str) -> None:
        self._retrieve_mode = RetrieveMode(value) if isinstance(value, str) else value

    @property
    def retrieve_lambda(self) -> float:
        """Hybrid 公式中的 λ：sim' = (1-λ)*sim + λ*utility。"""
        return self._retrieve_lambda

    @retrieve_lambda.setter
    def retrieve_lambda(self, value: float) -> None:
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError("retrieve_lambda must be in [0, 1]")
        self._retrieve_lambda = v

    @property
    def retriever_url(self) -> str:
        return self._retriever_url

    @retriever_url.setter
    def retriever_url(self, url: str) -> None:
        self._retriever_url = url.rstrip("/")

    def retriever_health(self) -> dict[str, Any]:
        """查询 retriever_server 健康状态；返回原始 JSON dict。"""
        try:
            import requests as _req
        except ImportError as e:
            raise ImportError("需要安装 requests：pip install requests") from e
        resp = _req.get(f"{self._retriever_url}/health", timeout=self._retriever_timeout)
        return resp.json()

    def insert_skills(
        self,
        items: list[SkillItem],
        *,
        use_eviction: bool = True,
    ) -> list[dict[str, Any]]:
        """批量插入 skill。use_eviction=True 时主库满则淘汰 utility 最小的一条再插入；否则满库抛错。

        空 id 的条目会分配 allocate_id()（并写回 item 需由调用方保证；此处会 replace id 后插入）。
        """
        results: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            it = item
            if not it.id:
                it = replace(item, id=self.allocate_id())
            try:
                if use_eviction:
                    info = self.add_with_eviction(it)
                    results.append({"index": idx, "ok": True, "id": it.id, **info})
                else:
                    self.add(it)
                    results.append({"index": idx, "ok": True, "id": it.id, "zone": "main", "evicted_id": None})
            except Exception as e:
                results.append({"index": idx, "ok": False, "id": it.id, "error": str(e)})
        self._next_id = self._init_next_id()
        return results

    def update_utilities_from_rewards(
        self,
        updates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """按 reward 行更新 utility。

        - **旧行**：``id``、``is_success``、``reward``（R_t）。将 ``is_success`` 映射为 acc∈{0,1} 后代入公式。
        - **Solver / group 行**（与 ``reward_manager`` 的 ``reward_info`` 对齐）：非空
          ``skill_id`` 列表、``group_infos['acc']`` 为非空列表；对 acc 做均值得到 acc_bar，并对列表中
          每个 id 各更新一次。``reward`` 可省略，默认 ``1.0``。
        """
        results: list[dict[str, Any]] = []
        for idx, entry in enumerate(updates):
            if not isinstance(entry, dict):
                results.append({"index": idx, "ok": False, "error": "entry must be a dict"})
                continue

            raw_sids = entry.get("skill_id")
            is_group = isinstance(raw_sids, (list, tuple)) and "skill_id" in entry
            if is_group and len(raw_sids) == 0:
                results.append({"index": idx, "ok": False, "error": "empty 'skill_id' list"})
                continue
            if is_group:
                skill_ids: list[str] = []
                for s in raw_sids:
                    t = str(s).strip() if s is not None else ""
                    if t:
                        skill_ids.append(t)
                if not skill_ids:
                    results.append({"index": idx, "ok": False, "error": "no valid id in 'skill_id'"})
                    continue
                acc_bar = _mean_group_acc(entry)
                if acc_bar is None:
                    results.append(
                        {
                            "index": idx,
                            "ok": False,
                            "error": "missing or empty group_infos['acc'] for skill_id row",
                        }
                    )
                    continue
                acc_val = float(acc_bar)
                rw = entry.get("reward", 1.0)
                if rw is None:
                    rw = 1.0
                try:
                    reward = float(rw)
                except (TypeError, ValueError):
                    results.append({"index": idx, "ok": False, "error": "'reward' must be a number"})
                    continue
            else:
                skill_id = entry.get("id")
                if not isinstance(skill_id, str) or not skill_id:
                    results.append(
                        {
                            "index": idx,
                            "ok": False,
                            "error": "missing or empty 'id' (and not a skill_id list row)",
                        }
                    )
                    continue
                skill_ids = [skill_id]
                raw_ok = entry.get("is_success")
                if raw_ok is None:
                    results.append({"index": idx, "id": skill_id, "ok": False, "error": "missing 'is_success'"})
                    continue
                try:
                    is_ok = parse_is_success(raw_ok)
                except TypeError as e:
                    results.append({"index": idx, "id": skill_id, "ok": False, "error": str(e)})
                    continue
                acc_val = 1.0 if is_ok else 0.0
                reward_raw = entry.get("reward")
                if reward_raw is None:
                    results.append({"index": idx, "id": skill_id, "ok": False, "error": "missing 'reward'"})
                    continue
                try:
                    reward = float(reward_raw)
                except (TypeError, ValueError):
                    results.append(
                        {"index": idx, "id": skill_id, "ok": False, "error": "'reward' must be a number"}
                    )
                    continue

            n_skills = len(skill_ids)
            for sub, skill_id in enumerate(skill_ids):
                item, zone = self.get_by_id_any(skill_id)
                if item is None:
                    r: dict[str, Any] = {
                        "index": idx,
                        "id": skill_id,
                        "ok": False,
                        "error": "skill not found",
                    }
                    if n_skills > 1:
                        r["sub"] = sub
                    results.append(r)
                    continue

                is_count_success = acc_val >= 0.5
                new_utility, _ = compute_next_utility(
                    item.utility,
                    reward,
                    acc_val,
                    alpha=self._utility_alpha,
                    tau=self._utility_tau,
                    u_min=self._utility_u_min,
                    u_max=self._utility_u_max,
                )
                try:
                    zone_result = self.update_with_zone_logic(
                        skill_id, is_count_success, new_utility, old_utility=item.utility
                    )
                except Exception as e:
                    r = {"index": idx, "id": skill_id, "ok": False, "error": str(e)}
                    if n_skills > 1:
                        r["sub"] = sub
                    results.append(r)
                    continue

                out: dict[str, Any] = {"index": idx, "id": skill_id, "ok": True, **zone_result}
                if n_skills > 1:
                    out["sub"] = sub
                results.append(out)

        return results

    def add(self, item: SkillItem) -> None:
        self._memory.add(item)

    def remove(self, skill_id: str) -> bool:
        return self._memory.remove(skill_id)

    def get(self, skill_id: str) -> SkillItem | None:
        return self._memory.get_by_id(skill_id)

    def get_by_name(self, skill_name: str) -> SkillItem | None:
        """按 skill_name 在主库线性查找；找到第一条匹配则返回，否则返回 None。"""
        for item in self._memory.values():
            if item.skill_name == skill_name:
                return item
        return None

    def get_by_name_any(self, skill_name: str) -> tuple[SkillItem | None, str]:
        """在主库中按 skill_name 查找。

        Returns:
            (item, zone)，zone 取值 'main' | 'not_found'
        """
        for item in self._memory.values():
            if item.skill_name == skill_name:
                return item, "main"
        return None, "not_found"

    def get_by_id_any(self, skill_id: str) -> tuple[SkillItem | None, str]:
        """在主库中按 id 查找。

        Returns:
            (item, zone)，zone 取值 'main' | 'not_found'
        """
        item = self._memory.get_by_id(skill_id)
        if item is not None:
            return item, "main"
        return None, "not_found"

    def add_with_eviction(self, item: SkillItem) -> dict[str, Any]:
        """主库未满则 ``add``；若 id 已存在则 ``replace``；若满则淘汰 utility 最小的一条再 ``add``。

        Returns:
            ``zone`` 恒为 ``\"main\"``；``evicted_id`` 为被驱逐的 id（无则为 None）；``replaced`` 表示是否覆盖已存在 id。
        """
        if self._memory.get_by_id(item.id) is not None:
            self.replace(item)
            return {"zone": "main", "evicted_id": None, "replaced": True}

        if not self._memory.is_full:
            self._memory.add(item)
            return {"zone": "main", "evicted_id": None, "replaced": False}

        victim = self._memory.min_utility_item()
        if victim is None:
            raise RuntimeError("skill memory is_full but min_utility_item is None")
        evicted_id = victim.id
        self._memory.remove(evicted_id)
        self._memory.add(item)
        return {"zone": "main", "evicted_id": evicted_id, "replaced": False}

    def update_with_zone_logic(
        self,
        skill_id: str,
        is_success: bool,
        new_utility: float,
        *,
        old_utility: float,
    ) -> dict[str, Any]:
        """在主库更新 utility 与 usage 计数（无警告区）。"""
        item, zone = self.get_by_id_any(skill_id)
        if item is None or zone == "not_found":
            raise KeyError(f"skill not found: id={skill_id!r}")

        changed = new_utility != old_utility
        new_success = item.skill_usage_success + (1 if is_success else 0)
        new_failure = item.skill_usage_failure + (0 if is_success else 1)

        self.update(
            item.id,
            utility=new_utility if changed else item.utility,
            skill_usage_success=new_success,
            skill_usage_failure=new_failure,
        )
        return {
            "zone": "main",
            "utility_before": old_utility,
            "utility_after": new_utility,
            "action": "updated" if changed else "no_change",
        }

    def update(self, skill_id: str, **kwargs: Any) -> bool:
        """
        按 id 部分更新字段；kwargs 的键须为 SkillItem 的 dataclass 字段名（如 utility, skill_name）。
        不存在则返回 False。
        """
        old = self._memory.get_by_id(skill_id)
        if old is None:
            return False
        allowed = {f.name for f in fields(SkillItem)}
        bad = set(kwargs) - allowed
        if bad:
            raise TypeError(f"unknown SkillItem fields: {sorted(bad)}")
        new_item = replace(old, **kwargs)
        self._memory.remove(skill_id)
        self._memory.add(new_item)
        return True

    def replace(self, item: SkillItem) -> bool:
        """整条替换；id 必须已存在。"""
        if self._memory.get_by_id(item.id) is None:
            return False
        self._memory.remove(item.id)
        self._memory.add(item)
        return True

    def list_all(self) -> list[SkillItem]:
        return list(self._memory.values())

    def list_ids(self) -> list[str]:
        return [sid for sid, _ in self._memory.items()]

    def set_max_capacity(self, n: int) -> None:
        self._memory.set_max_capacity(n)

    def get_max_capacity(self) -> int:
        return self._memory.max_capacity

    def current_size(self) -> int:
        return self._memory.capacity

    def is_full(self) -> bool:
        return self._memory.is_full

    def load_jsonl(self, path: str | Path) -> int:
        """委托 SkillMemory.bulk_load_from_jsonl；加载后同步自增 id 计数器。"""
        n = self._memory.bulk_load_from_jsonl(path)
        self._next_id = self._init_next_id()
        return n

    def save_jsonl(self, path: str | Path | None = None) -> Path:
        """将内存中所有 skill 全量写入 jsonl 文件（覆盖）。"""
        target = Path(path) if path is not None else self._persist_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for item in self._memory.values():
                f.write(item.to_jsonl_line() + "\n")
        return target

    def append_to_jsonl(self, item: SkillItem, path: str | Path | None = None) -> Path:
        """将单条 skill 追加写入 jsonl 文件。"""
        target = Path(path) if path is not None else self._persist_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(item.to_jsonl_line() + "\n")
        return target

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[SkillItem]:
        """通过 retriever_server HTTP 接口完成嵌入检索并返回排序后的 SkillItem 列表。"""
        try:
            import requests as _req
        except ImportError as e:
            raise ImportError("需要安装 requests：pip install requests") from e

        items: list[SkillItem] = self.list_all()
        if not items:
            return []

        candidates = [
            {
                "id": it.id,
                "problem_type": it.problem_type,
                "utility": it.utility,
            }
            for it in items
        ]
        payload: dict[str, Any] = {
            "question": question,
            "candidates": candidates,
            "mode": self._retrieve_mode.value,
            "retrieve_lambda": self._retrieve_lambda,
        }
        if top_k is not None:
            payload["top_k"] = top_k

        try:
            resp = _req.post(
                f"{self._retriever_url}/rank",
                json=payload,
                timeout=self._retriever_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(
                f"retriever_server 调用失败 ({self._retriever_url}/rank): {e}"
            ) from e

        if not data.get("ok"):
            raise RuntimeError(f"retriever_server 返回错误: {data.get('error')}")

        ranked_indices: list[int] = data["ranked_indices"]
        return [items[i] for i in ranked_indices]

    def sync_retriever_doc_cache(self) -> None:
        """将当前主库中全部 skill 的 ``problem_type`` 文档嵌入同步到 retriever（``POST /docs/replace`` 全量替换）。"""
        try:
            import requests as _req
        except ImportError as e:
            raise ImportError("需要安装 requests：pip install requests") from e
        its = self.list_all()
        items = [{"id": it.id, "text": it.problem_type} for it in its]
        resp = _req.post(
            f"{self._retriever_url}/docs/replace",
            json={"items": items},
            timeout=self._retriever_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            err = data.get("error", data)
            raise RuntimeError(f"retriever_server /docs/replace 失败: {err}")

    @staticmethod
    def skills_block_for_template(skills: list[SkillItem]) -> str:
        """将多条 skill 拼成 `skill_use_v1` 等模板里 `{skill}` 占位符的内容。

        每条用 `to_json_dict` 再 `json.dumps`；多条之间以 ``\\n\\n---\\n\\n`` 分隔。
        无检索结果时返回空串，避免占位误导模型。
        """
        if not skills:
            return ""
        parts = [json.dumps(s.to_json_dict(), ensure_ascii=False) for s in skills]
        return "\n\n---\n\n".join(parts)

    def skill_distill(self) -> list[SkillItem]:
        """占位：不做合并。"""
        return []
