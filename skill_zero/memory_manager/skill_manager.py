#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""管理 SkillMemory：CRUD、容量、retrieve（委托 retriever_server）与 skill_distill 占位。"""
from __future__ import annotations

import json
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .retrieval import RetrieveMode
from .skill_item import SkillItem
from .skill_memory import SkillMemory, WarningZone

DEFAULT_SKILLS_JSONL = Path("runs/skills_memory.jsonl")
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:8766"


class SkillManager:
    """对外封装 SkillMemory；retrieve 委托给独立的 retriever_server（HTTP）。"""

    def __init__(
        self,
        memory: SkillMemory | None = None,
        *,
        max_capacity: int = 10_000,
        warn_capacity: int = 200,
        retrieve_mode: RetrieveMode | str = RetrieveMode.EMBEDDING,
        retrieve_lambda: float = 0.5,
        retriever_url: str = DEFAULT_RETRIEVER_URL,
        retriever_timeout: float = 30.0,
        persist_path: str | Path | None = None,
    ) -> None:
        self._memory = memory if memory is not None else SkillMemory(max_capacity=max_capacity)
        self._warn = WarningZone(max_capacity=warn_capacity)
        self.retrieve_mode = retrieve_mode
        self.retrieve_lambda = retrieve_lambda
        self._retriever_url = retriever_url.rstrip("/")
        self._retriever_timeout = retriever_timeout
        # 持久化路径：优先使用传入参数，否则使用默认路径
        self._persist_path: Path = Path(persist_path) if persist_path else DEFAULT_SKILLS_JSONL
        # 自增 id 计数器，初始化时扫描已有 id 中的最大整数
        self._next_id: int = self._init_next_id()

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
        """在主库和警告区中按 skill_name 查找，优先主库。

        Returns:
            (item, zone)，zone 取值 'main' | 'warning' | 'not_found'
        """
        for item in self._memory.values():
            if item.skill_name == skill_name:
                return item, "main"
        for item in self._warn.values():
            if item.skill_name == skill_name:
                return item, "warning"
        return None, "not_found"

    def get_by_id_any(self, skill_id: str) -> tuple[SkillItem | None, str]:
        """在主库和警告区中按 id 查找，优先主库。

        Returns:
            (item, zone)，zone 取值 'main' | 'warning' | 'not_found'
        """
        item = self._memory.get_by_id(skill_id)
        if item is not None:
            return item, "main"
        item = self._warn.get_by_id(skill_id)
        if item is not None:
            return item, "warning"
        return None, "not_found"

    def add_with_eviction(self, item: SkillItem) -> dict[str, Any]:
        """主库未满则存入主库；否则存入警告区（警告区满时先淘汰最低 utility 的 skill）。

        Returns:
            {
              "zone": "main" | "warning",
              "evicted_id": str | None,   # 被淘汰出警告区的 skill id
            }
        """
        from .skill_memory import SkillMemoryFullError as _Full

        if not self._memory.is_full:
            self._memory.add(item)
            return {"zone": "main", "evicted_id": None}

        # 主库已满 → 进警告区
        evicted_id = None
        if self._warn.is_full:
            evicted = self._warn.evict_min_utility()
            if evicted is not None:
                evicted_id = evicted.id
                print(f"[SkillManager] warn zone evicted: id={evicted.id!r} utility={evicted.utility}", flush=True)

        self._warn.add(item)
        return {"zone": "warning", "evicted_id": evicted_id}

    def update_with_zone_logic(
        self,
        skill_id: str,
        is_success: bool,
        new_utility: float,
        *,
        old_utility: float,
    ) -> dict[str, Any]:
        """在正确的区域（主库/警告区）执行更新，并处理晋升/删除逻辑。

        调用方负责在调用前用 _compute_new_utility 算好 new_utility 和 old_utility。

        Returns:
            {
              "zone": "main" | "warning",
              "action": "updated" | "no_change" | "removed" | "promoted" | "stayed",
              "utility_before": float,
              "utility_after": float,
              "promoted_from_warn_id": str | None,   # 晋升时原 warn skill 的 id
              "demoted_to_warn_id": str | None,      # 晋升时被替换降级的主库 skill 的 id
              "evicted_warn_id": str | None,         # 降级时若警告区满，被淘汰的 warn skill id
            }
        """
        item, zone = self.get_by_id_any(skill_id)
        if item is None or zone == "not_found":
            raise KeyError(f"skill not found: id={skill_id!r}")

        base = {
            "zone": zone,
            "utility_before": old_utility,
            "utility_after": new_utility,
            "promoted_from_warn_id": None,
            "demoted_to_warn_id": None,
            "evicted_warn_id": None,
        }
        changed = new_utility != old_utility

        # 无论在主库还是警告区，只要调用 /update 就递增使用计数
        new_success = item.skill_usage_success + (1 if is_success else 0)
        new_failure = item.skill_usage_failure + (0 if is_success else 1)

        if zone == "main":
            self.update(item.id, utility=new_utility if changed else item.utility,
                        skill_usage_success=new_success, skill_usage_failure=new_failure)
            base["action"] = "updated" if changed else "no_change"
            return base

        # ── 警告区逻辑 ──────────────────────────────────────────────────────
        if not is_success:
            # 失败时彻底删除，不需要保留计数
            self._warn.remove(item.id)
            base["action"] = "removed"
            base["utility_after"] = old_utility
            return base

        # is_success=True：更新警告区中的 utility 和计数（计数始终递增）
        updated_item = item.__class__(
            skill_name=item.skill_name,
            problem_type=item.problem_type,
            key_insight=item.key_insight,
            method=item.method,
            skill_from=item.skill_from,
            id=item.id,
            problem=item.problem,
            utility=new_utility if changed else item.utility,
            skill_usage_success=new_success,
            skill_usage_failure=new_failure,
        )
        self._warn.remove(item.id)
        self._warn.add(updated_item)
        item = updated_item

        # 检查是否能晋升：只有 utility 实际发生变化时才判断晋升
        # （effective_reward=0 时 changed=False，不触发晋升）
        main_min = self._memory.min_utility_item()
        if changed and main_min is not None and new_utility > main_min.utility:
            # 晋升：warn → main，main 最低 → warn
            self._warn.remove(item.id)
            self._memory.remove(main_min.id)

            # 被替换的主库 skill 降级到警告区
            evicted_warn_id = None
            if self._warn.is_full:
                evicted = self._warn.evict_min_utility()
                if evicted is not None:
                    evicted_warn_id = evicted.id
                    print(f"[SkillManager] warn zone evicted during promotion: id={evicted.id!r}", flush=True)

            self._warn.add(main_min)
            self._memory.add(item)

            base["action"] = "promoted"
            base["promoted_from_warn_id"] = item.id
            base["demoted_to_warn_id"] = main_min.id
            base["evicted_warn_id"] = evicted_warn_id
        else:
            base["action"] = "stayed" if changed else "no_change"

        return base

    # ── 警告区查询 ────────────────────────────────────────────────────────────

    def warn_zone_items(self) -> list[SkillItem]:
        return list(self._warn.values())

    def warn_zone_size(self) -> int:
        return self._warn.capacity

    def warn_zone_capacity(self) -> int:
        return self._warn.max_capacity

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
        # 重新对齐 _next_id，避免与已加载的 id 冲突
        self._next_id = self._init_next_id()
        return n

    def save_jsonl(self, path: str | Path | None = None) -> Path:
        """将内存中所有 skill 全量写入 jsonl 文件（覆盖）。

        Args:
            path: 目标路径；为 None 时使用 self.persist_path。
        Returns:
            实际写入的文件路径。
        """
        target = Path(path) if path is not None else self._persist_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for item in self._memory.values():
                f.write(item.to_jsonl_line() + "\n")
        return target

    def append_to_jsonl(self, item: SkillItem, path: str | Path | None = None) -> Path:
        """将单条 skill 追加写入 jsonl 文件（增量持久化，无需重写全部内容）。

        Args:
            item: 要持久化的 SkillItem。
            path: 目标路径；为 None 时使用 self.persist_path。
        Returns:
            实际写入的文件路径。
        """
        target = Path(path) if path is not None else self._persist_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(item.to_jsonl_line() + "\n")
        return target

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[SkillItem]:
        """通过 retriever_server HTTP 接口完成嵌入检索并返回排序后的 SkillItem 列表。

        内部将主库与警告区的 candidates 合并后一起送给 retriever_server 排序，
        警告区的 skill 可被正常检索到，对外返回接口不变。

        - embedding 模式：按 cosine 相似度降序。
        - hybrid 模式：(1-λ)*sim + λ*utility 降序。
        """
        try:
            import requests as _req
        except ImportError as e:
            raise ImportError("需要安装 requests：pip install requests") from e

        # 主库 + 警告区合并
        items: list[SkillItem] = self.list_all() + self.warn_zone_items()
        if not items:
            return []

        candidates = [
            {"problem_type": it.problem_type, "utility": it.utility}
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

    def skill_distill(self) -> list[SkillItem]:
        """
        遍历 memory、合并相关 skill 生成新 skill（占位：不做合并）。
        后续可实现聚类或 LLM 合并，并写回 memory。
        """
        return []
