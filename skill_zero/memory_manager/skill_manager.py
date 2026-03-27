#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""管理 SkillMemory：CRUD、容量、retrieve（嵌入 / hybrid）与 skill_distill 占位。"""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .retrieval import RetrieveMode, rank_skills_for_retrieve
from .skill_item import SkillItem
from .skill_memory import SkillMemory

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class SkillManager:
    """对外封装 SkillMemory；retrieve 由 retrieve_mode 选择嵌入或 hybrid。"""

    def __init__(
        self,
        memory: SkillMemory | None = None,
        *,
        max_capacity: int = 10_000,
        retrieve_mode: RetrieveMode | str = RetrieveMode.EMBEDDING,
        retrieve_lambda: float = 0.5,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_device: str | None = None,
        embedding_trust_remote_code: bool = True,
    ) -> None:
        self._memory = memory if memory is not None else SkillMemory(max_capacity=max_capacity)
        self.retrieve_mode = retrieve_mode
        self.retrieve_lambda = retrieve_lambda
        self._embedding_model_name = embedding_model_name
        self._embedding_device = embedding_device
        self._embedding_trust_remote_code = embedding_trust_remote_code
        self._st_model: Any = None

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
    def embedding_model_name(self) -> str:
        return self._embedding_model_name

    @embedding_model_name.setter
    def embedding_model_name(self, name: str) -> None:
        self._embedding_model_name = str(name)
        self._st_model = None

    def _get_sentence_transformer(self) -> Any:
        if self._st_model is not None:
            return self._st_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "embedding 检索需要安装 sentence-transformers（及 PyTorch）。"
                "例如: pip install sentence-transformers"
            ) from e
        kwargs: dict[str, Any] = {"trust_remote_code": self._embedding_trust_remote_code}
        if self._embedding_device is not None:
            kwargs["device"] = self._embedding_device
        self._st_model = SentenceTransformer(self._embedding_model_name, **kwargs)
        return self._st_model

    def warm_up_embedding(self) -> None:
        """预加载嵌入模型（服务端启动时调用，避免首次 retrieve 再加载）。"""
        self._get_sentence_transformer()

    @property
    def embedding_loaded(self) -> bool:
        return self._st_model is not None

    def add(self, item: SkillItem) -> None:
        self._memory.add(item)

    def remove(self, skill_id: str) -> bool:
        return self._memory.remove(skill_id)

    def get(self, skill_id: str) -> SkillItem | None:
        return self._memory.get_by_id(skill_id)

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
        """委托 SkillMemory.bulk_load_from_jsonl。"""
        return self._memory.bulk_load_from_jsonl(path)

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[SkillItem]:
        """
        按 retrieve_mode 检索：用 question 与每条 skill 的 problem_type 做嵌入相似度。
        - embedding：仅按 sim 排序。
        - hybrid：(1-λ)*sim + λ*utility。
        """
        items = self.list_all()
        if not items:
            return []
        model = self._get_sentence_transformer()
        ranked = rank_skills_for_retrieve(
            question=question,
            items=items,
            model=model,
            mode=self._retrieve_mode,
            retrieve_lambda=self._retrieve_lambda,
        )
        if top_k is not None:
            ranked = ranked[: max(0, top_k)]
        return ranked

    def skill_distill(self) -> list[SkillItem]:
        """
        遍历 memory、合并相关 skill 生成新 skill（占位：不做合并）。
        后续可实现聚类或 LLM 合并，并写回 memory。
        """
        return []
