#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 problem_type 嵌入的检索排序；支持纯嵌入与 utility 混合。"""
from __future__ import annotations

from enum import Enum
from typing import Any

from .skill_item import SkillItem


class RetrieveMode(str, Enum):
    """retrieve 使用的策略；由 SkillManager.retrieve_mode 选择。"""

    EMBEDDING = "embedding"
    HYBRID = "hybrid"


def _encode_query(model: Any, question: str) -> Any:
    """Query 侧：优先使用 Qwen3 推荐的 prompt_name=\"query\"。"""
    enc_kw: dict[str, Any] = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    try:
        return model.encode([question], prompt_name="query", **enc_kw)[0]
    except TypeError:
        return model.encode([question], **enc_kw)[0]


def _encode_documents(model: Any, texts: list[str]) -> Any:
    """文档侧：对 skill 的 problem_type 编码（不传 query prompt）。"""
    import numpy as np

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def rank_skills_for_retrieve(
    *,
    question: str,
    items: list[SkillItem],
    model: Any,
    mode: RetrieveMode,
    retrieve_lambda: float,
) -> list[SkillItem]:
    """
    用 question 与每条 skill 的 problem_type 算相似度并排序。
    - EMBEDDING: 按 cosine sim（L2 归一化后的点积）降序。
    - HYBRID: sim' = (1-λ)*sim + λ*utility，降序。
    """
    import numpy as np

    if not items:
        return []
    doc_texts = [it.problem_type or "" for it in items]
    q_vec = _encode_query(model, question)
    doc_mat = _encode_documents(model, doc_texts)
    if doc_mat.size == 0:
        return list(items)
    # (n_docs,) cosine sim for L2-normalized rows
    sims = doc_mat @ q_vec.astype(np.float64)
    sims = np.asarray(sims, dtype=np.float64)

    if mode == RetrieveMode.EMBEDDING:
        order = np.argsort(-sims)
    elif mode == RetrieveMode.HYBRID:
        utilities = np.array([float(it.utility) for it in items], dtype=np.float64)
        mixed = (1.0 - retrieve_lambda) * sims + retrieve_lambda * utilities
        order = np.argsort(-mixed)
    else:
        raise ValueError(f"unsupported retrieve mode: {mode!r}")

    return [items[int(i)] for i in order]
