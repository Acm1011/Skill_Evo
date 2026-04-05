#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遍历 deepscaler.jsonl 中每条记录的 problem 字段，用本地 vLLM 嵌入模型
计算与全体 problem 的相似度，取 top-k（不含自身），并写出结果。

嵌入缓存：默认在数据目录旁写入 .npz（doc / query 两套向量）与 .meta.json，
源文件 mtime/size 与条数变化时会自动重算。

初始化与编码逻辑与 skill_zero/memory_manager/retriever_server.py 一致（Qwen3-Embedding instruct）。
vLLM 加载时需 task=embed，否则 Qwen3-Embedding 会被当成因果 LM 导致权重加载报错。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 与 retriever_server.py 默认一致
DEFAULT_EMBEDDING_MODEL = "/home/xzs/data/model/Qwen3-Embedding-0.6B"
DEFAULT_INSTRUCT_TASK = "Given a question, retrieve similar questions that involve similar underlying knowledge."
CACHE_VERSION = 1


def _default_jsonl_path() -> Path:
    return Path("/home/ycy/sdi/data/DeepMath-103K.jsonl")


def _problem_text_from_record(obj: dict[str, Any]) -> str | None:
    """
    支持多种 jsonl 行格式：
    - DeepMath-103K：extra_info.problem
    - 简单格式：顶层 problem
    - 仅有对话：prompt[0].content
    """
    extra = obj.get("extra_info")
    if isinstance(extra, dict):
        p = extra.get("problem")
        if isinstance(p, str) and p.strip():
            return p
    p = obj.get("problem")
    if isinstance(p, str) and p.strip():
        return p
    prompt = obj.get("prompt")
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            c = first.get("content")
            if isinstance(c, str) and c.strip():
                return c
    return None


def load_problems_from_jsonl(jsonl_path: Path) -> list[str]:
    problems: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{jsonl_path}:{line_no} JSON 解析失败") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{jsonl_path}:{line_no} 每行必须是 JSON 对象")
            p = _problem_text_from_record(obj)
            if p is None:
                raise ValueError(
                    f"{jsonl_path}:{line_no} 无法解析题目文本（需要 extra_info.problem、"
                    f"problem 或 prompt[0].content）"
                )
            problems.append(p)
    return problems


def _source_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
    }


def _build_query_text(question: str, instruct_task: str) -> str:
    """将问题格式化为 Qwen3-Embedding instruct 格式（与 retriever_server 一致）。"""
    return f"Instruct: {instruct_task}\nQuery:{question}"


def _embed(model: Any, texts: list[str]) -> np.ndarray:
    """调用 vLLM embed，返回 L2 归一化后的向量矩阵 (N, D)。"""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    import torch

    outputs = model.embed(texts)
    vecs = torch.tensor([o.outputs.embedding for o in outputs]).float()
    norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vecs = vecs / norms
    return vecs.numpy()


def _encode_queries(model: Any, questions: list[str], instruct_task: str) -> np.ndarray:
    instruct_texts = [_build_query_text(q, instruct_task) for q in questions]
    return _embed(model, instruct_texts)


def _encode_documents(model: Any, texts: list[str]) -> np.ndarray:
    return _embed(model, texts)


def _load_vllm_embedding_model(
    model_name: str,
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> Any:
    try:
        from vllm import LLM
    except ImportError as e:
        raise ImportError("离线嵌入需要 vllm：pip install vllm") from e
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    print(
        f"[offline_question_embedding] Loading model {model_name!r} via vLLM "
        f"(task=embed, tensor_parallel_size={tensor_parallel_size}, "
        f"gpu_memory_utilization={gpu_memory_utilization})...",
        flush=True,
    )
    # task 必须为 embed：auto 会把 Qwen3-Embedding 误判为 Qwen3ForCausalLM，
    # 加载权重时报 embed_tokens 不存在。
    model = LLM(
        model=model_name,
        task="embed",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print("[offline_question_embedding] Model ready.", flush=True)
    return model


def encode_batches_local(
    model: Any,
    texts: list[str],
    *,
    is_query: bool,
    instruct_task: str,
    batch_size: int,
) -> np.ndarray:
    """本地分批编码，返回 (N, D) float32。"""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if is_query:
            emb = _encode_queries(model, batch, instruct_task)
        else:
            emb = _encode_documents(model, batch)
        chunks.append(emb.astype(np.float32, copy=False))
    return np.vstack(chunks)


def _meta_path(cache_prefix: Path) -> Path:
    return Path(str(cache_prefix) + ".meta.json")


def _npz_path(cache_prefix: Path) -> Path:
    return Path(str(cache_prefix) + ".npz")


def try_load_embedding_cache(
    jsonl_path: Path,
    cache_prefix: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """若 meta 与源文件一致则返回 (doc_emb, query_emb)，否则 None。"""
    meta_fp = _meta_path(cache_prefix)
    npz_fp = _npz_path(cache_prefix)
    if not meta_fp.is_file() or not npz_fp.is_file():
        return None
    try:
        with meta_fp.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("version") != CACHE_VERSION:
        return None
    fp = _source_fingerprint(jsonl_path)
    if (
        meta.get("source_path") != fp["path"]
        or meta.get("source_size") != fp["size"]
        or meta.get("source_mtime_ns") != fp["mtime_ns"]
    ):
        return None
    loaded = np.load(npz_fp)
    doc = loaded["doc_embeddings"]
    query = loaded["query_embeddings"]
    if int(meta.get("num_problems", -1)) != doc.shape[0] or doc.shape[0] != query.shape[0]:
        return None
    return doc.astype(np.float32, copy=False), query.astype(np.float32, copy=False)


def save_embedding_cache(
    jsonl_path: Path,
    cache_prefix: Path,
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> None:
    fp = _source_fingerprint(jsonl_path)
    meta = {
        "version": CACHE_VERSION,
        "source_path": fp["path"],
        "source_size": fp["size"],
        "source_mtime_ns": fp["mtime_ns"],
        "num_problems": int(doc_embeddings.shape[0]),
        "embedding_dim": int(doc_embeddings.shape[1]) if doc_embeddings.ndim == 2 else 0,
    }
    cache_prefix = cache_prefix.resolve()
    cache_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _npz_path(cache_prefix),
        doc_embeddings=doc_embeddings.astype(np.float32, copy=False),
        query_embeddings=query_embeddings.astype(np.float32, copy=False),
    )
    with _meta_path(cache_prefix).open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_or_load_embeddings(
    jsonl_path: Path,
    cache_prefix: Path,
    *,
    embedding_model: str,
    instruct_task: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    batch_size: int,
    force_rebuild: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not force_rebuild:
        cached = try_load_embedding_cache(jsonl_path, cache_prefix)
        if cached is not None:
            return cached
    problems = load_problems_from_jsonl(jsonl_path)
    if not problems:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    t0 = time.monotonic()
    model = _load_vllm_embedding_model(
        embedding_model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    doc = encode_batches_local(
        model,
        problems,
        is_query=False,
        instruct_task=instruct_task,
        batch_size=batch_size,
    )
    query = encode_batches_local(
        model,
        problems,
        is_query=True,
        instruct_task=instruct_task,
        batch_size=batch_size,
    )
    print(
        f"[offline_question_embedding] Encoded {len(problems)} problems in "
        f"{time.monotonic() - t0:.1f}s.",
        flush=True,
    )
    if doc.shape != query.shape:
        raise RuntimeError(f"doc/query 嵌入形状不一致: {doc.shape} vs {query.shape}")
    save_embedding_cache(jsonl_path, cache_prefix, doc, query)
    return doc, query


def related_indices_topk(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """
    sims[i,j] = query_i · doc_j（向量已 L2 归一化，等价余弦相似度）。
    对每行排除自身下标，取 top_k 列下标，返回 (N, top_k) int64。
    """
    n = query_embeddings.shape[0]
    if n == 0:
        return np.zeros((0, top_k), dtype=np.int64)
    sims = query_embeddings @ doc_embeddings.T
    np.fill_diagonal(sims, -np.inf)
    k = min(top_k, max(0, n - 1))
    if k <= 0:
        return np.zeros((n, 0), dtype=np.int64)
    # 部分排序：每行取 k 个最大（最后 k 列为最大的 k 个下标，彼此无序）
    return np.argpartition(sims, kth=-k, axis=1)[:, -k:]


def dispatch_similar_problems(
    jsonl_path: Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    instruct_task: str = DEFAULT_INSTRUCT_TASK,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    top_k: int = 10,
    cache_prefix: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 32,
    force_rebuild_cache: bool = False,
    output_format: str = "jsonl",
) -> list[dict[str, Any]]:
    """
    主流程：加载/构建嵌入缓存，计算相似 top-k，写出 JSON/JSONL。

    每条输出：{"problem": str, "related_problem": [str, ...]}

    :param output_format: ``jsonl`` 每行一个对象；``json`` 为单个数组（数据量大时占内存）。
    """
    jsonl_path = jsonl_path.resolve()
    if cache_prefix is None:
        cache_prefix = jsonl_path.parent / (jsonl_path.stem + "_embedding_cache")
    if output_path is None:
        suf = ".jsonl" if output_format == "jsonl" else ".json"
        output_path = jsonl_path.parent / f"{jsonl_path.stem}_related_top{top_k}{suf}"

    problems = load_problems_from_jsonl(jsonl_path)
    doc_emb, query_emb = build_or_load_embeddings(
        jsonl_path,
        cache_prefix,
        embedding_model=embedding_model,
        instruct_task=instruct_task,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        batch_size=batch_size,
        force_rebuild=force_rebuild_cache,
    )
    if len(problems) != doc_emb.shape[0]:
        raise RuntimeError(
            f"条数与缓存不一致：jsonl {len(problems)} 条，嵌入 {doc_emb.shape[0]} 行，请使用 --force-rebuild-cache"
        )

    idx_mat = related_indices_topk(query_emb, doc_emb, top_k)
    # argpartition 无序，按相似度在每行内排序
    sims_full = query_emb @ doc_emb.T
    rows: list[dict[str, Any]] = []
    for i, p in enumerate(problems):
        cols = idx_mat[i]
        if cols.size == 0:
            rel: list[str] = []
        else:
            row_sims = sims_full[i, cols]
            order = np.argsort(row_sims)[::-1]
            cols_sorted = cols[order]
            rel = [problems[int(j)] for j in cols_sorted]
        rows.append({"problem": p, "related_problem": rel})

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif output_format == "json":
        with output_path.open("w", encoding="utf-8") as out:
            json.dump(rows, out, ensure_ascii=False, indent=2)
    else:
        raise ValueError("output_format 必须是 jsonl 或 json")
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="deepscaler 题目相似检索（本地 vLLM 嵌入）")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_default_jsonl_path(),
        help="输入 jsonl 路径（默认 skill_src/datas/deepscaler.jsonl）",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="vLLM 模型名或本地路径（默认与 retriever_server 一致）",
    )
    p.add_argument(
        "--instruct-task",
        type=str,
        default=DEFAULT_INSTRUCT_TASK,
        help="Query 侧 instruct 任务描述",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor 并行 GPU 数（默认 1）",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM 显存占用比例 0~1（默认 0.9）",
    )
    p.add_argument("--top-k", type=int, default=10, help="每条 problem 保留的最相似条数（不含自身）")
    p.add_argument(
        "--cache-prefix",
        type=Path,
        default=None,
        help="嵌入缓存文件前缀（默认 <jsonl 同目录>/<stem>_embedding_cache）",
    )
    p.add_argument("--output", type=Path, default=None, help="输出文件路径（默认自动生成）")
    p.add_argument("--batch-size", type=int, default=32, help="每批编码条数")
    p.add_argument("--force-rebuild-cache", action="store_true", help="忽略已有缓存重新编码")
    p.add_argument(
        "--format",
        choices=("jsonl", "json"),
        default="jsonl",
        dest="output_format",
        help="输出格式：jsonl（推荐）或 json 数组",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dispatch_similar_problems(
        args.jsonl,
        embedding_model=args.embedding_model,
        instruct_task=args.instruct_task,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        top_k=args.top_k,
        cache_prefix=args.cache_prefix,
        output_path=args.output,
        batch_size=args.batch_size,
        force_rebuild_cache=args.force_rebuild_cache,
        output_format=args.output_format,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
