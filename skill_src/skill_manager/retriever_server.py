#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立 Embedding 检索服务（vLLM 后端），由 ``skill_manager`` 包提供。

职责：
- 启动时通过 vLLM 加载 embedding 模型（task="embed"）。
- 提供 /encode 接口（对文本列表编码，返回 float 列表）。
- 提供 /rank  接口（传入 question + candidates 列表，直接返回排序结果）。
- 提供 /health 接口。
- 若超过 idle_timeout 秒内没有任何检索调用，自动卸载模型并退出进程，
  释放 GPU 显存（下次需要时由 start_retriever_server.sh 重新启动）。

Query 使用 Instruct 格式（适配 Qwen3-Embedding）：
  "Instruct: {task_description}\nQuery:{question}"
Document 直接传入原文，无需 instruct 前缀。

指定 ``--doc-cache-dir``（或由 ``MEMORY_PATH_DIR``/``SE_MEMORY_DIR`` 默认得到
``<Memory>/doc_embed_cache``）时，document 嵌入会落盘为 ``emb_<sha256(id)>.npy``，
与 ``/docs/replace`` 全量一致；进程重启后可从磁盘恢复，避免仅依赖内存。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

try:
    from flask import Flask, jsonify, request
except ImportError as e:
    raise ImportError("retriever_server 依赖 Flask：pip install flask") from e

import numpy as np

app = Flask(__name__)

# ─── 全局状态 ─────────────────────────────────────────────────────────────────
_model: Any = None
_model_lock = threading.Lock()
_model_name: str = ""
_tensor_parallel_size: int = 1
_gpu_memory_utilization: float = 0.9
# instruct 任务描述，用于 query 侧格式化
_instruct_task: str = "Given a question, retrieve relevant skills that help answer it"

_last_call_time: float = time.monotonic()
_idle_timeout: float = 300.0
_idle_timer: threading.Timer | None = None
_idle_lock = threading.Lock()
# 按 skill id 缓存 document 侧 L2 归一化向量，与 /docs/replace 全量同步
_doc_cache: dict[str, np.ndarray] = {}
_doc_cache_lock = threading.Lock()
# 非空时：将向量同步写入该目录下 emb_*.npy；/rank 在内存未命中时尝试读盘
_doc_cache_dir: str | None = None


def _embed_filename_for_id(skill_id: str) -> str:
    h = hashlib.sha256(skill_id.encode("utf-8")).hexdigest()
    return f"emb_{h}.npy"


def _load_doc_vec_from_disk(skill_id: str) -> np.ndarray | None:
    if not _doc_cache_dir:
        return None
    p = Path(_doc_cache_dir) / _embed_filename_for_id(skill_id)
    if not p.is_file():
        return None
    return np.load(p).astype(np.float64)


def _persist_doc_cache_to_dir(ids: list[str], vecs: np.ndarray) -> None:
    """全量替换目录：清空 emb_*.npy 后写入当前 vectors。"""
    if not _doc_cache_dir:
        return
    d = Path(_doc_cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("emb_*.npy"):
        try:
            f.unlink()
        except OSError:
            pass
    for i, sid in enumerate(ids):
        if i < int(vecs.shape[0]):
            np.save(d / _embed_filename_for_id(sid), vecs[i])


# ─── instruct 格式化 ───────────────────────────────────────────────────────────

def _build_query_text(question: str) -> str:
    """将问题格式化为 Qwen3-Embedding instruct 格式。"""
    return f"Instruct: {_instruct_task}\nQuery:{question}"


# ─── 模型管理 ─────────────────────────────────────────────────────────────────

def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    try:
        from vllm import LLM
    except ImportError as e:
        raise ImportError(
            "embedding 检索需要 vllm：pip install vllm"
        ) from e
    # 混合 GPU 型号时需要设置 PCI_BUS_ID 顺序，避免设备编号错乱
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    print(
        f"[retriever_server] Loading model {_model_name!r} via vLLM "
        f"(tensor_parallel_size={_tensor_parallel_size}, "
        f"gpu_memory_utilization={_gpu_memory_utilization})...",
        flush=True,
    )
    # task=embed：auto 会把 Qwen3-Embedding 误判为 Qwen3ForCausalLM，加载权重失败（embed_tokens）。
    _model = LLM(
        model=_model_name,
        task="embed",
        tensor_parallel_size=_tensor_parallel_size,
        gpu_memory_utilization=_gpu_memory_utilization,
    )
    print("[retriever_server] Model ready.", flush=True)
    return _model


def _unload_model() -> None:
    global _model
    with _model_lock:
        if _model is not None:
            print("[retriever_server] Idle timeout reached — unloading model and exiting.", flush=True)
            _model = None
    # os._exit 强制退出所有线程（包括 Flask worker），避免 sys.exit 被线程捕获
    os._exit(0)


def _reset_idle_timer() -> None:
    global _last_call_time, _idle_timer
    _last_call_time = time.monotonic()
    with _idle_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
        _idle_timer = threading.Timer(_idle_timeout, _unload_model)
        _idle_timer.daemon = True
        _idle_timer.start()


# ─── 编码工具 ─────────────────────────────────────────────────────────────────

def _embed(model: Any, texts: list[str]) -> np.ndarray:
    """调用 vLLM embed 接口，返回 L2 归一化后的向量矩阵 (N, D)。"""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    import torch
    outputs = model.embed(texts)
    vecs = torch.tensor([o.outputs.embedding for o in outputs]).float()
    # L2 归一化，保证与 sentence-transformers 行为一致
    norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vecs = vecs / norms
    return vecs.numpy()


def _encode_queries(model: Any, questions: list[str]) -> np.ndarray:
    """Query 侧：套 instruct 格式后编码。"""
    instruct_texts = [_build_query_text(q) for q in questions]
    return _embed(model, instruct_texts)


def _encode_documents(model: Any, texts: list[str]) -> np.ndarray:
    """Document 侧：直接编码，不加 instruct 前缀。"""
    return _embed(model, texts)


# ─── 路由 ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    with _model_lock:
        loaded = _model is not None
    with _doc_cache_lock:
        n_cache = len(_doc_cache)
    return jsonify({
        "ok": True,
        "model_loaded": loaded,
        "model_name": _model_name,
        "idle_timeout": _idle_timeout,
        "idle_remaining": max(0.0, round(_idle_timeout - (time.monotonic() - _last_call_time), 1)),
        "doc_cache_size": n_cache,
        "doc_cache_dir": _doc_cache_dir,
    })


@app.post("/docs/replace")
def docs_replace():
    """全量替换 document 嵌入缓存。body: ``{\"items\": [{\"id\": str, \"text\": str}, ...]}``。"""
    global _doc_cache
    body = request.get_json(silent=True) or {}
    items = body.get("items")
    if not isinstance(items, list):
        return jsonify({"ok": False, "error": "'items' must be a list"}), 400
    for it in items:
        if not isinstance(it, dict):
            return jsonify({"ok": False, "error": "each item must be an object"}), 400
        sid = it.get("id")
        if not isinstance(sid, str) or not sid.strip():
            return jsonify({"ok": False, "error": "each item needs non-empty 'id'"}), 400
        if "text" not in it or not isinstance(it.get("text"), str):
            return jsonify({"ok": False, "error": "each item needs string 'text' (problem_type)"}), 400

    _reset_idle_timer()

    texts = [str(it.get("text") or "") for it in items]
    ids = [str(it["id"]).strip() for it in items]

    with _model_lock:
        model = _load_model()
        if not texts:
            vecs = np.zeros((0, 1), dtype=np.float64)
        else:
            vecs = _encode_documents(model, texts).astype(np.float64)

    new_cache: dict[str, np.ndarray] = {}
    for i, sid in enumerate(ids):
        if i < vecs.shape[0]:
            new_cache[sid] = vecs[i]
    with _doc_cache_lock:
        _doc_cache = new_cache

    _persist_doc_cache_to_dir(ids, vecs)

    return jsonify({"ok": True, "n": len(new_cache), "doc_cache_dir": _doc_cache_dir})


@app.post("/encode")
def encode():
    """对文本列表编码，返回向量列表（float[][]）。

    请求体：
        ``{"texts": [str, ...], "is_query": bool}``
        - ``is_query=true``  对文本套 instruct 格式后编码（用于 question 侧）。
        - ``is_query=false`` 直接编码（用于 document 侧，默认）。
    """
    body = request.get_json(silent=True) or {}
    texts = body.get("texts")
    if not isinstance(texts, list) or not texts:
        return jsonify({"ok": False, "error": "'texts' must be a non-empty list"}), 400
    if not all(isinstance(t, str) for t in texts):
        return jsonify({"ok": False, "error": "all elements of 'texts' must be strings"}), 400

    is_query = bool(body.get("is_query", False))
    _reset_idle_timer()

    with _model_lock:
        model = _load_model()
        if is_query:
            vecs = _encode_queries(model, texts)
        else:
            vecs = _encode_documents(model, texts)

    return jsonify({"ok": True, "embeddings": vecs.tolist()})


@app.post("/rank")
def rank():
    """用 question 对 candidates 列表按相似度排序，返回排序后的索引。

    请求体：
    ```json
    {
      "question": str,
      "candidates": [
        {"id": str | null, "problem_type": str, "utility": float},
        ...
      ],
      "mode": "embedding" | "hybrid",
      "retrieve_lambda": float,
      "top_k": int | null
    }
    ```
    ``id`` 若与 ``POST /docs/replace`` 中缓存一致，则 document 侧优先用缓存向量。
    返回：
    ```json
    {"ok": true, "ranked_indices": [int, ...]}
    ```
    """
    body = request.get_json(silent=True) or {}
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"ok": False, "error": "missing or empty 'question'"}), 400

    candidates = body.get("candidates")
    if not isinstance(candidates, list):
        return jsonify({"ok": False, "error": "'candidates' must be a list"}), 400
    if len(candidates) == 0:
        return jsonify({"ok": True, "ranked_indices": []})

    mode = str(body.get("mode", "embedding")).lower()
    if mode not in ("embedding", "hybrid"):
        return jsonify({"ok": False, "error": "mode must be 'embedding' or 'hybrid'"}), 400

    try:
        lam = float(body.get("retrieve_lambda", 0.5))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "retrieve_lambda must be a float"}), 400

    top_k = body.get("top_k")
    if top_k is not None:
        try:
            top_k = int(top_k)
            if top_k < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "top_k must be a non-negative int"}), 400

    doc_texts = [str(c.get("problem_type") or "") for c in candidates]
    utilities = np.array([float(c.get("utility", 0.0)) for c in candidates], dtype=np.float64)

    _reset_idle_timer()

    n_c = len(candidates)
    doc_rows: list[np.ndarray | None] = [None] * n_c
    need_idx: list[int] = []
    need_texts: list[str] = []
    for i, c in enumerate(candidates):
        sid = c.get("id")
        if isinstance(sid, str) and sid.strip():
            st = sid.strip()
            with _doc_cache_lock:
                hit = _doc_cache.get(st)
            if hit is not None:
                doc_rows[i] = hit.astype(np.float64)
                continue
            dsk = _load_doc_vec_from_disk(st)
            if dsk is not None:
                doc_rows[i] = dsk
                with _doc_cache_lock:
                    _doc_cache[st] = dsk
                continue
        need_idx.append(i)
        need_texts.append(doc_texts[i])

    with _model_lock:
        model = _load_model()
        q_vec = _encode_queries(model, [question.strip()]).astype(np.float64)[0]
        if need_texts:
            enc = _encode_documents(model, need_texts).astype(np.float64)
            for j, row_i in enumerate(need_idx):
                doc_rows[row_i] = enc[j]

    parts: list[np.ndarray] = []
    for i in range(n_c):
        v = doc_rows[i]
        if v is None:
            return jsonify({"ok": False, "error": "internal: incomplete doc encodings"}), 500
        parts.append(v)
    doc_mat = np.stack(parts, axis=0)

    if doc_mat.size == 0:
        order = list(range(len(candidates)))
    else:
        sims = doc_mat @ q_vec
        if mode == "embedding":
            order_arr = np.argsort(-sims)
        else:
            mixed = (1.0 - lam) * sims + lam * utilities
            order_arr = np.argsort(-mixed)
        order = [int(i) for i in order_arr]

    if top_k is not None:
        order = order[:max(0, top_k)]

    return jsonify({"ok": True, "ranked_indices": order})


# ─── 启动 ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="独立 Embedding 检索服务（vLLM 后端）")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="vLLM 模型名或本地路径",
    )
    p.add_argument(
        "--instruct-task",
        type=str,
        default="Given a question, retrieve relevant skills that help answer it",
        help="Query 侧 instruct 任务描述",
    )
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=300.0,
        help="空闲超时秒数；超过此时间无检索调用则自动卸载模型并退出（默认 300s）",
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
        help="vLLM 显存占用比例 0~1（默认 0.9）；embedding 模型较小可设低值如 0.3",
    )
    p.add_argument(
        "--doc-cache-dir",
        type=str,
        default="",
        help="Document 嵌入落盘目录（*.npy）；空则环境 SE_RETRIEVER_DOC_CACHE_DIR 或 <MEMORY_PATH_DIR>/doc_embed_cache",
    )
    return p.parse_args(argv)


def _resolve_doc_cache_dir(cli_value: str) -> str | None:
    s = (cli_value or "").strip()
    if not s:
        s = (os.environ.get("SE_RETRIEVER_DOC_CACHE_DIR") or os.environ.get("RETRIEVER_DOC_CACHE_DIR") or "").strip()
    if not s:
        mem = (os.environ.get("MEMORY_PATH_DIR") or os.environ.get("SE_MEMORY_DIR") or "").strip()
        if mem:
            s = str(Path(mem) / "doc_embed_cache")
    return s or None


def main(argv: list[str] | None = None) -> None:
    global _model_name, _instruct_task, _idle_timeout, _tensor_parallel_size, _gpu_memory_utilization
    global _doc_cache_dir

    args = _parse_args(argv)
    _model_name = args.embedding_model
    _instruct_task = args.instruct_task
    _idle_timeout = args.idle_timeout
    _tensor_parallel_size = args.tensor_parallel_size
    _gpu_memory_utilization = args.gpu_memory_utilization
    _doc_cache_dir = _resolve_doc_cache_dir(args.doc_cache_dir)

    with _model_lock:
        _load_model()

    _reset_idle_timer()

    print(
        f"[retriever_server] Listening http://{args.host}:{args.port} "
        f"(idle_timeout={_idle_timeout}s) doc_cache_dir={_doc_cache_dir!r}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
