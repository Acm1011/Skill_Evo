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
"""
from __future__ import annotations

import argparse
import os
import threading
import time
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
    return jsonify({
        "ok": True,
        "model_loaded": loaded,
        "model_name": _model_name,
        "idle_timeout": _idle_timeout,
        "idle_remaining": max(0.0, round(_idle_timeout - (time.monotonic() - _last_call_time), 1)),
    })


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
        {"problem_type": str, "utility": float},
        ...
      ],
      "mode": "embedding" | "hybrid",
      "retrieve_lambda": float,
      "top_k": int | null
    }
    ```
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

    with _model_lock:
        model = _load_model()
        # question 套 instruct，documents 直接编码，一次 vLLM 调用批量处理
        all_texts = [_build_query_text(question.strip())] + doc_texts
        all_vecs = _embed(model, all_texts).astype(np.float64)

    q_vec = all_vecs[0]
    doc_mat = all_vecs[1:]

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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global _model_name, _instruct_task, _idle_timeout, _tensor_parallel_size, _gpu_memory_utilization

    args = _parse_args(argv)
    _model_name = args.embedding_model
    _instruct_task = args.instruct_task
    _idle_timeout = args.idle_timeout
    _tensor_parallel_size = args.tensor_parallel_size
    _gpu_memory_utilization = args.gpu_memory_utilization

    with _model_lock:
        _load_model()

    _reset_idle_timer()

    print(
        f"[retriever_server] Listening http://{args.host}:{args.port} "
        f"(idle_timeout={_idle_timeout}s)",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
