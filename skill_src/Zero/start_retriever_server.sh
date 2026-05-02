#!/usr/bin/env bash
set -euo pipefail

# 可由 main_o.sh 等通过环境变量覆盖；未设置时与此前脚本内写死值一致
HOST="${RETRIEVER_HOST:-127.0.0.1}"
PORT="${RETRIEVER_PORT:-8766}"
EMBEDDING_MODEL="${SE_RETRIEVER_EMBEDDING_MODEL:-/home/xzs/data/model/Qwen3-Embedding-0.6B}"
CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES:-1}"
TENSOR_PARALLEL_SIZE="${RETRIEVER_TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${RETRIEVER_GPU_MEMORY_UTILIZATION:-0.15}"
INSTRUCT_TASK="${RETRIEVER_INSTRUCT_TASK:-Given a question, retrieve relevant skills that help answer it}"
IDLE_TIMEOUT="${RETRIEVER_IDLE_TIMEOUT:-300}"
# doc 向量缓存目录（emb_*.npy）；优先级：
#   RETRIEVER_DOC_CACHE_DIR / SE_RETRIEVER_DOC_CACHE_DIR >
#   DOC_EMBED_CACHE_DIR（可直接指向 .../Memory/doc_embed_cache）>
#   MEMORY_PATH_DIR/doc_embed_cache
: "${RETRIEVER_DOC_CACHE_DIR:=${SE_RETRIEVER_DOC_CACHE_DIR:-}}"
if [ -z "${RETRIEVER_DOC_CACHE_DIR}" ] && [ -n "${DOC_EMBED_CACHE_DIR:-}" ]; then
  RETRIEVER_DOC_CACHE_DIR="${DOC_EMBED_CACHE_DIR}"
fi
if [ -z "${RETRIEVER_DOC_CACHE_DIR}" ] && [ -n "${MEMORY_PATH_DIR:-}" ]; then
  RETRIEVER_DOC_CACHE_DIR="${MEMORY_PATH_DIR}/doc_embed_cache"
fi

# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# skill_manager/ 的上一级为项目根目录
cd "${SCRIPT_DIR}/.."

CMD=(
    python -m skill_manager.retriever_server
    --host "${HOST}"
    --port "${PORT}"
    --embedding-model "${EMBEDDING_MODEL}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --instruct-task "${INSTRUCT_TASK}"
    --idle-timeout "${IDLE_TIMEOUT}"
)
if [ -n "${RETRIEVER_DOC_CACHE_DIR}" ]; then
  CMD+=(--doc-cache-dir "${RETRIEVER_DOC_CACHE_DIR}")
  export SE_RETRIEVER_DOC_CACHE_DIR="${RETRIEVER_DOC_CACHE_DIR}"
fi

echo "[start_retriever_server] $(date '+%Y-%m-%d %H:%M:%S')"
echo "[start_retriever_server] cmd: ${CMD[*]}"
if [ -n "${RETRIEVER_DOC_CACHE_DIR:-}" ]; then
  echo "[start_retriever_server] doc_embed_cache: ${RETRIEVER_DOC_CACHE_DIR}"
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "[start_retriever_server] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    exec env CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
else
    exec "${CMD[@]}"
fi
