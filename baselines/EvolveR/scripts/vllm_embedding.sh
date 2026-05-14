#!/usr/bin/env bash
# vLLM OpenAI-compatible server for embedding model (BGE-m3) used by experience retrieval.
# Match experience.embedding_model / served name, e.g. bge_m3. Port 8081 matches run_rl default EMBEDDING_API_URL.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
PORT="${EMBED_PORT}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}"
export CUDA_VISIBLE_DEVICES="${EMBED_CUDA}"
exec vllm serve "$EMBED_MODEL_PATH" --served-model-name "${SERVED_NAME}" --port "$PORT" \
  --task embed \
  --tensor-parallel-size "$GPU_NUM" \
  --gpu-memory-utilization "$EMBED_GPU_MEM_UTIL" \
  --max-model-len "$EMBED_MAX_MODEL_LEN" \
  --disable-log-requests
