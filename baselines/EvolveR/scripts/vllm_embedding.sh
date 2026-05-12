#!/usr/bin/env bash
# vLLM OpenAI-compatible server for embedding model (BGE-m3) used by experience retrieval.
# Match experience.embedding_model / served name, e.g. bge_m3. Port 8081 matches run_rl default EMBEDDING_API_URL.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
PORT="${EMBED_PORT:-8081}"
export CUDA_VISIBLE_DEVICES="${EMBED_CUDA:-0}"
exec vllm serve "$EMBED_MODEL_PATH" --served-model-name "${SERVED_NAME:-$EMBEDDING_MODEL}" --port "$PORT" \
  --tensor-parallel-size "$GPU_NUM" --max-model-len 8192 --disable-log-requests
