#!/usr/bin/env bash
# vLLM OpenAI-compatible server for embedding model (BGE-m3) used by experience retrieval.
# Match experience.embedding_model / served name, e.g. bge_m3. Port 8081 matches run_rl default EMBEDDING_API_URL.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
PORT="${EMBED_PORT}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}"
export CUDA_VISIBLE_DEVICES="${EMBED_CUDA}"

COMMON_ARGS=(
  "$EMBED_MODEL_PATH"
  --served-model-name "${SERVED_NAME}"
  --port "$PORT"
  --tensor-parallel-size "$GPU_NUM"
  --gpu-memory-utilization "$EMBED_GPU_MEM_UTIL"
  --max-model-len "$EMBED_MAX_MODEL_LEN"
  --disable-log-requests
)

if vllm serve --help 2>&1 | grep -q -- '--task'; then
  echo "[vllm_embedding] launching via 'vllm serve' with embedding task" >&2
  exec vllm serve "${COMMON_ARGS[@]}" --task embed
fi

if python -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- '--task'; then
  echo "[vllm_embedding] launching via legacy api_server with embedding task" >&2
  exec python -m vllm.entrypoints.openai.api_server "${COMMON_ARGS[@]}" --task embed
fi

echo "[vllm_embedding] current vLLM install has no '--task' flag; launching without it." >&2
echo "[vllm_embedding] if /v1/embeddings still fails, the environment's vLLM build does not support embedding serve." >&2
exec vllm serve "${COMMON_ARGS[@]}"
