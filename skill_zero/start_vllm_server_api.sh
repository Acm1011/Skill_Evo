#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-5000}"
MODEL_PATH="${MODEL_PATH:-/home/xzs/data/model/Qwen3-4B-Instruct-2507}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-6,7}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
DTYPE="${DTYPE:-bfloat16}"
API_KEY="${API_KEY:-}"

EXTRA_ARGS=()

if [ -n "$SERVED_MODEL_NAME" ]; then
  EXTRA_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [ -n "$MAX_MODEL_LEN" ]; then
  EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [ "$ENFORCE_EAGER" = "1" ]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
  EXTRA_ARGS+=(--enable-prefix-caching)
fi

if [ -n "$API_KEY" ]; then
  EXTRA_ARGS+=(--api-key "$API_KEY")
fi

echo "[start_vllm_server_api] PORT=${PORT} MODEL_PATH=${MODEL_PATH}"
echo "[start_vllm_server_api] TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE} GPU_MEM_UTIL=${GPU_MEM_UTIL}"

if [ -n "$VLLM_CUDA_VISIBLE_DEVICES" ]; then
  echo "[start_vllm_server_api] CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES}"
  exec env CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" \
    vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    "${EXTRA_ARGS[@]}"
else
  exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    "${EXTRA_ARGS[@]}"
fi
