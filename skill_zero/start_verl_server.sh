#!/usr/bin/env bash
# 前台启动 vLLM HTTP 服务（start_verl_server.py）。请在独立终端运行，再于另一终端执行 run_rollout.sh。
#
# 环境变量（均可选，有默认值）：
#   PYTHON
#   PORT                  监听端口
#   MODEL_PATH            模型路径或 HuggingFace id
#   GPU_MEM_UTIL
#   VLLM_TENSOR_PARALLEL_SIZE
#   VLLM_CUDA_VISIBLE_DEVICES   非空时仅对本进程设置 CUDA_VISIBLE_DEVICES，例如 "0" 或 "0,1"
#   VLLM_NO_ENFORCE_EAGER      设为 1 时传入 --no_enforce_eager（更快，但可能触发内存剖析断言失败）
#   VLLM_SHUTDOWN_TOKEN        非空则传入 --shutdown_token；run_rollout.sh 需 export 相同 SHUTDOWN_TOKEN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
PORT="${PORT:-5000}"
MODEL_PATH="${MODEL_PATH:-/home/xzs/data/model/Qwen3-4B-Instruct-2507}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-4,5}"
VLLM_NO_ENFORCE_EAGER="${VLLM_NO_ENFORCE_EAGER:-0}"
VLLM_SHUTDOWN_TOKEN="${VLLM_SHUTDOWN_TOKEN:-}"

EXTRA_ARGS=()
if [ "$VLLM_NO_ENFORCE_EAGER" = "1" ]; then
  EXTRA_ARGS+=(--no_enforce_eager)
fi
if [ -n "$VLLM_SHUTDOWN_TOKEN" ]; then
  EXTRA_ARGS+=(--shutdown_token "$VLLM_SHUTDOWN_TOKEN")
fi

echo "[start_verl_server] PORT=${PORT} MODEL_PATH=${MODEL_PATH}"
if [ -n "$VLLM_CUDA_VISIBLE_DEVICES" ]; then
  echo "[start_verl_server] CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES}"
  exec env CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" "$PYTHON" start_verl_server.py \
    --port "$PORT" \
    --model_path "$MODEL_PATH" \
    --gpu_mem_util "$GPU_MEM_UTIL" \
    --tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    "${EXTRA_ARGS[@]}"
else
  exec "$PYTHON" start_verl_server.py \
    --port "$PORT" \
    --model_path "$MODEL_PATH" \
    --gpu_mem_util "$GPU_MEM_UTIL" \
    --tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    "${EXTRA_ARGS[@]}"
fi
