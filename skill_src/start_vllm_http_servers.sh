#!/usr/bin/env bash

########################################
# start_vllm_http_servers.sh
#
# 通用：按 GPU 列表启动多个 vLLM OpenAI 兼容 HTTP 服务（vllm serve），每卡一个进程。
# 可用于 offline rollout、独立推理服务等任意需多卡 vLLM HTTP 的场景。
#
# 用法:
#   bash start_vllm_http_servers.sh 0,1,2,3 /path/to/model
#
# 可选环境变量:
#   MODEL="path/to/model"              （第二个位置参数优先于本变量）
#   CUDA_DEVICES_RAW="0,1"            （第一个位置参数优先于本变量）
#   API_KEY="token-abc123"             (vllm serve API key)
#   BASE_PORT=8760                     (第一个 server 端口)
#   DTYPE="auto"                       (vllm serve dtype)
#   WAIT_TIMEOUT=180                   (健康检查超时)
#   SERVED_MODEL_NAME="default"        (vllm serve model name，需与 HTTP 客户端一致)
#   TENSOR_PARALLEL_SIZE=1             (张量并行)
########################################

set -euo pipefail

CUDA_DEVICES_RAW="${1:-${CUDA_DEVICES_RAW:-}}"
MODEL="${2:-${MODEL:-}}"

if [ -z "$MODEL" ] || [ -z "$CUDA_DEVICES_RAW" ]; then
  echo "用法: bash $0 <cuda_devices> <model_path>"
  echo "  cuda_devices: 逗号分隔的 GPU IDs，如 0,1,2,3"
  echo "  model_path: vLLM 模型路径"
  echo ""
  echo "可选环境变量:"
  echo "  BASE_PORT=8760        (默认 8760)"
  echo "  API_KEY=...           (可选 API key)"
  echo "  DTYPE=auto            (默认 auto)"
  echo "  WAIT_TIMEOUT=180      (默认 180 秒)"
  echo "  TENSOR_PARALLEL_SIZE=1 (默认 1)"
  exit 1
fi

API_KEY="${API_KEY:-}"
BASE_PORT="${BASE_PORT:-8760}"
DTYPE="${DTYPE:-auto}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-default}"

mkdir -p logs

# 解析 CUDA 设备列表
IFS=',' read -r -a DEVICES <<< "$CUDA_DEVICES_RAW"

if [ "${#DEVICES[@]}" -eq 0 ]; then
  echo "错误: 没有解析到任何 CUDA 设备"
  exit 1
fi

echo "========== 启动 vLLM HTTP servers =========="
echo "CUDA 设备列表: ${DEVICES[*]}"
echo "模型路径: $MODEL"
echo "BASE_PORT: $BASE_PORT"
echo "DTYPE: $DTYPE"
echo ""

wait_for_port() {
  local port="$1"
  local timeout="$2"
  local elapsed=0

  while [ "$elapsed" -lt "$timeout" ]; do
    if command -v curl >/dev/null 2>&1; then
      if curl -s "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        return 0
      fi
    fi

    if command -v nc >/dev/null 2>&1; then
      if nc -z 127.0.0.1 "$port" >/dev/null 2>&1; then
        return 0
      fi
    fi

    sleep 1
    elapsed=$((elapsed + 1))
  done

  return 1
}

PIDS=()
cleanup() {
  local s=$?
  echo ""
  echo "清理 vLLM HTTP servers..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo "已退出"
  exit "${s:-0}"
}
trap cleanup INT TERM EXIT

# 启动所有 server
echo "启动所有 server..."
idx=0
for dev in "${DEVICES[@]}"; do
  port=$((BASE_PORT + idx))
  log_file="logs/vllm_gpu${dev}_port${port}.log"

  echo "  GPU ${dev} 端口 ${port} -> ${log_file}"

  # 构造 vllm serve 命令
  cmd=(
    vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$port"
    --dtype "$DTYPE"
    --served-model-name "$SERVED_MODEL_NAME"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization 0.95
  )

  if [ -n "$API_KEY" ]; then
    cmd+=(--api-key "$API_KEY")
  fi

  CUDA_VISIBLE_DEVICES="${dev}" "${cmd[@]}" > "$log_file" 2>&1 &
  PIDS+=($!)
  idx=$((idx + 1))
done

echo
echo "========== 检查服务是否启动成功 =========="

# 健康检查
idx=0
failed=0
for dev in "${DEVICES[@]}"; do
  port=$((BASE_PORT + idx))
  if wait_for_port "$port" "$WAIT_TIMEOUT"; then
    echo "✓ GPU ${dev} 的 server 启动成功，监听端口 $port"
  else
    echo "✗ GPU ${dev} 的 server 启动失败，端口 $port 在 ${WAIT_TIMEOUT} 秒内未就绪"
    echo "  请检查日志: logs/vllm_gpu${dev}_port${port}.log"
    failed=$((failed + 1))
  fi
  idx=$((idx + 1))
done

if [ "$failed" -gt 0 ]; then
  echo ""
  echo "错误: $failed 个 server 启动失败"
  exit 1
fi

echo
echo "========== 全部服务启动成功 =========="
echo "服务正在运行，按 Ctrl+C 停止..."
echo

# 等待所有 server 进程
wait
