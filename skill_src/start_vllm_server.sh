#!/usr/bin/env bash

set -euo pipefail

########################################
# 用法:
#   bash start_vllm_server.sh 0,1,2,3
#
# 可选环境变量:
#   MODEL="NousResearch/Meta-Llama-3-8B-Instruct"
#   API_KEY="token-abc123"
#   BASE_PORT=8000
#   DTYPE="auto"
#   WAIT_TIMEOUT=180
########################################

MODEL="${MODEL:-/home/xzs/data/model/Qwen3-4B-Instruct-2507}"
API_KEY="${API_KEY:-}"
BASE_PORT="${BASE_PORT:-8000}"
DTYPE="${DTYPE:-auto}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"

if [ $# -lt 1 ]; then
  echo "用法: bash $0 0,1,2,3"
  exit 1
fi

CUDA_DEVICES_RAW="$1"
mkdir -p logs

IFS=',' read -r -a DEVICES <<< "$CUDA_DEVICES_RAW"

if [ "${#DEVICES[@]}" -eq 0 ]; then
  echo "错误: 没有解析到任何 CUDA 设备"
  exit 1
fi

echo "CUDA 设备列表: ${DEVICES[*]}"
echo "MODEL=$MODEL"
echo "BASE_PORT=$BASE_PORT"
echo

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

echo "========== 启动 vLLM servers =========="

idx=0
for dev in "${DEVICES[@]}"; do
  local port=$((BASE_PORT + idx))
  local log_file="logs/gpu${dev}_port${port}.log"

  echo "在 GPU ${dev} 上启动 vLLM，端口 ${port}"
  CUDA_VISIBLE_DEVICES="${dev}" \
    vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --dtype "${DTYPE}" \
    --api-key "${API_KEY}" \
    > "${log_file}" 2>&1 &

  idx=$((idx + 1))
done

echo
echo "========== 检查服务是否启动成功 =========="

idx=0
for dev in "${DEVICES[@]}"; do
  local port=$((BASE_PORT + idx))
  if wait_for_port "${port}" "${WAIT_TIMEOUT}"; then
    echo "✓ GPU ${dev} 的 vLLM server 启动成功，监听端口 ${port}"
  else
    echo "✗ GPU ${dev} 的 vLLM server 启动失败，端口 ${port} 在 ${WAIT_TIMEOUT} 秒内未就绪"
    echo "  请检查日志: logs/gpu${dev}_port${port}.log"
    exit 1
  fi
  idx=$((idx + 1))
done

echo
echo "========== 全部服务启动成功 =========="