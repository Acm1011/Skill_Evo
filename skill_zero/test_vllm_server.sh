#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/home/xzs/data/model/Qwen3-4B-Instruct-2507}"
API_KEY="${API_KEY:-222}"
BASE_PORT="${BASE_PORT:-8100}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"

if [ $# -lt 1 ]; then
  echo "用法: bash $0 0,1,2,3"
  exit 1
fi

IFS=',' read -r -a DEVICES <<< "$1"

TOTAL=${#DEVICES[@]}
HALF=$((TOTAL / 2))
[ "$HALF" -lt 1 ] && HALF=1

KEEP_DEVICES=("${DEVICES[@]:0:$HALF}")
KILL_DEVICES=("${DEVICES[@]:$HALF}")

echo "全部设备: ${DEVICES[*]}"
echo "保留设备: ${KEEP_DEVICES[*]}"
echo "kill设备: ${KILL_DEVICES[*]}"
echo

########################################
# 等待服务
########################################
wait_ready() {
  port=$1
  for i in $(seq 1 $WAIT_TIMEOUT); do
    if curl -s http://127.0.0.1:$port/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

########################################
# 启动
########################################
echo "=== Step1: 启动全部服务 ==="

for i in "${!DEVICES[@]}"; do
  dev=${DEVICES[$i]}
  port=$((BASE_PORT + i))

  echo "启动 GPU $dev → port $port"

  CUDA_VISIBLE_DEVICES=$dev \
    vllm serve $MODEL \
    --port $port \
    --api-key $API_KEY \
    > logs_gpu${dev}.log 2>&1 &
done

echo
echo "等待服务就绪..."

for i in "${!DEVICES[@]}"; do
  port=$((BASE_PORT + i))
  dev=${DEVICES[$i]}

  if wait_ready $port; then
    echo "GPU $dev 启动成功 (port $port)"
  else
    echo "GPU $dev 启动失败"
    exit 1
  fi
done

########################################
# 推理验证
########################################
echo
echo "=== Step2: 验证全部服务可用 ==="

for i in "${!DEVICES[@]}"; do
  port=$((BASE_PORT + i))
  dev=${DEVICES[$i]}

  if curl -s http://127.0.0.1:$port/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
      "model": "'"$MODEL"'",
      "messages":[{"role":"user","content":"say OK"}],
      "max_tokens":5
    }' | grep -q '"choices"'; then
    echo "GPU $dev 推理正常"
  else
    echo "GPU $dev 推理失败"
    exit 1
  fi
done

########################################
# kill 一半（按端口）
########################################
echo
echo "=== Step3: kill 一半服务 ==="

for i in "${!DEVICES[@]}"; do
  dev=${DEVICES[$i]}
  port=$((BASE_PORT + i))

  if [ "$i" -ge "$HALF" ]; then
    echo "kill GPU $dev (port $port)"

    # 用端口杀
    lsof -ti:$port | xargs -r kill -15
  fi
done

sleep 5

########################################
# 验证 kill 成功
########################################
echo
echo "=== Step4: 验证被 kill 的确实死掉 ==="

for i in "${!DEVICES[@]}"; do
  dev=${DEVICES[$i]}
  port=$((BASE_PORT + i))

  if [ "$i" -ge "$HALF" ]; then
    if curl -s http://127.0.0.1:$port/health >/dev/null 2>&1; then
      echo "❌ GPU $dev 仍然存活（kill失败）"
      exit 1
    else
      echo "✅ GPU $dev 已成功停止"
    fi
  fi
done

########################################
# 验证保留的仍可用
########################################
echo
echo "=== Step5: 验证保留服务仍可用 ==="

for i in "${!DEVICES[@]}"; do
  dev=${DEVICES[$i]}
  port=$((BASE_PORT + i))

  if [ "$i" -lt "$HALF" ]; then
    if curl -s http://127.0.0.1:$port/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $API_KEY" \
      -d '{
        "model": "'"$MODEL"'",
        "messages":[{"role":"user","content":"say OK"}],
        "max_tokens":5
      }' | grep -q '"choices"'; then
      echo "✅ GPU $dev 仍正常工作"
    else
      echo "❌ GPU $dev 异常"
      exit 1
    fi
  fi
done

echo
echo "🎉 测试完成：partial kill 后系统正常"