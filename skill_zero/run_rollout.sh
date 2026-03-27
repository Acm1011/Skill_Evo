#!/usr/bin/env bash
# 调用 rollout.py。需保证另一终端已用 start_verl_server.sh 拉起服务且模型加载完成。
#
# 环境变量（与 rollout.py 对应）：
#   PYTHON, PORT（拼成 --server URL）
#   ROLLOUT_DATA, ROLLOUT_N, ROLLOUT_MAX_TOKENS, ROLLOUT_TEMPERATURE,
#   ROLLOUT_TOP_P, ROLLOUT_TOP_K, ROLLOUT_OUTPUT_ROOT, ROLLOUT_RUN_LABEL,
#   ROLLOUT_OUTPUT_JSONL（可选，非空则传入 --output_jsonl）,
#   ROLLOUT_FRESH（设为 1 则加 --fresh）,
#   ROLLOUT_SYSTEM, ROLLOUT_REQUEST_TIMEOUT
#
# rollout 全部成功后是否请求本机 vLLM 进程退出（默认开启，避免 GPU 一直占着）：
#   SHUTDOWN_SERVER_AFTER_ROLLOUT   设为 0 则关闭该行为
#   SHUTDOWN_TOKEN                  若服务启动时设置了 --shutdown_token，此处需与之一致（请求头 X-Shutdown-Token）
#
# 示例：
#   ./run_rollout.sh
#   ROLLOUT_N=4 ROLLOUT_DATA=datas/AIME_2020_2024.jsonl ./run_rollout.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
PORT="${PORT:-5000}"
SERVER="http://127.0.0.1:${PORT}"

ROLLOUT_DATA="${ROLLOUT_DATA:-datas/deepscaler.jsonl}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-4096}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-40}"
ROLLOUT_OUTPUT_ROOT="${ROLLOUT_OUTPUT_ROOT:-runs/rollout}"
ROLLOUT_RUN_LABEL="${ROLLOUT_RUN_LABEL:-}"
ROLLOUT_OUTPUT_JSONL="${ROLLOUT_OUTPUT_JSONL:-}"
ROLLOUT_FRESH="${ROLLOUT_FRESH:-0}"
ROLLOUT_SYSTEM="${ROLLOUT_SYSTEM:-Please reason step by step, and put your final answer within \\boxed{}.}"
ROLLOUT_REQUEST_TIMEOUT="${ROLLOUT_REQUEST_TIMEOUT:-3600}"
SHUTDOWN_SERVER_AFTER_ROLLOUT="${SHUTDOWN_SERVER_AFTER_ROLLOUT:-1}"
SHUTDOWN_TOKEN="${SHUTDOWN_TOKEN:-}"

ROLLOUT_CMD=(
  "$PYTHON" rollout.py
  --data "$ROLLOUT_DATA"
  --server "$SERVER"
  --n "$ROLLOUT_N"
  --max_tokens "$ROLLOUT_MAX_TOKENS"
  --temperature "$ROLLOUT_TEMPERATURE"
  --top_p "$ROLLOUT_TOP_P"
  --top_k "$ROLLOUT_TOP_K"
  --output_root "$ROLLOUT_OUTPUT_ROOT"
  --system "$ROLLOUT_SYSTEM"
  --request_timeout "$ROLLOUT_REQUEST_TIMEOUT"
)
if [ -n "$ROLLOUT_RUN_LABEL" ]; then
  ROLLOUT_CMD+=(--run_label "$ROLLOUT_RUN_LABEL")
fi
if [ -n "$ROLLOUT_OUTPUT_JSONL" ]; then
  ROLLOUT_CMD+=(--output_jsonl "$ROLLOUT_OUTPUT_JSONL")
fi
if [ "$ROLLOUT_FRESH" = "1" ]; then
  ROLLOUT_CMD+=(--fresh)
fi

echo "[run_rollout] 执行: ${ROLLOUT_CMD[*]}"
"${ROLLOUT_CMD[@]}"
rc=$?

if [ "$SHUTDOWN_SERVER_AFTER_ROLLOUT" = "1" ] && [ "$rc" -eq 0 ]; then
  echo "[run_rollout] rollout 成功，向 ${SERVER} 发送关机请求 …"
  CURL_ARGS=(-sfS -X POST --max-time 30 "${SERVER}/shutdown")
  if [ -n "$SHUTDOWN_TOKEN" ]; then
    CURL_ARGS+=(-H "X-Shutdown-Token: ${SHUTDOWN_TOKEN}")
  fi
  if curl "${CURL_ARGS[@]}"; then
    echo "[run_rollout] 已请求 vLLM 服务退出（进程应很快结束）"
  else
    echo "[run_rollout] 警告: 未能触发关机（服务未监听、已退出或 token 不匹配）" >&2
  fi
fi

exit "$rc"
