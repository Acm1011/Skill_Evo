#!/usr/bin/env bash
# 调用 solver_with_skills.py。需保证另一终端已用 start_verl_server.sh 拉起服务且模型加载完成。
#
# 环境变量（与 solver_with_skills.py 对应）：
#   PYTHON, PORT（拼成 --server URL）
#   SOLVER_DATA              题目 jsonl（与 rollout 相同格式）
#   SOLVER_SKILLS_JSONL      skills.jsonl，按 id 与题目对齐
#   SOLVER_N, SOLVER_MAX_TOKENS, SOLVER_TEMPERATURE,
#   SOLVER_TOP_P, SOLVER_TOP_K, SOLVER_OUTPUT_ROOT, SOLVER_RUN_LABEL,
#   SOLVER_OUTPUT_JSONL（可选）, SOLVER_FRESH（设为 1 则 --fresh）,
#   SOLVER_SYSTEM, SOLVER_REQUEST_TIMEOUT
#   SOLVER_NO_TQDM           设为 1 则加 --no_tqdm
#   SOLVER_MODE              direct（默认，按 id 匹配）或 retrieve（嵌入检索 top-k）
#   SOLVER_RETRIEVE_K        retrieve 模式下 top-k
#   EMBEDDING_MODEL          默认 Qwen/Qwen3-Embedding-0.6B
#   EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE（cuda:2 / cpu；仅写数字如 2 时脚本会传给 Python，由 solver 转为 cuda:2）
#
# 任务成功后是否请求本机关闭 vLLM（与 run_rollout.sh 相同变量名）：
#   SHUTDOWN_SERVER_AFTER_ROLLOUT   默认 1；设为 0 则不请求 /shutdown
#   SHUTDOWN_TOKEN
#
# 示例：
#   ./run_solver_with_skills.sh
#   SOLVER_DATA=datas/AIME-24.jsonl SOLVER_N=4 ./run_solver_with_skills.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
PORT="${PORT:-5000}"
SERVER="http://127.0.0.1:${PORT}"

SOLVER_DATA="${SOLVER_DATA:-datas/AIME-24.jsonl}"
SOLVER_SKILLS_JSONL="${SOLVER_SKILLS_JSONL:-runs/skill_induction/AIME-24/20260324_191118_128430/skills.jsonl}"
SOLVER_N="${SOLVER_N:-8}"
SOLVER_MAX_TOKENS="${SOLVER_MAX_TOKENS:-4096}"
SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-1.0}"
SOLVER_TOP_P="${SOLVER_TOP_P:-1.0}"
SOLVER_TOP_K="${SOLVER_TOP_K:-40}"
SOLVER_OUTPUT_ROOT="${SOLVER_OUTPUT_ROOT:-runs/solver_with_skills_retrieval}"
SOLVER_RUN_LABEL="${SOLVER_RUN_LABEL:-}"
SOLVER_OUTPUT_JSONL="${SOLVER_OUTPUT_JSONL:-}"
SOLVER_FRESH="${SOLVER_FRESH:-0}"
SOLVER_SYSTEM="${SOLVER_SYSTEM:-Please reason step by step, and put your final answer within \\boxed{}.}"
SOLVER_REQUEST_TIMEOUT="${SOLVER_REQUEST_TIMEOUT:-3600}"
SOLVER_NO_TQDM="${SOLVER_NO_TQDM:-0}"
SOLVER_MODE="${SOLVER_MODE:-retrieve}"
SOLVER_RETRIEVE_K="${SOLVER_RETRIEVE_K:-3}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/home/xzs/data/model/Qwen3-Embedding-0.6B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:2}"

SHUTDOWN_SERVER_AFTER_ROLLOUT="${SHUTDOWN_SERVER_AFTER_ROLLOUT:-1}"
SHUTDOWN_TOKEN="${SHUTDOWN_TOKEN:-}"

SOLVER_CMD=(
  "$PYTHON" solver_with_skills.py
  --data "$SOLVER_DATA"
  --skills_jsonl "$SOLVER_SKILLS_JSONL"
  --server "$SERVER"
  --n "$SOLVER_N"
  --max_tokens "$SOLVER_MAX_TOKENS"
  --temperature "$SOLVER_TEMPERATURE"
  --top_p "$SOLVER_TOP_P"
  --top_k "$SOLVER_TOP_K"
  --output_root "$SOLVER_OUTPUT_ROOT"
  --system "$SOLVER_SYSTEM"
  --request_timeout "$SOLVER_REQUEST_TIMEOUT"
  --solver_mode "$SOLVER_MODE"
  --retrieve_k "$SOLVER_RETRIEVE_K"
  --embedding_model "$EMBEDDING_MODEL"
  --embedding_batch_size "$EMBEDDING_BATCH_SIZE"
)
if [ -n "$EMBEDDING_DEVICE" ]; then
  SOLVER_CMD+=(--embedding_device "$EMBEDDING_DEVICE")
fi
if [ -n "$SOLVER_RUN_LABEL" ]; then
  SOLVER_CMD+=(--run_label "$SOLVER_RUN_LABEL")
fi
if [ -n "$SOLVER_OUTPUT_JSONL" ]; then
  SOLVER_CMD+=(--output_jsonl "$SOLVER_OUTPUT_JSONL")
fi
if [ "$SOLVER_FRESH" = "1" ]; then
  SOLVER_CMD+=(--fresh)
fi
if [ "$SOLVER_NO_TQDM" = "1" ]; then
  SOLVER_CMD+=(--no_tqdm)
fi

echo "[run_solver_with_skills] 执行: ${SOLVER_CMD[*]}"
"${SOLVER_CMD[@]}"
rc=$?

if [ "$SHUTDOWN_SERVER_AFTER_ROLLOUT" = "1" ] && [ "$rc" -eq 0 ]; then
  echo "[run_solver_with_skills] 任务成功，向 ${SERVER} 发送关机请求 …"
  CURL_ARGS=(-sfS -X POST --max-time 30 "${SERVER}/shutdown")
  if [ -n "$SHUTDOWN_TOKEN" ]; then
    CURL_ARGS+=(-H "X-Shutdown-Token: ${SHUTDOWN_TOKEN}")
  fi
  if curl "${CURL_ARGS[@]}"; then
    echo "[run_solver_with_skills] 已请求 vLLM 服务退出"
  else
    echo "[run_solver_with_skills] 警告: 未能触发关机" >&2
  fi
fi

exit "$rc"
