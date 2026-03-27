#!/usr/bin/env bash
# 调用 skill_induction.py：需已启动 start_verl_server.sh（与 rollout 共用同一 HTTP 服务）。
#
# 环境变量（覆盖默认值）：
#   PYTHON, PORT（拼成 --server）
#   ROLLOUTS_JSONL   输入 rollouts.jsonl 路径
#   SKILL_K          每题采样轨迹数 --k
#   SKILL_OUTPUT_ROOT
#   SKILL_RUN_LABEL
#   SKILL_SEED, SKILL_MAX_TOKENS, SKILL_TEMPERATURE, SKILL_REQUEST_TIMEOUT
#   SKILL_NO_TQDM     设为 1 则加 --no_tqdm
#
# 示例：
#   ROLLOUTS_JSONL=runs/rollout/AIME-24/foo/rollouts.jsonl SKILL_K=6 ./run_skill_induction.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
PORT="${PORT:-5000}"
SERVER="http://127.0.0.1:${PORT}"

ROLLOUTS_JSONL="${ROLLOUTS_JSONL:-runs/rollout/AIME-24/20260324_113909_059596/rollouts.jsonl}"
SKILL_K="${SKILL_K:-4}"
SKILL_OUTPUT_ROOT="${SKILL_OUTPUT_ROOT:-runs/skill_induction}"
SKILL_RUN_LABEL="${SKILL_RUN_LABEL:-}"
SKILL_SEED="${SKILL_SEED:-42}"
SKILL_MAX_TOKENS="${SKILL_MAX_TOKENS:-512}"
SKILL_TEMPERATURE="${SKILL_TEMPERATURE:-0.7}"
SKILL_TOP_P="${SKILL_TOP_P:-0.95}"
SKILL_REQUEST_TIMEOUT="${SKILL_REQUEST_TIMEOUT:-600}"
SKILL_NO_TQDM="${SKILL_NO_TQDM:-0}"

CMD=(
  "$PYTHON" skill_induction.py
  --rollouts_jsonl "$ROLLOUTS_JSONL"
  --server "$SERVER"
  --k "$SKILL_K"
  --output_root "$SKILL_OUTPUT_ROOT"
  --seed "$SKILL_SEED"
  --max_tokens "$SKILL_MAX_TOKENS"
  --temperature "$SKILL_TEMPERATURE"
  --top_p "$SKILL_TOP_P"
  --request_timeout "$SKILL_REQUEST_TIMEOUT"
)
if [ -n "$SKILL_RUN_LABEL" ]; then
  CMD+=(--run_label "$SKILL_RUN_LABEL")
fi
if [ "$SKILL_NO_TQDM" = "1" ]; then
  CMD+=(--no_tqdm)
fi

echo "[run_skill_induction] ${CMD[*]}"
exec "${CMD[@]}"
