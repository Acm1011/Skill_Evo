#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?please set CHECKPOINT_ROOT}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8899}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/baselines/preliminary/outputs/grpo_skill_utility_eval}"
MIN_STEP="${MIN_STEP:--1}"
MAX_STEP="${MAX_STEP:--1}"
WAIT_FOR_DONE="${WAIT_FOR_DONE:-true}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-0}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ARGS=(
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --server-url "${SERVER_URL}"
  --output-dir "${OUTPUT_DIR}"
  --min-step "${MIN_STEP}"
  --max-step "${MAX_STEP}"
  --wait-timeout "${WAIT_TIMEOUT}"
  --poll-interval "${POLL_INTERVAL}"
)

if [[ "${WAIT_FOR_DONE}" == "true" ]]; then
  ARGS+=(--wait)
fi

python3 -m baselines.preliminary.replay_saved_checkpoints "${ARGS[@]}" "$@"
