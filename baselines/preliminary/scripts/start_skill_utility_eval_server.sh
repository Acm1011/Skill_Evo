#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SKILLS_RUN_DIR="${SKILLS_RUN_DIR:-${REPO_ROOT}/baselines/preliminary/outputs/source_linked_eval}"
TRAJECTORIES="${TRAJECTORIES:-${REPO_ROOT}/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/baselines/preliminary/outputs/grpo_skill_utility_eval}"
ROLLOUT_LOG_ROOT="${ROLLOUT_LOG_ROOT:-${OUTPUT_DIR}/logs/rollout_servers}"

SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8899}"
QUESTION_SELECTION="${QUESTION_SELECTION:-tail}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
STUDENT_ROLLOUT_N="${STUDENT_ROLLOUT_N:-4}"

EVAL_GPU_IDS="${EVAL_GPU_IDS:-2,3}"
EVAL_N_GPUS="${EVAL_N_GPUS:-2}"
ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT:-8760}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export SE_GPU_IDS="${EVAL_GPU_IDS}"
export SE_N_GPUS="${EVAL_N_GPUS}"
export SE_ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT}"
export SE_ROLLOUT_HOST="${ROLLOUT_HOST}"

python3 -m baselines.preliminary.skill_utility_eval_server \
  --skills-run-dir "${SKILLS_RUN_DIR}" \
  --trajectories "${TRAJECTORIES}" \
  --output-dir "${OUTPUT_DIR}" \
  --rollout-log-root "${ROLLOUT_LOG_ROOT}" \
  --sample-size "${SAMPLE_SIZE}" \
  --question-selection "${QUESTION_SELECTION}" \
  --student-rollout-n "${STUDENT_ROLLOUT_N}" \
  --gpu-ids "${EVAL_GPU_IDS}" \
  --n-gpus "${EVAL_N_GPUS}" \
  --rollout-host "${ROLLOUT_HOST}" \
  --rollout-base-port "${ROLLOUT_BASE_PORT}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  "$@"
