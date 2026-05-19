#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASELINE_DIR="${ROOT_DIR}/baselines/RetroAgent"
UPSTREAM_DIR="${ROOT_DIR}/RetroAgent/rl_trained_self_reflection"

export PYTHONPATH="${BASELINE_DIR}:${UPSTREAM_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"

TRAIN_FILE="${TRAIN_FILE:-${BASELINE_DIR}/outputs/deepmath_rl_train.parquet}"
VAL_FILE="${VAL_FILE:-${BASELINE_DIR}/outputs/deepmath_rl_val.parquet}"
MODEL_PATH="${MODEL_PATH:-~/models/deepseek-llm-7b-chat}"
RUN_NAME="${RUN_NAME:-retroagent_math_single_turn}"
PROJECT_NAME="${PROJECT_NAME:-RetroAgent-Math}"
REFLECTION_FILE="${REFLECTION_FILE:-${BASELINE_DIR}/outputs/${RUN_NAME}_reflections.json}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

mkdir -p "$(dirname "${REFLECTION_FILE}")"

python "${BASELINE_DIR}/run_ppo_math.py" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  critic.ppo_micro_batch_size_per_gpu=4 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.val_before_train=True \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  env.rollout.n="${GROUP_SIZE}" \
  +env.reflection_memory.filepath="${REFLECTION_FILE}" \
  "$@"
