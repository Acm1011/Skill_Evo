#!/usr/bin/env bash
# GRPO PPO with EvolveR (math, search_experience, no Wiki by default). Uses vendored verl + evolver in this tree.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common_env.sh"

# Do not call external retriever for <search_knowledge> unless explicitly enabled
export EVOLVER_KNOWLEDGE_SEARCH="${EVOLVER_KNOWLEDGE_SEARCH:-0}"
# Math outcome reward (Hendrycks-style + <answer>); set EVOLVER_QA_OUTCOME=1 for legacy QA EM
export EVOLVER_QA_OUTCOME="${EVOLVER_QA_OUTCOME:-0}"

DATA_TRAIN="${TRAIN_FILE}"
DATA_VAL="${VAL_FILE}"
GPU_NUM="${NGPUS_PER_NODE:-8}"

if [ ! -f "$DATA_TRAIN" ]; then
  echo "TRAIN_FILE not found: $DATA_TRAIN" >&2
  echo "Run scripts/prepare_math_parquet.sh first or override TRAIN_FILE." >&2
  exit 1
fi

if [ ! -f "$DATA_VAL" ]; then
  echo "VAL_FILE not found: $DATA_VAL" >&2
  echo "Run scripts/prepare_math_parquet.sh first or override VAL_FILE." >&2
  exit 1
fi

mkdir -p "$EXPERIENCE_EXPORT_DIR/$EXPERIMENT_NAME"
mkdir -p "$CKPTS_DIR"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export MKL_SERVICE_FORCE_INTEL="${MKL_SERVICE_FORCE_INTEL:-1}"
export HYDRA_FULL_ERROR=1

cd "$EVOR"

python3 run_ppo_math.py \
  data.train_files="${DATA_TRAIN}" \
  data.val_files="${DATA_VAL}" \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-128}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-256}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
  data.max_start_length="${MAX_START_LENGTH:-2048}" \
  data.max_obs_length="${MAX_OBS_LENGTH:-2048}" \
  data.shuffle_train_dataloader=true \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH:-128}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH:-32}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP:-1}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.4}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false \
  algorithm.state_masking.mask_sections="['information','experience']" \
  actor_rollout_ref.rollout.n_agent="${N_AGENT:-8}" \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  +trainer.val_only=false \
  +trainer.val_before_train=false \
  trainer.val_do_sample=false \
  trainer.val_temperature=0.6 \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="${GPU_NUM}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.test_freq="${TEST_FREQ:-50}" \
  trainer.project_name="${WAND_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-5}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-1000}" \
  trainer.default_local_dir="${CKPTS_DIR}/${EXPERIMENT_NAME}" \
  rewards.weights.format="${RW_FORMAT:-0.05}" \
  rewards.weights.outcome=1.0 \
  rewards.weights.info_gain=0 \
  rewards.weights.experience="${RW_EXPERIENCE:-0.1}" \
  experience.enable="${USE_EXPERIENCE}" \
  experience.vdb_server_url="${VDB_SERVER_URL}" \
  experience.organize_interval=1 \
  experience.export_interval=50 \
  experience.clean_low_metric_threshold=0.3 \
  experience.clean_interval=10 \
  experience.experience_data_dir="${EXPERIENCE_EXPORT_DIR}" \
  experience.embedding_api_url="${EMBEDDING_API_URL}" \
  experience.trajectory_choice_ratio=0.25 \
  experience.retrieve_component.principle=true \
  experience.retrieve_component.structure=true \
  experience.retrieve_component.success_trajectory=false \
  experience.retrieve_component.failure_trajectory=false \
  max_turns=10 \
  retriever.url="${RETRIEVE_URL}" \
  retriever.topk=3 \
  2>&1 | tee "$EXPERIENCE_EXPORT_DIR/$EXPERIMENT_NAME/train_log.log"
