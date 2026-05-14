#!/usr/bin/env bash
# DAPO-like (parameter-only) variant based on run_qwen3-8b_GRPO.sh
# Note: this does not switch to recipe.dapo.main_dapo / RayDAPOTrainer.

set -x
CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES

# =============================================================================
# 路径与实验命名
# =============================================================================
WORK_DIR="/home/ycy/sdi/skill_saved/Skill_Evo/baseline"
TENSORBOARD_DIR="/home/ycy/sdi/skill_saved/Skill_Evo/baseline/tensorboard_log/dapo_like_qwen3_4b"
TRAIN_FILES="/home/ycy/sdi/data/train_data.parquet"
VAL_FILES="/home/ycy/sdi/data/temp_data.parquet"
MODEL_PATH="/home/ycy/sdi/models/Qwen3-4B-Instruct-2507"
PROJECT_NAME='verl_dapo_like_qwen3_4b'
EXPERIMENT_NAME='dapo_like_qwen3_4b'

# =============================================================================
# 数据
# =============================================================================
TRAIN_BATCH_SIZE=128
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=4096
FILTER_OVERLONG_PROMPTS=True
DATA_TRUNCATION='error'

# =============================================================================
# Actor / rollout / ref
# =============================================================================
ACTOR_LR="1e-6"
USE_REMOVE_PADDING=True
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=8
USE_KL_LOSS=False
KL_LOSS_COEF=0.0
KL_LOSS_TYPE=low_var_kl
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
ENTROPY_COEFF=0
ENABLE_GRADIENT_CHECKPOINTING=True
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIMIZER_OFFLOAD=False
ROLLOUT_LOG_PROB_MICRO_BATCH_PER_GPU=8
TENSOR_MODEL_PARALLEL_SIZE=2
ROLLOUT_NAME=vllm
ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
ROLLOUT_N=4
REF_LOG_PROB_MICRO_BATCH_PER_GPU=8
REF_PARAM_OFFLOAD=True
USE_KL_IN_REWARD=False

# =============================================================================
# DAPO-like parameters (main_ppo only)
# =============================================================================
REWARD_MANAGER='dapo'
ENABLE_FILTER_GROUPS=True
FILTER_GROUPS_METRIC='acc'
MAX_NUM_GEN_BATCHES=10

# =============================================================================
# Trainer
# =============================================================================
CRITIC_WARMUP=0
TRAINER_LOGGER='["console","wandb","tensorboard"]'
TOTAL_TRAINING_STEPS=200
VAL_BEFORE_TRAIN=False
N_GPUS_PER_NODE=4
NNODES=1
SAVE_FREQ=10
TEST_FREQ=10
TOTAL_EPOCHS=15

# =============================================================================
cd "${WORK_DIR}"
export TENSORBOARD_DIR

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS}" \
    data.truncation="${DATA_TRUNCATION}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
    algorithm.filter_groups.enable="${ENABLE_FILTER_GROUPS}" \
    algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}" \
    algorithm.filter_groups.max_num_gen_batches="${MAX_NUM_GEN_BATCHES}" \
    reward_model.reward_manager="${REWARD_MANAGER}" \
    trainer.critic_warmup="${CRITIC_WARMUP}" \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" "$@"
