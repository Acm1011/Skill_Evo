#!/usr/bin/env bash
# Math GRPO/DAPO-style RL with verl (no agent env). Skills are already in parquet user text.
# Run from anywhere; requires SkillRL checkout with verl at SKILLRL_ROOT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SKILLRL_ROOT="${SKILLRL_ROOT:-$REPO_ROOT/SkillRL}"

TRAIN_FILE="${TRAIN_FILE:?set TRAIN_FILE (parquet from prepare_rl_parquet.sh)}"
TEST_FILE="${TEST_FILE:?set TEST_FILE}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH (SFT merged or base HF path)}"
CKPTS_DIR="${CKPTS_DIR:-$REPO_ROOT/baselines/SkillRL/outputs/rl_ckpts}"

project_name="${PROJECT_NAME:-DeepMathSkillRL}"
exp_name="${EXP_NAME:-grpo-deepmath-skills}"

adv_estimator="${ADV_ESTIMATOR:-grpo}"
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length="${MAX_PROMPT_LENGTH:-$((1024 * 2))}"
max_response_length="${MAX_RESPONSE_LENGTH:-$((1024 * 4))}"
enable_overlong_buffer="${ENABLE_OVERLONG_BUFFER:-True}"
overlong_buffer_len="${OVERLONG_BUFFER_LEN:-$((1024 * 4))}"
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-8}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

sp_size="${SP_SIZE:-1}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}"
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload="${FSDP_OFFLOAD:-True}"
gen_tp="${GEN_TP:-1}"
fsdp_size="${FSDP_SIZE:-1}"

cd "$SKILLRL_ROOT"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.6}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.save_freq="${SAVE_FREQ:-10}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-10}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-200}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations="${LOG_VAL_GEN:-5}"
