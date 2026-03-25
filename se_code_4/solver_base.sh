#!/usr/bin/env bash
set -euo pipefail

# 加载资源清理函数库
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# 直接获取参数
model_dir=/root/users/ycy/models/shares
model_name=Qwen3-4B-Base
exp_name=weakness_${model_name}
solver_model_path=${model_dir}/${model_name}

project_name='Self-evolving-Agent'
WORKING_DIR=/root/users/ycy/Self-evolving-Agent
saved_results_dir=/root/users/ycy/saved_data2
tb_dir=/root/users/ycy/saved_results
solver_path_dir=${saved_results_dir}/Solver_Base
tensorboard_dir=${tb_dir}/tensorboard_log
storage_path=${solver_path_dir}/${exp_name}
CKPTS_DIR=${storage_path}/ckpts/
data_pool_path=${storage_path}/weakness_data_pool.json
tensorboard_path=${tensorboard_dir}/Solver_Base-${exp_name}
mkdir -p ${CKPTS_DIR} ${tensorboard_path}
export TENSORBOARD_DIR=${tensorboard_path}

adv_estimator=grpo

# DAPO related parameters

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 8))

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
filter_lower=0.25
filter_high=0.75

max_num_gen_batches=5 # 10 个 batch 的数据fileter以后，还没有凑够1个batch
train_prompt_bsz=512
val_batch_size=512 # 验证集 batch size
#gen_prompt_bsz=$((train_prompt_bsz * 2))
n_resp_per_prompt=8
train_prompt_mini_bsz=$((train_prompt_bsz / 2))

# Paths
data_dir=/root/users/ycy/data
TRAIN_FILE=${data_dir}/DeepMath-103K_sampled_12800.parquet
TEST_FILE=${data_dir}/all_math_data.parquet

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

# val
val_temperature=1.0
val_top_p=1.0
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_rollout_n=4


# 记录训练进程PID，以便后续清理
echo "启动训练进程..."
TRAINING_PID=""

cd ${WORKING_DIR}
# 启动训练进程并记录PID
python3 -m se_code.main_solver_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.filter_lower=${filter_lower} \
    algorithm.filter_groups.filter_high=${filter_high} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${solver_model_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_rollout_n} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=solver_base \
    reward_model.reward_kwargs.storage_path=${storage_path} \
    +reward_model.reward_kwargs.data_pool_path=${data_pool_path} \
    reward_model.reward_kwargs.filter_lower=${filter_lower} \
    reward_model.reward_kwargs.filter_high=${filter_high} \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="Solver_Base-${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.total_epochs=4 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto &

TRAINING_PID=$!

# 等待训练完成
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?



# 检查训练是否成功
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "训练失败，退出码: $TRAINING_EXIT_CODE"
    exit $TRAINING_EXIT_CODE
fi

echo "训练成功完成"
sleep 10


echo "${exp_name} solver training finished"