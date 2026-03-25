#!/usr/bin/env bash
# Challenger训练脚本
# 用法: ./challenger.sh <exp_name> <challenger_model_path> <solver_model_path> <challenger_training_steps> [--no-cleanup]

set -euo pipefail


# 导入资源清理库
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# 启动前强制清理vLLM进程和端口（防止端口冲突）
echo "启动前强制清理vLLM进程和端口..."
force_cleanup_vllm_processes


# 解析参数
exp_name="$1"
challenger_model_path="$2"
solver_model_path="$3"
challenger_training_steps="$4"
topics_path="$5"
echo "[exp_name]:${exp_name}"
echo "[challenger_model_path]: ${challenger_model_path}"
echo "[solver_model_path]: ${solver_model_path}"
echo "[challenger_training_steps]: ${challenger_training_steps}"
echo "[topics_path]: ${topics_path}"
# 验证参数
if [ -z "$exp_name" ]; then
    echo "Error: exp_name 不能为空"
    exit 1
fi

if [ -z "$challenger_model_path" ]; then
    echo "Error: challenger_model_path 不能为空"
    exit 1
fi

if [ -z "$solver_model_path" ]; then
    echo "Error: solver_model_path 不能为空"
    exit 1
fi

if [ -z "$challenger_training_steps" ]; then
    echo "Error: challenger_training_steps 不能为空"
    exit 1
fi

# 验证challenger_training_steps是否为数字
if ! [[ "$challenger_training_steps" =~ ^[0-9]+$ ]]; then
    echo "Error: challenger_training_steps 必须是数字，当前值: $challenger_training_steps"
    exit 1
fi

# 验证模型路径是否存在
if [ ! -d "$challenger_model_path" ]; then
    echo "Error: challenger_model_path 不存在: $challenger_model_path"
    exit 1
fi

if [ ! -d "$solver_model_path" ]; then
    echo "Error: solver_model_path 不存在: $solver_model_path"
    exit 1
fi

# 设置日志文件路径
project_name='Self-evolving-Agent'
WORKING_DIR=/root/users/ycy/Self-evolving-Agent
saved_results_dir=/root/users/ycy/saved_results
challenger_path_dir=${saved_results_dir}/Challenger
solver_path_dir=${saved_results_dir}/Solver
tensorboard_dir=${saved_results_dir}/tensorboard_log
storage_path=${challenger_path_dir}/${exp_name}
CKPTS_DIR=${storage_path}/ckpts/
tensorboard_path=${tensorboard_dir}/Challenger-${exp_name}
mkdir -p ${CKPTS_DIR} ${tensorboard_path}
export TENSORBOARD_DIR=${tensorboard_path}

bash ${SCRIPT_DIR}/challenger_rule_reward.sh $solver_model_path 
# 等待vLLM服务启动并跟踪进程
echo "等待vLLM服务启动..."
sleep 10  # 增加等待时间确保服务完全启动



#rollout_query_num=8
rollout_query_num=4
query_top_p=0.99
query_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
kl_loss_coef=0.01
query_temperature=1.0
batch_size=128
ppo_mini_batch_size=$((batch_size / 4))
micro_batch_size_per_gpu=$((ppo_mini_batch_size / 4))
num_query=$((batch_size * challenger_training_steps))
#num_query=8
tp=1

cd ${WORKING_DIR}
# 启动训练进程并记录PID
echo "启动主训练进程..."
TRAINING_PID=""

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m se_code.main_challenger \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.num_querys=${num_query} \
    data.filter_overlong_prompts=True \
    data.topics_path=$topics_path \
    data.dynamic_topics=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${challenger_model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${query_temperature} \
    actor_rollout_ref.rollout.top_p=${query_top_p} \
    actor_rollout_ref.rollout.top_k="${query_top_k}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=${rollout_query_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=challenger_rule \
    reward_model.reward_kwargs.storage_path=${storage_path} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="Challenger-${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.total_training_steps=${challenger_training_steps} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.save_freq=${challenger_training_steps} \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 &

# 记录训练进程PID
TRAINING_PID=$!

# 等待训练完成
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?


# 检查训练是否成功
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "训练失败，退出码: $TRAINING_EXIT_CODE"
    echo "开始清理进程..."
    exit $TRAINING_EXIT_CODE
fi

echo "训练成功完成"
sleep 10


pkill python

echo "${exp_name} challenger training finished"
echo "model path: ${CKPTS_DIR}/${exp_name}"
