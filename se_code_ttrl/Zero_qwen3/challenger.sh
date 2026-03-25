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
question_reward="$5"
group_question_repetion_penalty="$6"
gen_question_func="$7"
ttrl_train_file="$8"
echo "[exp_name]:${exp_name}"
echo "[challenger_model_path]: ${challenger_model_path}"
echo "[solver_model_path]: ${solver_model_path}"
echo "[challenger_training_steps]: ${challenger_training_steps}"
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

# 从环境变量获取路径配置，如果未设置则使用默认值
project_name="${SE_PROJECT_NAME:-Self-evolving-Agent}"
dir="${SE_BASE_DIR:-/home/ycy/data1}"
model_dir="${SE_MODEL_DIR:-${dir}/models}"
data_dir="${SE_DATA_DIR:-${dir}/data}"
saved_results_dir="${SE_SAVED_RESULTS_DIR:-${dir}/saved_results}"
WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
challenger_path_dir="${SE_CHALLENGER_DIR:-${saved_results_dir}/Challenger}"
solver_path_dir="${SE_SOLVER_DIR:-${saved_results_dir}/Solver}"
tensorboard_dir="${SE_TENSORBOARD_DIR:-${saved_results_dir}/tensorboard_log}"
storage_path=${challenger_path_dir}/${exp_name}
CKPTS_DIR=${storage_path}/ckpts/
tensorboard_path=${tensorboard_dir}/Challenger-${exp_name}
prompt_path="${SE_PROMPT_DIR:-${WORKING_DIR}/${SE_CODE_MODULE:-se_code_auto}}"
mkdir -p ${CKPTS_DIR} ${tensorboard_path}
export TENSORBOARD_DIR=${tensorboard_path}

echo "[路径配置] 工作目录: ${WORKING_DIR}"
echo "[路径配置] Prompt目录: ${prompt_path}"
echo "[路径配置] 存储路径: ${storage_path}"

# 从环境变量获取 GPU 配置，如果未设置则使用默认值
CHALLENGER_GPUS="${SE_CHALLENGER_GPUS:-4,5}"
N_CHALLENGER_GPUS="${SE_N_CHALLENGER_GPUS:-2}"
REWARD_GPUS="${SE_REWARD_GPUS:-6,7}"
REWARD_PORTS="${SE_REWARD_PORTS:-5000,5001}"

echo "[GPU配置] Challenger GPUs: ${CHALLENGER_GPUS} (共 ${N_CHALLENGER_GPUS} 张)"
echo "[GPU配置] Reward GPUs: ${REWARD_GPUS}"
echo "[GPU配置] Reward Ports: ${REWARD_PORTS}"

# 导出环境变量供子脚本使用
export SE_CHALLENGER_GPUS="${CHALLENGER_GPUS}"
export SE_N_CHALLENGER_GPUS="${N_CHALLENGER_GPUS}"
export SE_REWARD_GPUS="${REWARD_GPUS}"
export SE_REWARD_PORTS="${REWARD_PORTS}"

bash ${SCRIPT_DIR}/challenger_reward.sh $solver_model_path 
# 等待vLLM服务启动并跟踪进程
echo "等待vLLM服务启动..."
sleep 10  # 增加等待时间确保服务完全启动



#rollout_query_num=8
rollout_query_num=4
query_top_p=0.99
query_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
kl_loss_coef=0.01
query_temperature=1.0
if [ "$SE_N_GPUS" -eq 4 ]; then
    batch_size=32
else
    batch_size=32
fi
ppo_mini_batch_size=$((batch_size / 4))
micro_batch_size_per_gpu=$((ppo_mini_batch_size / 4))
num_query=$((batch_size * challenger_training_steps))
#num_query=8
tp=1

cd ${WORKING_DIR}
# 启动训练进程并记录PID
echo "启动主训练进程..."
echo "使用 GPU: ${CHALLENGER_GPUS}, 共 ${N_CHALLENGER_GPUS} 张"
TRAINING_PID=""

# 使用环境变量指定的代码模块
CODE_MODULE="${SE_CODE_MODULE:-se_code_auto}"
CUDA_VISIBLE_DEVICES=${CHALLENGER_GPUS} python3 -m ${CODE_MODULE}.main_challenger \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=3072 \
    data.max_response_length=4096 \
    data.num_querys=${num_query} \
    +data.get_prompts_func=${gen_question_func} \
    +data.gen_question_func=${gen_question_func} \
    +data.prompt_path=${prompt_path} \
    data.filter_overlong_prompts=True \
    data.dynamic_topics=False \
    data.truncation='error' \
    +data.ttrl_icl_files=${ttrl_train_file} \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=${rollout_query_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=challenger \
    reward_model.reward_kwargs.storage_path=${storage_path} \
    +reward_model.reward_kwargs.question_reward=${question_reward} \
    +reward_model.reward_kwargs.group_question_repetion_penalty=${group_question_repetion_penalty} \
    +reward_model.reward_kwargs.gen_question_func=${gen_question_func} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="Challenger-${exp_name}" \
    trainer.n_gpus_per_node=${N_CHALLENGER_GPUS} \
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
