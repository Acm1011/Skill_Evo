#!/usr/bin/env bash
# Synthesizer RL 训练脚本
# 用法: ./Synthesizer.sh <exp_name> <synthesizer_model_path> <training_steps> <train_data_file> [额外参数]
#
# 流程:
#   1. 使用 solver_offline_driver 产出的 merged parquet/jsonl 作为训练数据
#   2. Synthesizer 模型接收 skill_generation prompt，输出 JSON skill
#   3. Reward 通过 rollout server 评估 skill 的效果（new_acc - old_acc）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# ============ 解析参数 ============
exp_name="$1"
synthesizer_model_path="$2"
synthesizer_training_steps="$3"
train_data_file="$4"

echo "[exp_name]: ${exp_name}"
echo "[synthesizer_model_path]: ${synthesizer_model_path}"
echo "[synthesizer_training_steps]: ${synthesizer_training_steps}"
echo "[train_data_file]: ${train_data_file}"

# ============ 验证参数 ============
if [ -z "$exp_name" ]; then
    echo "Error: exp_name 不能为空"
    exit 1
fi

if [ -z "$synthesizer_model_path" ]; then
    echo "Error: synthesizer_model_path 不能为空"
    exit 1
fi

if [ -z "$synthesizer_training_steps" ]; then
    echo "Error: synthesizer_training_steps 不能为空"
    exit 1
fi

if ! [[ "$synthesizer_training_steps" =~ ^[0-9]+$ ]]; then
    echo "Error: synthesizer_training_steps 必须是数字，当前值: $synthesizer_training_steps"
    exit 1
fi

if [ ! -d "$synthesizer_model_path" ]; then
    echo "Error: synthesizer_model_path 不存在: $synthesizer_model_path"
    exit 1
fi

if [ ! -f "$train_data_file" ]; then
    echo "Error: train_data_file 不存在: $train_data_file"
    exit 1
fi

# ============ 路径配置 ============
project_name="${SE_PROJECT_NAME:-Skill_Evo}"
dir="${SE_BASE_DIR:-/home/ycy/sdi}"
model_dir="${SE_MODEL_DIR:-${dir}/models}"
data_dir="${SE_DATA_DIR:-${dir}/data}"
saved_results_dir="${SE_SAVED_RESULTS_DIR:-${dir}/skill_saved}"
WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
synthesizer_path_dir="${SE_Synthsizer_DIR:-${WORKING_DIR}/Synthsizer}"
tensorboard_dir="${SE_TENSORBOARD_DIR:-${saved_results_dir}/tensorboard_log}"
storage_path=${synthesizer_path_dir}/${exp_name}
CKPTS_DIR=${storage_path}/ckpts/
tensorboard_path=${tensorboard_dir}/Synthesizer-${exp_name}
mkdir -p ${CKPTS_DIR} ${tensorboard_path}
export TENSORBOARD_DIR=${tensorboard_path}

echo "[路径配置] 工作目录: ${WORKING_DIR}"
echo "[路径配置] 存储路径: ${storage_path}"
echo "[路径配置] 训练数据: ${train_data_file}"

# ============ GPU 配置 ============
# Synthesizer 使用前半 GPU（与 rollout server 后半 GPU 互补）
SYNTHESIZER_GPUS="${SE_SYNTHESIZER_GPUS:-0,1}"
N_SYNTHESIZER_GPUS="${SE_N_SYNTHESIZER_GPUS:-2}"

echo "[GPU配置] Synthesizer GPUs: ${SYNTHESIZER_GPUS} (共 ${N_SYNTHESIZER_GPUS} 张)"

# ============ 训练超参数 ============
rollout_query_num=4
query_top_p=0.99
query_top_k=-1
kl_loss_coef=0.01
query_temperature=1.0
if [ "${SE_N_GPUS:-4}" -eq 4 ]; then
    batch_size=32
else
    batch_size=32
fi
ppo_mini_batch_size=$((batch_size / 4))
micro_batch_size_per_gpu=$((ppo_mini_batch_size / 4))
num_query=$((batch_size * synthesizer_training_steps))
tp=1

# skill 输出较短（JSON 格式），所以 max_response_length 可以适当小一些
max_prompt_length=4096
max_response_length=512

cd ${WORKING_DIR}
echo "启动 Synthesizer RL 训练..."
echo "使用 GPU: ${SYNTHESIZER_GPUS}, 共 ${N_SYNTHESIZER_GPUS} 张"
TRAINING_PID=""

CODE_MODULE="${SE_CODE_MODULE:-skill_src}"

CUDA_VISIBLE_DEVICES=${SYNTHESIZER_GPUS} python3 -m ${CODE_MODULE}.main_synthesizer \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_data_file}" \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.num_querys=${num_query} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${synthesizer_model_path} \
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
    reward_model.reward_manager=synthesizer \
    reward_model.reward_kwargs.storage_path=${storage_path} \
    +reward_model.reward_kwargs.use_skill_type=skill_use_v1 \
    +reward_model.reward_kwargs.random_q_coef=0.5 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="Synthesizer-${exp_name}" \
    trainer.n_gpus_per_node=${N_SYNTHESIZER_GPUS} \
    trainer.total_training_steps=${synthesizer_training_steps} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.save_freq=${synthesizer_training_steps} \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 &

TRAINING_PID=$!

wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "Synthesizer 训练失败，退出码: $TRAINING_EXIT_CODE"
    exit $TRAINING_EXIT_CODE
fi

echo "Synthesizer 训练成功完成"
sleep 5

echo "${exp_name} Synthesizer training finished"
echo "model path: ${CKPTS_DIR}"
