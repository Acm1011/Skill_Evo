#!/usr/bin/env bash
# =============================================================================
# Synthesizer.sh - Synthesizer 完整训练执行器
# =============================================================================
# 用法（由 main.sh 调用，参数和超参数均通过位置参数 + 环境变量传入）:
#   ./Synthesizer.sh <exp_name> <synthesizer_model_path> <solver_model_path> \
#                    <training_steps> <data_file> [<embedding_cache_path>]
#   第 6 个参数可选：embedding cache 目录（含 *.meta.json + *.npz）；省略或与 SE_EMBEDDING_CACHE_PATH 均未设置时为空（随机采样）。
#
# 流程:
#   Step 1. 用后半 GPU 启动 rollout server (start_rollout_servers.sh) + 健康检查
#   Step 2. solver_offline_driver 离线 rollout → merged parquet
#   Step 3. 前半 GPU 做 Synthesizer RL 训练（rollout server 保持运行供 reward 评估）
#   退出时自动清理 rollout server
#
# 超参数环境变量（由 main.sh export）:
#   Offline rollout: SE_OFFLINE_ROLLOUT_STEPS / SE_OFFLINE_ROLLOUT_BATCH_SIZE /
#                    SE_OFFLINE_ROLLOUT_N / SE_OFFLINE_NUM_RANDOM_Q / SE_OFFLINE_SKILL_TYPE
#   RL training:     SYNTH_BATCH_SIZE / SYNTH_ROLLOUT_QUERY_NUM / SYNTH_QUERY_TOP_P /
#                    SYNTH_QUERY_TOP_K / SYNTH_KL_LOSS_COEF / SYNTH_QUERY_TEMPERATURE /
#                    SYNTH_TP / SYNTH_MAX_PROMPT_LENGTH / SYNTH_MAX_RESPONSE_LENGTH /
#                    SYNTH_LR / SYNTH_GPU_MEM_UTIL / SYNTH_RANDOM_Q_COEF / SYNTH_USE_SKILL_TYPE
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# ========================== 解析位置参数 ==========================
exp_name="$1"
synthesizer_model_path="$2"
solver_model_path="$3"
synthesizer_training_steps="$4"
data_file="$5"
# 可选第 6 参；仅当 $6 未设置时才回退 SE_EMBEDDING_CACHE_PATH；显式传空字符串则保持为空（不传 --embedding-cache-path）
embedding_cache_path="${6-${SE_EMBEDDING_CACHE_PATH:-}}"
echo "[exp_name]: ${exp_name}"
echo "[synthesizer_model_path]: ${synthesizer_model_path}"
echo "[solver_model_path]: ${solver_model_path}"
echo "[synthesizer_training_steps]: ${synthesizer_training_steps}"
echo "[data_file]: ${data_file}"
echo "[embedding_cache_path]: ${embedding_cache_path}"

# ========================== 验证参数 ==========================
for var_name in exp_name synthesizer_model_path solver_model_path synthesizer_training_steps data_file; do
    if [ -z "${!var_name}" ]; then
        echo "Error: ${var_name} 不能为空"; exit 1
    fi
done
if ! [[ "$synthesizer_training_steps" =~ ^[0-9]+$ ]]; then
    echo "Error: synthesizer_training_steps 必须是数字，当前值: $synthesizer_training_steps"; exit 1
fi
[ -d "$synthesizer_model_path" ] || { echo "Error: synthesizer_model_path 不存在: $synthesizer_model_path"; exit 1; }
[ -d "$solver_model_path" ]      || { echo "Error: solver_model_path 不存在: $solver_model_path"; exit 1; }
[ -f "$data_file" ]              || { echo "Error: data_file 不存在: $data_file"; exit 1; }

# ========================== 路径配置（仅派生本脚本特有的） ==========================
# SE_PROJECT_NAME / SE_WORKING_DIR / SE_Synthsizer_DIR / SE_ROLLOUT_DIR /
# SE_TENSORBOARD_DIR / SE_CODE_MODULE 等已由 run_with_gpus.sh export
storage_path="${SE_Synthsizer_DIR}/workspace/${exp_name}"
CKPTS_DIR="${SE_Synthsizer_DIR}/ckpts/"
tensorboard_path="${SE_TENSORBOARD_DIR}"
mkdir -p "${CKPTS_DIR}" "${tensorboard_path}" "${storage_path}"
export TENSORBOARD_DIR="${tensorboard_path}"

echo "[路径配置] 工作目录: ${SE_WORKING_DIR}"
echo "[路径配置] 存储路径: ${storage_path}"

# ========================== GPU 分配 ==========================
HALF=$((SE_N_GPUS / 2))
IFS=',' read -ra ALL_GPUS <<< "${SE_GPU_IDS}"

SYNTH_GPU_IDS=""
for ((i=0; i<HALF; i++)); do
    SYNTH_GPU_IDS="${SYNTH_GPU_IDS:+${SYNTH_GPU_IDS},}${ALL_GPUS[$i]}"
done

ROLLOUT_GPU_IDS=""
for ((i=HALF; i<SE_N_GPUS; i++)); do
    ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:+${ROLLOUT_GPU_IDS},}${ALL_GPUS[$i]}"
done

N_SYNTH_GPUS="${HALF}"
N_ROLLOUT_SERVERS="${HALF}"
ROLLOUT_BASE_PORT="${SE_ROLLOUT_BASE_PORT:-8760}"

echo "[GPU配置] Synthesizer GPUs: ${SYNTH_GPU_IDS} (${N_SYNTH_GPUS} 张)"
echo "[GPU配置] Rollout GPUs:     ${ROLLOUT_GPU_IDS} (${N_ROLLOUT_SERVERS} 张)"

# ===========================================================================
# Step 1: 启动 rollout server + 健康检查
# ===========================================================================
echo ""
echo "========== Step 1: 启动 rollout server =========="

export SE_N_GPUS_SAVED="${SE_N_GPUS}"
export SE_GPU_IDS_SAVED="${SE_GPU_IDS}"
export SE_N_GPUS="${N_ROLLOUT_SERVERS}"
export SE_GPU_IDS="${ROLLOUT_GPU_IDS}"
export ROLLOUT_SERVER_MODEL="${solver_model_path}"
export SE_ROLLOUT_N_SERVERS="${N_ROLLOUT_SERVERS}"
export SE_ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT}"
# run_with_gpus.sh 按「整机 SE_N_GPUS」填了 SE_ROLLOUT_SERVER_URLS；此处只起了后半
# N_ROLLOUT_SERVERS 个 server。若不取消，driver/reward 仍会连到未监听端口 → Connection refused。
unset SE_ROLLOUT_SERVER_URLS
unset SE_ROLLOUT_PORTS

ROLLOUT_SERVER_PID=""
cleanup_rollout() {
    if [ -n "${ROLLOUT_SERVER_PID}" ]; then
        echo "清理 rollout server (PID: ${ROLLOUT_SERVER_PID})..."
        kill "${ROLLOUT_SERVER_PID}" 2>/dev/null || true
        wait "${ROLLOUT_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_rollout EXIT

bash "${SCRIPT_DIR}/start_rollout_servers.sh" --model "${solver_model_path}" &
ROLLOUT_SERVER_PID=$!
echo "Rollout server PID: ${ROLLOUT_SERVER_PID}"

MAX_WAIT=300
for ((i=0; i<N_ROLLOUT_SERVERS; i++)); do
    port=$((ROLLOUT_BASE_PORT + i))
    waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            echo "  server ${i} (port ${port}) 就绪 (${waited}s)"
            break
        fi
        sleep 5; waited=$((waited + 5))
    done
    if [ $waited -ge $MAX_WAIT ]; then
        echo "Error: rollout server ${i} (port ${port}) 启动超时"; exit 1
    fi
done

# ===========================================================================
# Step 2: 离线 rollout
# ===========================================================================
echo ""
echo "========== Step 2: 离线 rollout =========="

OFFLINE_WORK_DIR="${storage_path}"
OFFLINE_MERGE_DIR="${OFFLINE_WORK_DIR}/merged"
mkdir -p "${OFFLINE_WORK_DIR}" "${OFFLINE_MERGE_DIR}"

echo "  data_file: ${data_file}"
echo "  steps=${SE_OFFLINE_ROLLOUT_STEPS}  batch=${SE_OFFLINE_ROLLOUT_BATCH_SIZE}  rollout_n=${SE_OFFLINE_ROLLOUT_N}"

cd "${SE_WORKING_DIR}"
offline_cmd=(
    python3 -m "${SE_CODE_MODULE}.solver_offline_driver" run
    --data-files "${data_file}"
    --steps "${SE_OFFLINE_ROLLOUT_STEPS}"
    --batch-size "${SE_OFFLINE_ROLLOUT_BATCH_SIZE}"
    --work-dir "${OFFLINE_WORK_DIR}"
    --merge-output-dir "${OFFLINE_MERGE_DIR}"
    --merge-prefix "train_data"
    --skill-type "${SE_OFFLINE_SKILL_TYPE}"
    --rollout-n "${SE_OFFLINE_ROLLOUT_N}"
    --num-random-questions "${SE_OFFLINE_NUM_RANDOM_Q}"
    --model-path "${solver_model_path}"
    --reset-state
)
if [ -n "$embedding_cache_path" ]; then
    offline_cmd+=(--embedding-cache-path "${embedding_cache_path}")
fi
"${offline_cmd[@]}" || { echo "Error: 离线 rollout 失败"; exit 1; }

echo "离线 rollout 完成"

TRAIN_DATA_FILE="${OFFLINE_MERGE_DIR}/train_data.parquet"
[ -f "${TRAIN_DATA_FILE}" ] || TRAIN_DATA_FILE="${OFFLINE_MERGE_DIR}/train_data.jsonl"
[ -f "${TRAIN_DATA_FILE}" ] || { echo "Error: 未找到训练数据"; exit 1; }
echo "训练数据: ${TRAIN_DATA_FILE}"

# ===========================================================================
# Step 3: Synthesizer RL 训练
# ===========================================================================
echo ""
echo "========== Step 3: Synthesizer RL 训练 =========="

export SE_N_GPUS="${SE_N_GPUS_SAVED}"
export SE_GPU_IDS="${SE_GPU_IDS_SAVED}"

# 从环境变量读取超参数（由 main.sh export）
batch_size="${SYNTH_BATCH_SIZE}"
rollout_query_num="${SYNTH_ROLLOUT_QUERY_NUM}"
query_top_p="${SYNTH_QUERY_TOP_P}"
query_top_k="${SYNTH_QUERY_TOP_K}"
kl_loss_coef="${SYNTH_KL_LOSS_COEF}"
query_temperature="${SYNTH_QUERY_TEMPERATURE}"
tp="${SYNTH_TP}"
max_prompt_length="${SYNTH_MAX_PROMPT_LENGTH}"
max_response_length="${SYNTH_MAX_RESPONSE_LENGTH}"
gpu_mem_util="${SYNTH_GPU_MEM_UTIL}"
random_q_coef="${SYNTH_RANDOM_Q_COEF}"
use_skill_type="${SYNTH_USE_SKILL_TYPE}"

ppo_mini_batch_size=$((batch_size / 4))
micro_batch_size_per_gpu=$((ppo_mini_batch_size / 4))

echo "  GPU: ${SYNTH_GPU_IDS} (${N_SYNTH_GPUS} 张)"
echo "  batch=${batch_size}   kl=${kl_loss_coef}  temp=${query_temperature}"

CUDA_VISIBLE_DEVICES=${SYNTH_GPU_IDS} python3 -m ${SE_CODE_MODULE}.main_synthesizer \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA_FILE}" \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=${rollout_query_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=synthesizer \
    reward_model.reward_kwargs.storage_path=${storage_path} \
    +reward_model.reward_kwargs.use_skill_type=${use_skill_type} \
    +reward_model.reward_kwargs.random_q_coef=${random_q_coef} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.project_name="${SE_PROJECT_NAME}" \
    trainer.experiment_name="Synthesizer-${exp_name}" \
    trainer.n_gpus_per_node=${N_SYNTH_GPUS} \
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

echo ""
echo "=============================================="
echo "  ${exp_name} Synthesizer 训练完成"
echo "  model path: ${CKPTS_DIR}"
echo "=============================================="
