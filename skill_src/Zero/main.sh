#!/usr/bin/env bash
# =============================================================================
# main.sh - Skill Synthesizer 训练主流程
# =============================================================================
#
# 流程（参考架构图 Stage1）:
#   1.1  用后半 GPU 启动 rollout server，然后调用 solver_offline_driver
#        对训练数据做离线 rollout，产出 merged parquet（含 prompt、raw_q_info、random_q_info）
#   1.2  用前半 GPU 启动 Synthesizer RL 训练（rollout server 保持运行，供 reward 评估使用）
#
# 依赖环境变量（由 run_with_gpus.sh 导出）:
#   SE_N_GPUS / SE_GPU_IDS / SE_CHALLENGER_GPUS / SE_REWARD_GPUS 等
#   SE_ROLLOUT_BASE_PORT / SE_ROLLOUT_SERVER_URLS 等
#   SE_Synthsizer_DIR / SE_ROLLOUT_DIR / SE_MODEL_DIR / SE_DATA_DIR 等
# =============================================================================

set -xeuo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============ 模型与数据配置 ============
base_model_name="${SE_BASE_MODEL_NAME:-Qwen2.5-3B-Instruct}"
base_model_path="${SE_MODEL_DIR}/${base_model_name}"
data_name="${SE_DATA_NAME:-DeepMath-103K}"
data_file="${SE_DATA_DIR}/${data_name}.jsonl"

# Synthesizer 使用的模型（可以和 solver 基座模型不同）
synthesizer_model_name="${SE_SYNTHESIZER_MODEL_NAME:-${base_model_name}}"
synthesizer_model_path="${SE_MODEL_DIR}/${synthesizer_model_name}"

variant="data_${data_name}_model_${base_model_name}"
exp_name="${variant}-V1"

# ============ 训练参数 ============
# offline rollout 参数
offline_rollout_steps="${SE_OFFLINE_ROLLOUT_STEPS:-10}"
offline_rollout_batch_size="${SE_OFFLINE_ROLLOUT_BATCH_SIZE:-32}"
offline_rollout_n="${SE_OFFLINE_ROLLOUT_N:-10}"
offline_num_random_questions="${SE_OFFLINE_NUM_RANDOM_Q:-10}"
offline_skill_type="${SE_OFFLINE_SKILL_TYPE:-skill_generation_v1}"

# Synthesizer 训练参数
synthesizer_training_steps="${SE_SYNTHESIZER_TRAINING_STEPS:-15}"

# ============ 路径 ============
saved_results_dir="${SE_SAVED_RESULTS_DIR}"
synthesizer_dir="${SE_Synthsizer_DIR}"
rollout_dir="${SE_ROLLOUT_DIR}"
tensorboard_dir="${SE_TENSORBOARD_DIR:-${saved_results_dir}/tensorboard_log}"
WORKING_DIR="${SE_WORKING_DIR}"

mkdir -p "${synthesizer_dir}" "${rollout_dir}" "${tensorboard_dir}" "${WORKING_DIR}/logs"

function now() {
    date '+%Y-%m-%d-%H-%M'
}
exec > >(tee -a "${WORKING_DIR}/logs/train_synth_${variant}-$(now).log") 2>&1

cd "${WORKING_DIR}"

# ============ GPU 分配 ============
# 前半 GPU 用于 Synthesizer RL 训练
# 后半 GPU 用于 rollout server（offline rollout + 在线 reward 评估）
HALF=$((SE_N_GPUS / 2))

IFS=',' read -ra ALL_GPUS <<< "${SE_GPU_IDS}"

# rollout server 使用后半 GPU
ROLLOUT_GPU_IDS=""
for ((i=HALF; i<SE_N_GPUS; i++)); do
    if [ -z "$ROLLOUT_GPU_IDS" ]; then
        ROLLOUT_GPU_IDS="${ALL_GPUS[$i]}"
    else
        ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS},${ALL_GPUS[$i]}"
    fi
done

# Synthesizer 使用前半 GPU
SYNTH_GPU_IDS=""
for ((i=0; i<HALF; i++)); do
    if [ -z "$SYNTH_GPU_IDS" ]; then
        SYNTH_GPU_IDS="${ALL_GPUS[$i]}"
    else
        SYNTH_GPU_IDS="${SYNTH_GPU_IDS},${ALL_GPUS[$i]}"
    fi
done

N_ROLLOUT_SERVERS=${HALF}
ROLLOUT_BASE_PORT="${SE_ROLLOUT_BASE_PORT:-8760}"

# 构建 rollout server URL 列表
ROLLOUT_SERVER_URLS=""
ROLLOUT_PORTS=""
for ((i=0; i<N_ROLLOUT_SERVERS; i++)); do
    port=$((ROLLOUT_BASE_PORT + i))
    url="http://127.0.0.1:${port}"
    if [ -z "$ROLLOUT_SERVER_URLS" ]; then
        ROLLOUT_SERVER_URLS="${url}"
        ROLLOUT_PORTS="${port}"
    else
        ROLLOUT_SERVER_URLS="${ROLLOUT_SERVER_URLS} ${url}"
        ROLLOUT_PORTS="${ROLLOUT_PORTS},${port}"
    fi
done

echo "=============================================="
echo "  GPU 分配"
echo "  Rollout Server GPUs: ${ROLLOUT_GPU_IDS} (${N_ROLLOUT_SERVERS} 张)"
echo "  Synthesizer GPUs:    ${SYNTH_GPU_IDS} (${HALF} 张)"
echo "  Rollout Server URLs: ${ROLLOUT_SERVER_URLS}"
echo "=============================================="

# ============ Step 1.1: 启动 rollout server + 离线 rollout ============
echo ""
echo "=============================================="
echo "  Step 1.1: 启动 rollout server 并执行离线 rollout"
echo "=============================================="

# 覆盖环境变量使 start_rollout_servers.sh 只用后半 GPU
export SE_N_GPUS_ORIG="${SE_N_GPUS}"
export SE_N_GPUS="${N_ROLLOUT_SERVERS}"
export SE_GPU_IDS="${ROLLOUT_GPU_IDS}"
export ROLLOUT_SERVER_MODEL="${base_model_path}"
export SE_ROLLOUT_N_SERVERS="${N_ROLLOUT_SERVERS}"
export SE_ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT}"
export SE_ROLLOUT_SERVER_URLS="${ROLLOUT_SERVER_URLS}"
export SE_ROLLOUT_PORTS="${ROLLOUT_PORTS}"

ROLLOUT_SERVER_PID=""
echo "启动 rollout server（后台）..."
bash "${SCRIPT_DIR}/start_rollout_servers.sh" --model "${base_model_path}" &
ROLLOUT_SERVER_PID=$!
echo "Rollout server 后台 PID: ${ROLLOUT_SERVER_PID}"

# 等待 rollout server 健康
echo "等待 rollout server 启动..."
MAX_WAIT=300
WAITED=0
FIRST_PORT=$((ROLLOUT_BASE_PORT))
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://127.0.0.1:${FIRST_PORT}/health" > /dev/null 2>&1; then
        echo "Rollout server 就绪 (等待 ${WAITED}s)"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  仍在等待 rollout server... (${WAITED}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "Error: rollout server 启动超时 (${MAX_WAIT}s)"
    kill $ROLLOUT_SERVER_PID 2>/dev/null || true
    exit 1
fi

# 确保所有 server 就绪
for ((i=0; i<N_ROLLOUT_SERVERS; i++)); do
    port=$((ROLLOUT_BASE_PORT + i))
    echo "  检查 server ${i} (port ${port})..."
    for ((w=0; w<60; w++)); do
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            echo "  server ${i} 就绪"
            break
        fi
        sleep 2
    done
done

# 执行离线 rollout
OFFLINE_WORK_DIR="${rollout_dir}/${exp_name}"
OFFLINE_MERGE_DIR="${OFFLINE_WORK_DIR}/merged"
mkdir -p "${OFFLINE_WORK_DIR}" "${OFFLINE_MERGE_DIR}"

echo "开始离线 rollout..."
echo "  数据文件: ${data_file}"
echo "  steps: ${offline_rollout_steps}, batch_size: ${offline_rollout_batch_size}"
echo "  work_dir: ${OFFLINE_WORK_DIR}"

CODE_MODULE="${SE_CODE_MODULE:-skill_src}"

python3 -m ${CODE_MODULE}.solver_offline_driver run \
    --data-files "${data_file}" \
    --steps "${offline_rollout_steps}" \
    --batch-size "${offline_rollout_batch_size}" \
    --work-dir "${OFFLINE_WORK_DIR}" \
    --merge-output-dir "${OFFLINE_MERGE_DIR}" \
    --merge-prefix "train_data" \
    --skill-type "${offline_skill_type}" \
    --rollout-n "${offline_rollout_n}" \
    --num-random-questions "${offline_num_random_questions}" \
    --model-path "${base_model_path}" \
    --reset-state || {
    echo "Error: 离线 rollout 失败"
    kill $ROLLOUT_SERVER_PID 2>/dev/null || true
    exit 1
}

echo "离线 rollout 完成"

# 训练数据路径
TRAIN_DATA_FILE="${OFFLINE_MERGE_DIR}/train_data.parquet"
if [ ! -f "${TRAIN_DATA_FILE}" ]; then
    TRAIN_DATA_FILE="${OFFLINE_MERGE_DIR}/train_data.jsonl"
fi

if [ ! -f "${TRAIN_DATA_FILE}" ]; then
    echo "Error: 未找到训练数据文件: ${OFFLINE_MERGE_DIR}/train_data.{parquet,jsonl}"
    kill $ROLLOUT_SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "训练数据文件: ${TRAIN_DATA_FILE}"

# ============ Step 1.2: Synthesizer RL 训练 ============
echo ""
echo "=============================================="
echo "  Step 1.2: 启动 Synthesizer RL 训练"
echo "  Rollout server 保持运行供 reward 评估使用"
echo "=============================================="

# 恢复 GPU 数量并设置 Synthesizer GPU
export SE_N_GPUS="${SE_N_GPUS_ORIG}"
export SE_SYNTHESIZER_GPUS="${SYNTH_GPU_IDS}"
export SE_N_SYNTHESIZER_GPUS="${HALF}"

bash "${SCRIPT_DIR}/Synthesizer.sh" \
    "${exp_name}" \
    "${synthesizer_model_path}" \
    "${synthesizer_training_steps}" \
    "${TRAIN_DATA_FILE}" || {
    echo "Error: Synthesizer RL 训练失败"
    kill $ROLLOUT_SERVER_PID 2>/dev/null || true
    exit 1
}

echo ""
echo "=============================================="
echo "  训练完成，清理 rollout server"
echo "=============================================="

# 停止 rollout server
kill $ROLLOUT_SERVER_PID 2>/dev/null || true
wait $ROLLOUT_SERVER_PID 2>/dev/null || true

echo "所有训练完成！"
echo "  Synthesizer 模型: ${SE_Synthsizer_DIR}/${exp_name}/ckpts/"
echo "  Rollout 数据: ${OFFLINE_WORK_DIR}"
