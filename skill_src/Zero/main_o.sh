#!/usr/bin/env bash

set -xeuo pipefail
# 获取脚本所在目录
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SE_RAY_TEMP_ROOT="${SE_RAY_TEMP_ROOT:-/home/ycy/sdi/tmp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# 路径与模型、数据
# =============================================================================
dir="${SE_BASE_DIR:-/home/ycy/sdi/}"
SE_MODEL_DIR="${SE_MODEL_DIR:-${dir}/models}"
SE_DATA_DIR="${SE_DATA_DIR:-${dir}/data}"
export SE_MODEL_DIR SE_DATA_DIR
model_dir="${SE_MODEL_DIR}"
data_dir="${SE_DATA_DIR}"
export data_dir

base_model_name="${SE_BASE_MODEL_NAME:-Qwen3-4B-Instruct-2507}"
base_model_path=${SE_MODEL_DIR}/${base_model_name}
data_name="${SE_DATA_NAME:-DeepMath-103K}"
data_file=${SE_DATA_DIR}/${data_name}.jsonl

project_name="${SE_PROJECT_NAME:-Skill_Evo}"
export project_name
exp_name=data_${data_name}_model_${base_model_name}
export exp_name
variant="${exp_name}"
initial_version=V1
SE_SKILL_SAVED_ROOT="${SE_SKILL_SAVED_ROOT:-/home/ycy/sdi/skill_saved}"

WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
saved_results_dir="${SE_SAVED_RESULTS_DIR:-${SE_SKILL_SAVED_ROOT}/${exp_name}}"
synthesizer_path_dir="${SE_SYNTHESIZER_DIR:-${saved_results_dir}/Synthesizer}"
solver_path_dir="${SE_SOLVER_DIR:-${saved_results_dir}/Solver}"
memory_path_dir="${SE_MEMORY_DIR:-${saved_results_dir}/Memory}"
log_path_dir="${SE_LOG_DIR:-${saved_results_dir}/logs}"
tensorboard_dir="${SE_TENSORBOARD_DIR:-${saved_results_dir}/tensorboard_log}"
TENSORBOARD_PATH="${TENSORBOARD_PATH:-${tensorboard_dir}}"
export TENSORBOARD_PATH tensorboard_dir

mkdir -p ${saved_results_dir} ${synthesizer_path_dir} ${solver_path_dir} ${memory_path_dir} ${log_path_dir} ${tensorboard_dir}
export SYNTHESIZER_PATH_DIR=${synthesizer_path_dir}
export SOLVER_PATH_DIR=${solver_path_dir}
export MEMORY_PATH_DIR=${memory_path_dir}
export LOG_PATH_DIR=${log_path_dir}
export EXP_NAME=${exp_name}
export WORKING_DIR=${WORKING_DIR}
export data_file
export SE_DATA_FILE="${data_file}"

# =============================================================================
# 通用：代码模块
# =============================================================================
export SE_CODE_MODULE="${SE_CODE_MODULE:-skill_src}"

# =============================================================================
# GPU 拓扑（Synthesizer.sh 需要 SE_N_GPUS / SE_GPU_IDS；请与 retriever 用卡错开避免争用）
# =============================================================================
SE_N_GPUS="${SE_N_GPUS:-8}"
SE_GPU_IDS="${SE_GPU_IDS:-0,1,2,3,4,5,6,7}"
export SE_N_GPUS SE_GPU_IDS

# =============================================================================
# Offline Rollout 参数
# =============================================================================
export SE_OFFLINE_ROLLOUT_STEPS="${SE_OFFLINE_ROLLOUT_STEPS:-2}"
export SE_OFFLINE_ROLLOUT_BATCH_SIZE="${SE_OFFLINE_ROLLOUT_BATCH_SIZE:-16}"
export SE_OFFLINE_ROLLOUT_N="${SE_OFFLINE_ROLLOUT_N:-4}"
export SE_OFFLINE_NUM_RANDOM_Q="${SE_OFFLINE_NUM_RANDOM_Q:-4}"
export SE_OFFLINE_SKILL_TYPE="${SE_OFFLINE_SKILL_TYPE:-skill_generation_v1}"

# =============================================================================
# Synthesizer RL 训练超参数
# =============================================================================
synthesizer_training_steps="${SE_SYNTHESIZER_TRAINING_STEPS:-20}"
export synthesizer_training_steps
export SE_SYNTHESIZER_STEPS="${synthesizer_training_steps}"

export SYNTH_BATCH_SIZE="${SYNTH_BATCH_SIZE:-16}"
export SYNTH_ROLLOUT_QUERY_NUM="${SYNTH_ROLLOUT_QUERY_NUM:-4}"
export SYNTH_QUERY_TOP_P="${SYNTH_QUERY_TOP_P:-0.99}"
export SYNTH_QUERY_TOP_K="${SYNTH_QUERY_TOP_K:--1}"
export SYNTH_KL_LOSS_COEF="${SYNTH_KL_LOSS_COEF:-0.01}"
export SYNTH_QUERY_TEMPERATURE="${SYNTH_QUERY_TEMPERATURE:-1.0}"
export SYNTH_TP="${SYNTH_TP:-1}"
export SYNTH_MAX_PROMPT_LENGTH="${SYNTH_MAX_PROMPT_LENGTH:-8192}"
export SYNTH_MAX_RESPONSE_LENGTH="${SYNTH_MAX_RESPONSE_LENGTH:-4096}"
export SYNTH_GPU_MEM_UTIL="${SYNTH_GPU_MEM_UTIL:-0.60}"
export SYNTH_RANDOM_Q_COEF="${SYNTH_RANDOM_Q_COEF:-0.5}"
export SYNTH_USE_SKILL_TYPE="${SYNTH_USE_SKILL_TYPE:-skill_use_v1}"

# =============================================================================
# Solver 训练超参数（供 solver.sh 子进程；tensorboard 路径与 exp 名）
# =============================================================================
solver_retrain_steps="${SE_SOLVER_RETRAIN_STEPS:-40}"
export solver_retrain_steps
solver_batch_size="${SE_SOLVER_BATCH_SIZE:-256}"
rollout_n="${SE_SOLVER_ROLLOUT_N:-4}"
export solver_batch_size rollout_n

# =============================================================================
# 评估与 retriever 服务（memory 同步前需 HTTP 检索；与 start_retriever_server 对齐）
# =============================================================================
solver_eval_step=15
solver_eval_temperature=0.6
solver_eval_num_iter=15

RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8766}"
SE_RETRIEVER_EMBEDDING_MODEL="${SE_RETRIEVER_EMBEDDING_MODEL:-/home/xzs/data/model/Qwen3-Embedding-0.6B}"
RETRIEVER_CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES:-1}"
RETRIEVER_TENSOR_PARALLEL_SIZE="${RETRIEVER_TENSOR_PARALLEL_SIZE:-1}"
RETRIEVER_GPU_MEMORY_UTILIZATION="${RETRIEVER_GPU_MEMORY_UTILIZATION:-0.3}"
RETRIEVER_INSTRUCT_TASK="${RETRIEVER_INSTRUCT_TASK:-Given a question, retrieve relevant skills that help answer it}"
RETRIEVER_IDLE_TIMEOUT="${RETRIEVER_IDLE_TIMEOUT:-300}"
export RETRIEVER_HOST RETRIEVER_PORT
export SE_RETRIEVER_EMBEDDING_MODEL
export RETRIEVER_CUDA_VISIBLE_DEVICES
export RETRIEVER_TENSOR_PARALLEL_SIZE
export RETRIEVER_GPU_MEMORY_UTILIZATION
export RETRIEVER_INSTRUCT_TASK
export RETRIEVER_IDLE_TIMEOUT
export SE_RETRIEVER_URL="http://${RETRIEVER_HOST}:${RETRIEVER_PORT}"

# start_retriever_server.sh 也读 RETRIEVER_* / SE_RETRIEVER_*
RETRIEVER_MAX_WAIT_S="${RETRIEVER_MAX_WAIT_S:-300}"

# memory_func_after_sync：等待检索服务 /health
retriever_memory_sync_after_synth() {
    local ev="$1"
    echo "启动 retriever 服务 (memory 同步用)..."
    bash "${SCRIPT_DIR}/start_retriever_server.sh" &
    local _wait=0
    while [ "${_wait}" -lt "${RETRIEVER_MAX_WAIT_S}" ]; do
        if curl -sf "http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health" > /dev/null 2>&1; then
            echo "  retriever 已就绪 (http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health)"
            break
        fi
        sleep 2
        _wait=$((_wait + 2))
    done
    if [ "${_wait}" -ge "${RETRIEVER_MAX_WAIT_S}" ]; then
        echo "Error: retriever 在 ${RETRIEVER_MAX_WAIT_S}s 内未就绪" >&2
        pkill python 2> /dev/null || true
        return 1
    fi
    echo "更新 skill memory（after Synthesizer, ${ev}）并准备 Solver 数据..."
    bash "${SCRIPT_DIR}/memory_func_after_sync.sh" "${ev}" || {
        pkill python 2> /dev/null || true
        return 1
    }
    echo "关闭 retriever 相关 python 进程 (pkill python)..."
    pkill python 2> /dev/null || true
    return 0
}

function now() {
    date '+%Y-%m-%d-%H-%M'
}
exec > >(tee -a "${log_path_dir}/main_${variant}-$(now).log") 2>&1

cd ${WORKING_DIR}
echo "开始第一轮训练..."
echo "训练 Synthesizer..."
bash ${SCRIPT_DIR}/Synthesizer.sh ${initial_version} ${base_model_path} ${base_model_path} ${synthesizer_training_steps} ${data_file} || {
    echo "Error: 第一轮 Synthesizer 训练失败"
    exit 1
}

retriever_memory_sync_after_synth "${initial_version}" || {
    echo "Error: 第一轮 memory_func_after_sync 失败"
    exit 1
}

echo "训练 Solver..."
bash ${SCRIPT_DIR}/solver.sh ${initial_version} ${base_model_path} ${solver_retrain_steps} || {
    echo " Error: 第一轮 Solver 训练失败"
    exit 1
}

echo "按 Solver reward 更新 skill utility（after Solver）..."
bash ${SCRIPT_DIR}/memory_func_after_solver.sh "${initial_version}" || {
    echo "Error: 第一轮 memory_func_after_solver 失败"
    exit 1
}

for iter in $(seq 2 ${solver_eval_num_iter}); do
    prev=$((iter-1))
    prev_exp_version=V${prev}
    exp_version=V${iter}

    prev_synthesizer_model_path=${synthesizer_path_dir}/${prev_exp_version}/ckpts/global_step_${synthesizer_training_steps}/actor/huggingface
    cur_synthesizer_model_path=${synthesizer_path_dir}/${exp_version}/ckpts/global_step_${synthesizer_training_steps}/actor/huggingface
    prev_solver_model_path=${solver_path_dir}/${prev_exp_version}/ckpts/global_step_${solver_retrain_steps}/actor/huggingface
    cur_solver_model_path=${solver_path_dir}/${exp_version}/ckpts/global_step_${solver_retrain_steps}/actor/huggingface

    echo "开始第 ${iter} 轮训练..."
    echo "训练 Challenger (${exp_name})..."
    bash ${SCRIPT_DIR}/Synthesizer.sh \
        ${exp_version} ${prev_synthesizer_model_path} \
        ${prev_solver_model_path} ${synthesizer_training_steps} ${data_file} || {
        echo "Error: 第 ${iter} 轮 Challenger 训练失败"
        exit 1
    }

    retriever_memory_sync_after_synth "${exp_version}" || {
        echo "Error: 第 ${iter} 轮 memory_func_after_sync 失败"
        exit 1
    }

    echo "训练 Solver (${exp_name})..."
    bash ${SCRIPT_DIR}/solver.sh \
        ${exp_version} ${prev_solver_model_path} ${solver_retrain_steps} || {
        echo "Error: 第 ${iter} 轮 Solver 训练失败"
        exit 1
    }

    echo "按 reward 更新 skill utility（${exp_version}）..."
    bash ${SCRIPT_DIR}/memory_func_after_solver.sh "${exp_version}" || {
        echo "Error: 第 ${iter} 轮 memory_func_after_solver 失败"
        exit 1
    }

    echo "第 ${iter} 轮训练完成"
done

echo "所有训练完成！"
echo "开始评估..."
dataset="${SE_EVAL_DATASET:-AIME24}"
eval_script="${SE_EVAL_SCRIPT:-${WORKING_DIR}/evaluation/eval_single_math_data.sh}"
bash "$eval_script" "${variant}"  "${solver_eval_step}" "${solver_eval_num_iter}" "${base_model_name}" "${dataset}" || {
    echo "Error: 评估失败"
    exit 1
}
