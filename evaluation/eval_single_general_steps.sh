#!/bin/bash
# =============================================================================
# eval_single_general_steps.sh - 评测单个模型多个 checkpoint steps 的通用任务
# =============================================================================

set -euo pipefail
export VLLM_DISABLE_COMPILE_CACHE=1

EXP_NAME="${SB_EXP_NAME:-data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1_skillrl}"
CKPTS_DIR="${SB_CKPTS_DIR:-/home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b/skillrl_grpo_qwen3_4b}"
STEPS="${SB_STEPS:-}"
BASE_MODEL_NAME="${SB_MODEL_NAME:-Qwen3-4B-Instruct-2507}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.1}"
SKIP_BASE_MODEL="${SKIP_BASE_MODEL:-false}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --ckpts_dir)
            CKPTS_DIR="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --base_model_name)
            BASE_MODEL_NAME="$2"
            shift 2
            ;;
        --sample_ratio)
            SAMPLE_RATIO="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --skip_base_model)
            SKIP_BASE_MODEL="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    echo "Error: EXP_NAME 为空"
    exit 1
fi

project_name="${SB_PROJECT_NAME:-Skill_Evo}"
dir="${SB_BASB_DIR:-/home/ycy/sdi}"
model_dir="${SB_MODEL_DIR:-${dir}/models}"
data_dir="${SB_DATA_DIR:-${dir}/data}"
saved_results_dir="${SB_SAVED_RESULTS_DIR:-/home/ycy/sdi/skill_saved/evaluation}"
WORKING_DIR="${SB_WORKING_DIR:-${dir}/${project_name}}"
save_path_dir="${saved_results_dir}/evaluation"
eval_path="${WORKING_DIR}/evaluation"
eval_model_dir="${saved_results_dir}/Solver_ttrl_Base"
eval_saved_path_dir="${save_path_dir}/${EXP_NAME}_temperature${TEMPERATURE}"
base_model_eval_results_dir="${eval_saved_path_dir}/step_0"

if [ -z "$CKPTS_DIR" ]; then
    CKPTS_DIR="${eval_model_dir}/${EXP_NAME}/ckpts"
fi

if [ ! -d "$CKPTS_DIR" ]; then
    echo "Error: Checkpoint 目录不存在: $CKPTS_DIR"
    exit 1
fi

if [ -z "$STEPS" ]; then
    shopt -s nullglob
    step_dirs=("${CKPTS_DIR}"/global_step_*)
    shopt -u nullglob
    if [ ${#step_dirs[@]} -eq 0 ]; then
        echo "Error: 没有找到任何 global_step_* 目录"
        exit 1
    fi
    STEPS=$(printf '%s\n' "${step_dirs[@]}" | sed 's/.*global_step_//' | sort -n | tr '\n' ' ')
fi

mkdir -p "${eval_saved_path_dir}" "${WORKING_DIR}/eval_logs"
cd "${eval_path}"

function now() {
    date '+%Y-%m-%d-%H-%M'
}

exec > >(tee -a "${WORKING_DIR}/eval_logs/eval_general_steps_${EXP_NAME}-$(now).log") 2>&1

GENERAL_EVAL_SCRIPTS=(
    "eval_bbeh_step.py"
    "eval_mmlupro_step.py"
    "eval_supergpqa_step.py"
    "eval_gpqa_step.py"
)

GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))
if [ ${#GPU_QUEUE[@]} -eq 0 ]; then
    echo "Error: No GPUs detected."
    exit 1
fi

declare -A gpu_status
declare -A pids
declare -A gpu_model
declare -A gpu_task

for gpu_id in "${GPU_QUEUE[@]}"; do
    gpu_status["$gpu_id"]="idle"
done

model_list=()
model_paths=()

if [ "$SKIP_BASE_MODEL" != "true" ]; then
    base_model_path="${model_dir}/${BASE_MODEL_NAME}"
    if [ -d "$base_model_path" ]; then
        model_list+=("${BASE_MODEL_NAME}")
        model_paths+=("${base_model_path}")
    else
        echo "Warning: 基础模型不存在，跳过: ${base_model_path}"
    fi
fi

for step in $STEPS; do
    step_name="${EXP_NAME}-step${step}"
    step_path="${CKPTS_DIR}/global_step_${step}/actor/huggingface"
    if [ ! -d "$step_path" ]; then
        echo "Warning: Checkpoint 目录不存在: ${step_path}"
        continue
    fi
    if ls "${step_path}"/*.safetensors >/dev/null 2>&1 || \
       ls "${step_path}"/*.bin >/dev/null 2>&1 || \
       [ -f "${step_path}/model.safetensors.index.json" ] || \
       [ -f "${step_path}/pytorch_model.bin.index.json" ]; then
        model_list+=("${step_name}")
        model_paths+=("${step_path}")
    else
        echo "Warning: Checkpoint 不完整，跳过: ${step_path}"
    fi
done

if [ ${#model_list[@]} -eq 0 ]; then
    echo "Error: 没有找到可评测模型"
    exit 1
fi

echo "=============================================="
echo "  Single Model Multi-Step General Evaluation"
echo "=============================================="
echo "实验名称: ${EXP_NAME}"
echo "Checkpoint 目录: ${CKPTS_DIR}"
echo "评测 Steps: ${STEPS}"
echo "基础模型: ${BASE_MODEL_NAME}"
echo "跳过基础模型: ${SKIP_BASE_MODEL}"
echo "温度标签: ${TEMPERATURE}"
echo "采样比例: ${SAMPLE_RATIO}"
echo "结果保存目录: ${eval_saved_path_dir}"
echo "任务: ${GENERAL_EVAL_SCRIPTS[*]}"
echo "=============================================="

check_dataset_completed() {
    local model_name="$1"
    local eval_script="$2"
    local dataset_name
    dataset_name=$(basename "$eval_script" .py | sed 's/^eval_//' | sed 's/_step$//')
    local step
    step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
    local result_file
    if [ -n "$step" ]; then
        result_file="${eval_saved_path_dir}/step_${step}/${dataset_name}_final_results.json"
    else
        result_file="${base_model_eval_results_dir}/${dataset_name}_final_results.json"
    fi
    [ -f "$result_file" ]
}

get_step_dir_for_model() {
    local model_name="$1"
    local step
    step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
    if [ -n "$step" ]; then
        echo "${eval_saved_path_dir}/step_${step}"
    else
        echo "${base_model_eval_results_dir}"
    fi
}

get_available_gpus() {
    local available=()
    for gpu_id in "${GPU_QUEUE[@]}"; do
        if [ "${gpu_status[$gpu_id]}" = "idle" ]; then
            local memory_used
            local memory_total
            local memory_percent
            local gpu_util
            memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            memory_percent=$((memory_used * 100 / memory_total))
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ' || echo "0")
            if [ "$memory_percent" -lt 5 ] && [ "$gpu_util" -lt 5 ]; then
                available+=("$gpu_id")
            fi
        fi
    done
    echo "${available[@]}"
}

start_eval_job() {
    local gpu_id="$1"
    local model_path="$2"
    local model_name="$3"
    local eval_script="$4"
    local step
    local step_arg=""
    step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
    if [ -n "$step" ]; then
        step_arg="--step ${step}"
    else
        step_arg="--step 0"
    fi

    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting ${eval_script} for ${model_name} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${eval_script}" \
        --model_path "${model_path}" \
        --model_name "${model_name}" \
        --save_path_dir "${eval_saved_path_dir}" \
        --data_path_dir "${data_dir}" \
        --sample_ratio "${SAMPLE_RATIO}" \
        ${step_arg} &
    local pid=$!
    sleep 8
    if kill -0 "$pid" 2>/dev/null; then
        gpu_status["$gpu_id"]="busy"
        pids["$gpu_id"]="$pid"
        gpu_model["$gpu_id"]="$model_name"
        gpu_task["$gpu_id"]="$eval_script"
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    return 1
}

free_gpu() {
    local gpu_id="$1"
    unset pids["$gpu_id"]
    unset gpu_model["$gpu_id"]
    unset gpu_task["$gpu_id"]
    gpu_status["$gpu_id"]="idle"
}

check_completed_jobs() {
    for gpu_id in "${!pids[@]}"; do
        local pid="${pids[$gpu_id]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            local model_name="${gpu_model[$gpu_id]}"
            local eval_script="${gpu_task[$gpu_id]}"
            local exit_code=0
            if ! wait "$pid"; then
                exit_code=$?
            fi
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Finished ${eval_script} for ${model_name} on GPU ${gpu_id} (exit=${exit_code})"
            free_gpu "$gpu_id"
        fi
    done
}

declare -a task_queue=()
for i in "${!model_list[@]}"; do
    for eval_script in "${GENERAL_EVAL_SCRIPTS[@]}"; do
        if check_dataset_completed "${model_list[$i]}" "${eval_script}"; then
            echo "Skip completed: ${model_list[$i]} / ${eval_script}"
            continue
        fi
        task_queue+=("${model_list[$i]}|${model_paths[$i]}|${eval_script}")
    done
done

if [ ${#task_queue[@]} -eq 0 ]; then
    echo "所有通用任务都已完成，无需重新评测。"
else
    echo "待执行任务数: ${#task_queue[@]}"
fi

task_index=0
while [ $task_index -lt ${#task_queue[@]} ] || [ ${#pids[@]} -gt 0 ]; do
    check_completed_jobs

    read -r -a available_gpus <<< "$(get_available_gpus)"
    while [ $task_index -lt ${#task_queue[@]} ] && [ ${#available_gpus[@]} -gt 0 ]; do
        IFS='|' read -r model_name model_path eval_script <<< "${task_queue[$task_index]}"
        if start_eval_job "${available_gpus[0]}" "$model_path" "$model_name" "$eval_script"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Started task $((task_index + 1))/${#task_queue[@]}: ${model_name} / ${eval_script}"
            task_index=$((task_index + 1))
        else
            echo "Warning: 启动失败，稍后重试: ${model_name} / ${eval_script}"
            break
        fi
        read -r -a available_gpus <<< "$(get_available_gpus)"
    done

    if [ ${#pids[@]} -gt 0 ] || [ $task_index -lt ${#task_queue[@]} ]; then
        sleep 5
    fi
done

echo "开始聚合各 step 的通用任务结果..."
for model_name in "${model_list[@]}"; do
    step_dir="$(get_step_dir_for_model "$model_name")"
    python aggregate_general_eval_results_step.py --step_dir "${step_dir}" || true
done

python generate_general_results_table_step.py \
    --exp_name "${EXP_NAME}" \
    --save_path_dir "${eval_saved_path_dir}" \
    --base_model_dir "${base_model_eval_results_dir}" || true

echo "通用任务评测完成。"
