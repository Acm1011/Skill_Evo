#!/bin/bash
# =============================================================================
# eval_single_math_data_steps.sh - 评测单个模型的多个 checkpoint steps 在指定数学数据集上的表现
# =============================================================================
#
# 用法:
#   ./eval_single_math_data_steps.sh --exp_name <exp_name> --dataset <dataset> [options]
#
# 参数:
#   --exp_name          实验名称 (必需)
#   --dataset           数据集名称 (必需): AIME24, AIME25, AMC23, MATH500, Minerva, OlympiadBench
#   --ckpts_dir         checkpoint 目录 (可选，默认自动根据 exp_name 计算)
#   --steps             可选：手动指定要评测的 steps (空格分隔)，不指定则自动发现
#   --base_model_name   基础模型名称 (默认: 自动从 exp_name 提取)
#   --skip_base_model   跳过基础模型评测 (默认: false)
#
# 示例:
#   # 评测 AIME24 数据集，自动发现所有 steps
#   ./eval_single_math_data_steps.sh --exp_name ttrl_Qwen2.5-Math-1.5B_AIME24_bsz8_epoch80 --dataset AIME24
#
#   # 手动指定 steps
#   ./eval_single_math_data_steps.sh --exp_name ttrl_Qwen3-4B-Base_bsz128 --dataset AIME24 --steps "20 40 60 80"
#
# =============================================================================

set -euo pipefail
export VLLM_DISABLE_COMPILE_CACHE=1

# =============================================================================
# 参数解析
# =============================================================================

EXP_NAME=""
INPUT_DATASET=""
CKPTS_DIR=""
STEPS=""
BASE_MODEL_NAME="${BASE_MODEL_NAME:-}"
SKIP_BASE_MODEL="${SKIP_BASE_MODEL:-false}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --dataset)
            INPUT_DATASET="$2"
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
        --skip_base_model)
            SKIP_BASE_MODEL="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# =============================================================================
# 参数验证
# =============================================================================

if [ -z "$EXP_NAME" ]; then
    echo "Error: --exp_name is required"
    echo "Usage: $0 --exp_name <exp_name> --dataset <dataset> [options]"
    exit 1
fi

if [ -z "$INPUT_DATASET" ]; then
    echo "Error: --dataset is required"
    echo "Usage: $0 --exp_name <exp_name> --dataset <dataset> [options]"
    exit 1
fi

# 数据集配置
declare -A dataset_n_samples_map=(
    ["AIME24"]=32
    ["AIME25"]=32
    ["AMC23"]=1
    ["MATH500"]=1
    ["Minerva"]=1
    ["OlympiadBench"]=1
)
declare -A dataset_temperature_map=(
    ["AIME24"]=0.6
    ["AIME25"]=0.6
    ["AMC23"]=0.0
    ["MATH500"]=0.0
    ["Minerva"]=0.0
    ["OlympiadBench"]=0.0
)

n_samples=${dataset_n_samples_map[$INPUT_DATASET]:-}
temperature=${dataset_temperature_map[$INPUT_DATASET]:-}

if [ -z "$n_samples" ] || [ -z "$temperature" ]; then
    echo "Error: Unknown dataset '$INPUT_DATASET'. Supported datasets: AIME24, AIME25, AMC23, MATH500, Minerva, OlympiadBench"
    exit 1
fi

# =============================================================================
# 自动从 EXP_NAME 提取 BASE_MODEL_NAME
# =============================================================================

if [ -z "$BASE_MODEL_NAME" ]; then
    # 格式: ttrl_{model_name}_{dataset}_{bsz}_{epoch}
    BASE_MODEL_NAME=$(echo "$EXP_NAME" | cut -d'_' -f2)
    
    if [ -z "$BASE_MODEL_NAME" ]; then
        echo "Error: 无法从 EXP_NAME 提取模型名，请使用 --base_model_name 手动指定"
        exit 1
    fi
    
    echo "自动提取 BASE_MODEL_NAME: $BASE_MODEL_NAME"
fi



project_name="${SE_PROJECT_NAME:-Self-evolving-Agent}"
dir="${SE_BASE_DIR:-/home/ycy/data1}"
model_dir="${SE_MODEL_DIR:-${dir}/models}"
data_dir="${SE_DATA_DIR:-${dir}/data}"
saved_results_dir="${SE_SAVED_RESULTS_DIR:-/home/ycy/data4/ttrl_saved}"
WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
save_path_dir=${saved_results_dir}/evaluation
eval_path=${WORKING_DIR}/evaluation
tb_path_dir=/home/ycy/data3/ttrl_saved/eval_tb_log
eval_model_dir=${saved_results_dir}/Solver_ttrl_Base

# 如果用户没有指定 CKPTS_DIR，则根据 EXP_NAME 自动计算
if [ -z "$CKPTS_DIR" ]; then
    CKPTS_DIR=${eval_model_dir}/${EXP_NAME}/ckpts
    echo "自动计算 CKPTS_DIR: $CKPTS_DIR"
fi

# 验证 checkpoint 目录是否存在
if [ ! -d "$CKPTS_DIR" ]; then
    echo "Error: Checkpoint 目录不存在: $CKPTS_DIR"
    exit 1
fi

# =============================================================================
# 自动发现 Steps
# =============================================================================

if [ -z "$STEPS" ]; then
    echo "自动发现 checkpoint steps..."
    
    STEPS=$(ls -d "${CKPTS_DIR}"/global_step_* 2>/dev/null | \
            sed 's/.*global_step_//' | \
            sort -n | \
            tr '\n' ' ')
    
    if [ -z "$STEPS" ]; then
        echo "Error: 没有找到任何 global_step_* 目录在 $CKPTS_DIR"
        exit 1
    fi
    
    echo "发现的 steps: $STEPS"
fi

# 评测结果保存目录
eval_saved_path_dir=${save_path_dir}/${EXP_NAME}_data${INPUT_DATASET}_temperature${temperature}_n_samples${n_samples}

mkdir -p "${eval_saved_path_dir}" "${tb_path_dir}" "${WORKING_DIR}/eval_logs"

cd "${eval_path}"

# =============================================================================
# 日志设置
# =============================================================================

function now() {
    date '+%Y-%m-%d-%H-%M'
}

exec > >(tee -a "${WORKING_DIR}/eval_logs/eval_${EXP_NAME}_${INPUT_DATASET}-$(now).log") 2>&1

echo "=============================================="
echo "  Single Model Multi-Step Math Evaluation"
echo "=============================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "配置:"
echo "  实验名称:      $EXP_NAME"
echo "  数据集:        $INPUT_DATASET"
echo "  n_samples:     $n_samples"
echo "  temperature:   $temperature"
echo "  Checkpoint 目录: $CKPTS_DIR"
echo "  评测 Steps:    $STEPS"
echo "  基础模型:      $BASE_MODEL_NAME"
echo "  跳过基础模型:  $SKIP_BASE_MODEL"
echo "  结果保存目录:  $eval_saved_path_dir"
echo "=============================================="
echo ""

# =============================================================================
# GPU 初始化
# =============================================================================

GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))

if [ ${#GPU_QUEUE[@]} -eq 0 ]; then
    echo "Error: No GPUs detected."
    exit 1
fi

echo "Available GPUs: ${GPU_QUEUE[@]} (Total: ${#GPU_QUEUE[@]})"

# =============================================================================
# 全局跟踪变量
# =============================================================================

declare -A gpu_status
declare -A pids
declare -A model_gpu_mapping
declare -A model_path_mapping
declare -A task_completed
declare -A gpu_dataset

for gpu_id in "${GPU_QUEUE[@]}"; do
    gpu_status["$gpu_id"]="idle"
done

# =============================================================================
# 构建模型列表 (单个模型的多个 steps)
# =============================================================================

model_list=()
model_paths=()

# Base model 评测结果保存在全局目录
base_model_eval_results_dir=${save_path_dir}/${BASE_MODEL_NAME}
base_model_path="${model_dir}/${BASE_MODEL_NAME}"
BASE_MODEL_NEEDS_EVAL="false"

# 检查是否需要评测基础模型
if [ "$SKIP_BASE_MODEL" != "true" ]; then
    if [ ! -d "${base_model_path}" ]; then
        echo "Warning: 基础模型不存在: ${base_model_path}，跳过..."
    elif [ -f "${base_model_eval_results_dir}/${INPUT_DATASET}_Overall_results.jsonl" ]; then
        echo "基础模型 ${BASE_MODEL_NAME} 在数据集 ${INPUT_DATASET} 上的评测结果已存在"
    else
        echo "基础模型 ${BASE_MODEL_NAME} 需要评测"
        BASE_MODEL_NEEDS_EVAL="true"
        model_list+=("${BASE_MODEL_NAME}")
        model_paths+=("${base_model_path}")
    fi
else
    echo "跳过基础模型评测 (--skip_base_model)"
fi

# 添加各个 step 的 checkpoint
valid_steps=0
invalid_steps=0

for step in $STEPS; do
    step_name="${EXP_NAME}-step${step}"
    step_path="${CKPTS_DIR}/global_step_${step}/actor/huggingface"
    
    if [ -d "$step_path" ]; then
        if ls "${step_path}"/*.safetensors >/dev/null 2>&1 || ls "${step_path}"/*.bin >/dev/null 2>&1; then
            model_list+=("$step_name")
            model_paths+=("$step_path")
            echo "✓ 添加 checkpoint: $step_name -> $step_path"
            valid_steps=$((valid_steps + 1))
        else
            echo "✗ Warning: Checkpoint 不完整 (无模型文件): $step_path"
            invalid_steps=$((invalid_steps + 1))
        fi
    else
        echo "✗ Warning: Checkpoint 目录不存在: $step_path"
        invalid_steps=$((invalid_steps + 1))
    fi
done

echo ""
echo "=============================================="
echo "模型统计:"
echo "  有效 checkpoints: $valid_steps"
echo "  无效/跳过: $invalid_steps"
echo "  总模型数量: ${#model_list[@]}"
if [ ${#model_list[@]} -gt 0 ]; then
    echo ""
    echo "最终模型列表:"
    for i in "${!model_list[@]}"; do
        echo "  [$((i+1))] ${model_list[$i]}"
    done
fi
echo "=============================================="
echo ""

if [ ${#model_list[@]} -eq 0 ]; then
    echo "Error: 没有找到任何可评测的模型"
    exit 1
fi

# =============================================================================
# 辅助函数
# =============================================================================

get_available_gpus() {
    local available_gpus=()
    for gpu_id in "${GPU_QUEUE[@]}"; do
        if [ "${gpu_status[$gpu_id]}" = "idle" ]; then
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            if [ "$memory_percent" -lt 5 ] && [ "$gpu_util" -lt 5 ]; then
                available_gpus+=("$gpu_id")
            fi
        fi
    done
    echo "${available_gpus[@]}"
}

check_main_task_completed() {
    local model_name="$1"
    local dataset="$2"
    
    # 从 model_name 提取 step（格式：xxx-stepN）
    local step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
    
    local result_file
    if [ -n "$step" ]; then
        # 新目录结构: step_N/{dataset}_Overall_results.jsonl
        result_file="${eval_saved_path_dir}/step_${step}/${dataset}_Overall_results.jsonl"
    elif [ "$model_name" == "$BASE_MODEL_NAME" ]; then
        # Base model 保存在全局目录
        result_file="${base_model_eval_results_dir}/${dataset}_Overall_results.jsonl"
    else
        # 兼容旧目录结构
        result_file="${eval_saved_path_dir}/${model_name}/${dataset}_Overall_results.jsonl"
    fi
    
    if [ -f "$result_file" ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Main task [${dataset}] for model [${model_name}] already completed"
        return 0
    else
        return 1
    fi
}

start_math_task_job() {
    local gpu_id="$1"
    local model_path="$2"
    local model_name="$3"
    local dataset="$4"
    
    # 从 model_name 提取 step（格式：xxx-stepN）
    local step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
    local step_arg=""
    local target_save_dir="${eval_saved_path_dir}"
    
    if [ -n "$step" ]; then
        step_arg="--step ${step}"
    elif [ "$model_name" == "$BASE_MODEL_NAME" ]; then
        # Base model 保存在全局目录（不带 step 参数）
        target_save_dir="${save_path_dir}"
    fi
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting math task [${dataset}] for model [${model_name}] (step=${step:-base}) on GPU [${gpu_id}]"
    
    CUDA_VISIBLE_DEVICES="${gpu_id}" python eval_all_math_step.py \
        --model_path "${model_path}" \
        --model_name "${model_name}" \
        --dataset "${dataset}" \
        --save_path_dir "${target_save_dir}" \
        --n_samples "${n_samples}" \
        --temperature "${temperature}" \
        --data_path_dir "${data_dir}" \
        ${step_arg} &
    local pid=$!
    
    sleep 10
    
    if kill -0 "$pid" 2>/dev/null; then
        gpu_status["${gpu_id}"]="busy"
        pids["${gpu_id}"]="$pid"
        model_gpu_mapping["${gpu_id}"]="${model_name}"
        model_path_mapping["${gpu_id}"]="${model_path}"
        gpu_dataset["${gpu_id}"]="${dataset}"
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started math task [${dataset}] for model [${model_name}]"
        return 0
    else
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Error: Failed to start math task [${dataset}] for model [${model_name}]"
        kill -TERM "$pid" 2>/dev/null || true
        return 1
    fi
}

free_gpu() {
    local gpu_id="$1"
    local model_name="${model_gpu_mapping[$gpu_id]:-unknown}"
    
    unset pids["$gpu_id"]
    gpu_status["$gpu_id"]="idle"
    unset model_gpu_mapping["$gpu_id"]
    unset model_path_mapping["$gpu_id"]
    unset task_completed["$gpu_id"]
    unset gpu_dataset["$gpu_id"]
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Freed GPU [${gpu_id}] (was running model [${model_name}])"
}

cleanup_zombie_processes() {
    local orphaned_pids=()
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            local pid=$(echo "$line" | awk '{print $2}')
            local gpu_used=$(echo "$line" | grep -o 'CUDA_VISIBLE_DEVICES=[0-9]*' | cut -d'=' -f2)
            
            local is_tracked=false
            for tracked_gpu in "${!pids[@]}"; do
                if [ "${pids[$tracked_gpu]}" = "$pid" ]; then
                    is_tracked=true
                    break
                fi
            done
            
            if [ "$is_tracked" = false ] && [ -n "$gpu_used" ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Found orphaned process PID [${pid}] on GPU [${gpu_used}]"
                orphaned_pids+=("$pid")
            fi
        fi
    done < <(ps aux | grep -E "eval_all_math_step.py" | grep -v grep)
    
    for pid in "${orphaned_pids[@]}"; do
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Killing orphaned process PID [${pid}]"
        kill -TERM "$pid" 2>/dev/null || true
    done
}

comprehensive_cleanup() {
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Performing comprehensive GPU state cleanup..."
    
    for gpu_id in "${GPU_QUEUE[@]}"; do
        if nvidia-smi -i "$gpu_id" >/dev/null 2>&1; then
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            if [ "${gpu_status[$gpu_id]}" = "busy" ] && [ "$memory_percent" -lt 5 ] && [ "$gpu_util" -lt 5 ]; then
                local model_name="${model_gpu_mapping[$gpu_id]}"
                local pid="${pids[$gpu_id]}"
                
                if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] process finished for model [${model_name}]"
                else
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] marked as busy but has low usage (${memory_percent}%), forcing cleanup..."
                    free_gpu "$gpu_id"
                fi
            fi
        fi
    done
    
    cleanup_zombie_processes
}

check_completed_jobs() {
    for gpu_id in "${!pids[@]}"; do
        pid="${pids[$gpu_id]}"
        
        if ! kill -0 "$pid" 2>/dev/null; then
            local model_name="${model_gpu_mapping[$gpu_id]}"
            
            if [ -z "${task_completed[$gpu_id]:-}" ]; then
                local dataset="${gpu_dataset[$gpu_id]}"
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [DONE] GPU [${gpu_id}] finished: [${dataset}] for [${model_name}] (PID ${pid})"
                task_completed["$gpu_id"]=1
                
                # 从 model_name 提取 step（格式：xxx-stepN）
                local step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
                local step_arg=""
                local target_save_dir="${eval_saved_path_dir}"
                
                if [ -n "$step" ]; then
                    step_arg="--step ${step}"
                elif [ "$model_name" == "$BASE_MODEL_NAME" ]; then
                    target_save_dir="${save_path_dir}"
                fi
                
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Running post_eval_step for [${model_name}] (step=${step:-base}) on dataset [${dataset}]..."
                python post_eval_step.py \
                    --save_path_dir "${target_save_dir}" \
                    --dataset "${dataset}" \
                    --model_name "${model_name}" \
                    --n_samples "${n_samples}" \
                    --temperature "${temperature}" \
                    ${step_arg}
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [POST] Post-processing completed for [${model_name}] on [${dataset}]"
                
                free_gpu "$gpu_id"
            fi
        fi
    done
    
    # 检测卡住的进程
    for gpu_id in "${!pids[@]}"; do
        pid="${pids[$gpu_id]}"
        
        if kill -0 "$pid" 2>/dev/null; then
            local model_name="${model_gpu_mapping[$gpu_id]}"
            
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            if [ "$memory_percent" -lt 2 ] && [ "$memory_used" -lt 100 ] && [ "$gpu_util" -lt 5 ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Warning: GPU [${gpu_id}] has very low memory usage (${memory_percent}%) and utilization (${gpu_util}%) for model [${model_name}]"
                
                local cpu_usage=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
                if [ "$cpu_usage" = "0.0" ] || [ -z "$cpu_usage" ]; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Detected stuck process PID [${pid}] on GPU [${gpu_id}], terminating..."
                    kill -TERM "$pid" 2>/dev/null || true
                    sleep 2
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -KILL "$pid" 2>/dev/null || true
                    fi
                    free_gpu "$gpu_id"
                fi
            fi
        fi
    done
}

# =============================================================================
# 主执行循环 - 数学任务
# =============================================================================

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting math tasks for dataset: ${INPUT_DATASET}"

declare -a math_task_queue=()

for i in "${!model_list[@]}"; do
    model_name="${model_list[$i]}"
    model_path="${model_paths[$i]}"
    
    if [ ! -d "$model_path" ]; then
        continue
    fi
    
    if check_main_task_completed "$model_name" "$INPUT_DATASET"; then
        continue
    fi
    math_task_queue+=("${model_name}|${model_path}|${INPUT_DATASET}")
done

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Math task queue: ${#math_task_queue[@]} tasks"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Task queue details:"
for i in "${!math_task_queue[@]}"; do
    echo "  [$i] ${math_task_queue[$i]}"
done

if [ ${#math_task_queue[@]} -eq 0 ]; then
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] No math tasks to run"
else

cleanup_counter=0
math_task_index=0

while [ $math_task_index -lt ${#math_task_queue[@]} ] || [ ${#pids[@]} -gt 0 ]; do
    cleanup_counter=$((cleanup_counter + 1))
    if [ $((cleanup_counter % 10)) -eq 0 ]; then
        comprehensive_cleanup
    else
        cleanup_zombie_processes
    fi
    
    check_completed_jobs
    available_gpus=($(get_available_gpus))
    
    while [ $math_task_index -lt ${#math_task_queue[@]} ] && [ ${#available_gpus[@]} -ge 1 ]; do
        task="${math_task_queue[$math_task_index]}"
        IFS='|' read -r model_name model_path dataset <<< "$task"
        
        if check_main_task_completed "$model_name" "$dataset"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] Math task $((math_task_index + 1))/${#math_task_queue[@]}: [${dataset}] for [${model_name}] already completed"
            math_task_index=$((math_task_index + 1))
            continue
        fi
        
        if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]]; then
            if start_math_task_job "${available_gpus[0]}" "$model_path" "$model_name" "$dataset"; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [START] Math task $((math_task_index + 1))/${#math_task_queue[@]}: [${dataset}] for [${model_name}] on GPU ${available_gpus[0]}"
                math_task_index=$((math_task_index + 1))
                available_gpus=($(get_available_gpus))
            else
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [RETRY] Math task $((math_task_index + 1))/${#math_task_queue[@]}: [${dataset}] for [${model_name}] failed to start, will retry"
                break
            fi
        else
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [WARN] Invalid GPU ID, skipping math task $((math_task_index + 1))/${#math_task_queue[@]}"
            math_task_index=$((math_task_index + 1))
        fi
    done
    
    if [ ${#pids[@]} -gt 0 ] || [ $math_task_index -lt ${#math_task_queue[@]} ]; then
        completed=$((math_task_index - ${#pids[@]}))
        total_tasks=${#math_task_queue[@]}
        progress=0
        if [ $total_tasks -gt 0 ]; then
            progress=$((completed * 100 / total_tasks))
        fi
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [STATUS] Math tasks: Running ${#pids[@]} jobs, Completed ${completed}/${total_tasks} (${progress}%), Pending $((total_tasks - math_task_index))"
        sleep 30
    fi
done

fi

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [COMPLETE] All math tasks completed! (${#math_task_queue[@]} tasks)"

# =============================================================================
# 上传 TensorBoard
# =============================================================================

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Uploading evaluation results to TensorBoard..."

# 确定 base model 的评测结果目录（作为 step 0）
base_model_arg=""
if [ "$SKIP_BASE_MODEL" != "true" ] && [ -d "${base_model_eval_results_dir}" ]; then
    base_model_arg="--base_model_dir ${base_model_eval_results_dir}"
    echo "Base model 结果目录 (step 0): ${base_model_eval_results_dir}"
fi


python tb_single_math_step.py \
    --exp_name="${EXP_NAME}" \
    --dataset="${INPUT_DATASET}" \
    --save_path_dir="${eval_saved_path_dir}" \
    --tb_path_dir="${tb_path_dir}" \
    ${base_model_arg}

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All evaluations completed successfully!"
echo ""
echo "=============================================="
echo "评测完成！"
echo "  实验名称: ${EXP_NAME}"
echo "  数据集: ${INPUT_DATASET}"
echo "  结果目录: ${eval_saved_path_dir}"
echo "  TensorBoard: ${tb_path_dir}"
if [ -n "$base_model_arg" ]; then
    echo "  Base model (step 0): ${base_model_eval_results_dir}"
fi
echo "=============================================="
