#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1

# 激活 conda 环境
# CONDA_ENV="${SE_CONDA_ENV:-se}"
# CONDA_BASE="/home/ycy/miniconda3"
# if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
#     source "${CONDA_BASE}/etc/profile.d/conda.sh"
#     conda activate "$CONDA_ENV"
#     echo "Activated conda environment: $CONDA_ENV"
#     echo "Python path: $(which python)"
# else
#     echo "Warning: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
# fi

project_name="${SE_PROJECT_NAME:-Self-evolving-Agent}"
dir="${SE_BASE_DIR:-/home/ycy/data1}"
model_dir="${SE_MODEL_DIR:-${dir}/models}"
data_dir="${SE_DATA_DIR:-${dir}/data}"
saved_results_dir="${SE_SAVED_RESULTS_DIR:-/home/ycy/data3/ttrl_saved}"
WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
save_path_dir=${saved_results_dir}/evaluation
solver_path_dir="${SE_SOLVER_DIR:-${saved_results_dir}/Solver_ttrl}"
eval_path=${WORKING_DIR}/evaluation
cd ${eval_path}
prefix=$1
eval_step=$2
num_iter=$3
base_model_name=$4
INPUT_DATASET=$5  # 保存原始输入，避免被循环变量覆盖

# 参数验证
if [ -z "$prefix" ] || [ -z "$eval_step" ] || [ -z "$num_iter" ] || [ -z "$base_model_name" ] || [ -z "$INPUT_DATASET" ]; then
    echo "Usage: $0 <prefix> <eval_step> <num_iter> <base_model_name> <dataset>"
    echo "Example: $0 my_experiment 10 20 Qwen2.5-Math-1.5B AIME24"
    exit 1
fi

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
n_samples=${dataset_n_samples_map[$INPUT_DATASET]}
temperature=${dataset_temperature_map[$INPUT_DATASET]}

# 验证数据集是否在支持列表中
if [ -z "$n_samples" ] || [ -z "$temperature" ]; then
    echo "Error: Unknown dataset '$INPUT_DATASET'. Supported datasets: AIME24, AIME25, AMC23, MATH500, Minerva, OlympiadBench"
    exit 1
fi

echo "==> Configuration:"
echo "    prefix: $prefix"
echo "    eval_step: $eval_step"
echo "    num_iter: $num_iter"
echo "    base_model_name: $base_model_name"
echo "    dataset: $INPUT_DATASET"
echo "    n_samples: $n_samples"
echo "    temperature: $temperature"

# Base model 评测结果保存在全局目录（只评测一次）
global_base_model_eval_dir=${save_path_dir}
base_model_eval_results_dir=${global_base_model_eval_dir}/${base_model_name}

eval_saved_path_dir=${save_path_dir}/${prefix}_data${INPUT_DATASET}_step${eval_step}_temperature${temperature}_n_samples${n_samples}
tb_path_dir=${saved_results_dir}/eval_tb_log
mkdir -p ${eval_saved_path_dir} ${tb_path_dir} ${WORKING_DIR}/eval_logs ${global_base_model_eval_dir}
echo "eval_saved_path_dir: ${eval_saved_path_dir}"
echo "tb_path_dir: ${tb_path_dir}"
echo "WORKING_DIR: ${WORKING_DIR}"
echo "global_base_model_eval_dir: ${global_base_model_eval_dir}"

model_list=()
BASE_MODEL_NEEDS_EVAL="false"
base_model_path="${model_dir}/${base_model_name}"

# 检查 base model 是否需要评测
if [ ! -d "${base_model_path}" ]; then
    echo "Warning: 基础模型不存在: ${base_model_path}，跳过..."
elif [ -d "${base_model_eval_results_dir}" ]; then
    # 检查是否有完整的评测结果（检查关键结果文件）
    if [ -f "${base_model_eval_results_dir}/${INPUT_DATASET}_Overall_results.jsonl" ]; then
        echo "基础模型 ${base_model_name} 在数据集 ${INPUT_DATASET} 上的评测结果已存在: ${base_model_eval_results_dir}，跳过评测。"
    else
        echo "基础模型 ${base_model_name} 在数据集 ${INPUT_DATASET} 上的评测结果不完整，需要重新评测"
        BASE_MODEL_NEEDS_EVAL="true"
        model_list+=("${base_model_name}")
    fi
else
    echo "基础模型 ${base_model_name} 需要评测，结果将保存到: ${base_model_eval_results_dir}"
    BASE_MODEL_NEEDS_EVAL="true"
    model_list+=("${base_model_name}")
fi

for i in $(seq 1 ${num_iter}); do
  model_list+=("${prefix}-V${i}")
done
echo "最终的 model_list: ${model_list[@]}"

function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
exec > >(tee -a "${WORKING_DIR}/eval_logs/eval_${prefix}_${INPUT_DATASET}-$(now).log") 2>&1
TASKS=(${INPUT_DATASET})

# Initialize GPU queue - use all available GPUs
GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))
#GPU_QUEUE=(5 6 7)
if [ ${#GPU_QUEUE[@]} -eq 0 ]; then
    echo "Error: No GPUs detected."
    exit 1
fi

echo "Available GPUs: ${GPU_QUEUE[@]} (Total: ${#GPU_QUEUE[@]})"

# Global tracking variables
declare -A gpu_status
declare -A pids
declare -A model_gpu_mapping  # Track which GPUs are assigned to which model
declare -A model_path_mapping  # Track model paths for each GPU
declare -A model_info           # Store model information for pending models
declare -a retry_task_queue=()  # Queue for failed tasks that need retry

# Initialize GPU status
for gpu_id in "${GPU_QUEUE[@]}"; do
    gpu_status["$gpu_id"]="idle"
done

# Function to cleanup GPU processes and free memory
cleanup_gpu_processes() {
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up GPU processes to free memory..."
    
    # Show current GPU processes before cleanup
    echo "Current GPU processes:"
    nvidia-smi pmon -c 1 2>/dev/null || echo "nvidia-smi pmon not available"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9

    # Find and kill processes using GPU memory
    echo "Finding processes using GPU memory..."
    local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
    
    if [ -n "$gpu_pids" ]; then
        echo "Killing GPU processes: $gpu_pids"
        for pid in $gpu_pids; do
            if [ -n "$pid" ] && [ "$pid" != "N/A" ]; then
                echo "Killing process $pid"
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        
        # Wait for processes to terminate gracefully
        sleep 3
        
        # Force kill if still running
        for pid in $gpu_pids; do
            if [ -n "$pid" ] && [ "$pid" != "N/A" ]; then
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Force killing process $pid"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        done
    else
        echo "No GPU processes found"
    fi
    
    # Clear GPU memory cache using Python
    echo "Clearing GPU memory cache..."
    python3 -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print('GPU cache cleared and garbage collected')
else:
    print('CUDA not available')
" 2>/dev/null || echo "Failed to clear GPU cache"
    
    # Wait a moment for memory to be freed
    sleep 2
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU cleanup completed"
    echo "GPU status after cleanup:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
}

# Function to get available GPUs
get_available_gpus() {
    local available_gpus=()
    for gpu_id in "${GPU_QUEUE[@]}"; do
        # Check both script status and actual GPU usage
        if [ "${gpu_status[$gpu_id]}" = "idle" ]; then
            # Check actual GPU memory usage
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            
            # Check GPU utilization
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            # Remove any whitespace from gpu_util
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            # Only consider GPU available if memory usage is low (< 5%) AND GPU utilization is low (< 5%)
            if [ "$memory_percent" -lt 5 ] && [ "$gpu_util" -lt 5 ]; then
                available_gpus+=("$gpu_id")
            else
                if [ "$memory_percent" -ge 5 ]; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] has high memory usage (${memory_percent}%), skipping..." >&2
                fi
                if [ "$gpu_util" -ge 5 ]; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] has high utilization (${gpu_util}%), skipping..." >&2
                fi
            fi
        fi
    done
    echo "${available_gpus[@]}"
}

# Function to check if main task is already completed
check_main_task_completed() {
    local model_name="$1"
    local dataset="$2"
    
    # Check if Overall_results.jsonl file exists
    local result_file
    if [ "$model_name" == "$base_model_name" ]; then
        # Base model 保存在全局目录
        result_file="${base_model_eval_results_dir}/${dataset}_Overall_results.jsonl"
    else
        result_file="${eval_saved_path_dir}/${model_name}/${dataset}_Overall_results.jsonl"
    fi
    
    if [ -f "$result_file" ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Main task [${dataset}] for model [${model_name}] already completed (found: ${result_file})"
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

# Function to start a math task job on one GPU
start_math_task_job() {
    local gpu_id="$1"
    local model_path="$2"
    local model_name="$3"
    local dataset="$4"
    
    # Base model 保存在全局目录
    local target_save_dir="${eval_saved_path_dir}"
    if [ "$model_name" == "$base_model_name" ]; then
        target_save_dir="${global_base_model_eval_dir}"
    fi
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting math task [${dataset}] for model [${model_name}] on GPU [${gpu_id}] (save_dir: ${target_save_dir})"
    
    # Determine n_samples and temperature based on dataset
    local n_samples=${dataset_n_samples_map[$dataset]}
    local temp=${dataset_temperature_map[$dataset]}
    
    # Start evaluation script on GPU
    CUDA_VISIBLE_DEVICES="${gpu_id}" python eval_all_math_para.py --model_path "${model_path}" --model_name "${model_name}" --dataset "${dataset}" --save_path_dir "${target_save_dir}" --n_samples "${n_samples}" --temperature "${temp}" --data_path_dir "${data_dir}" &
    local pid=$!
    
    # Wait a moment to check if process started successfully
    sleep 10
    
    # Check if process is still running
    if kill -0 "$pid" 2>/dev/null; then
        # Process started successfully, now mark GPU as busy
        gpu_status["${gpu_id}"]="busy"
        pids["${gpu_id}"]="$pid"
        model_gpu_mapping["${gpu_id}"]="${model_name}"
        model_path_mapping["${gpu_id}"]="${model_path}"
        gpu_dataset["${gpu_id}"]="${dataset}"
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started math task [${dataset}] for model [${model_name}]"
        return 0
    else
        # Process failed to start, clean up and add to retry queue
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Error: Failed to start math task [${dataset}] for model [${model_name}]"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 5
        kill -KILL "$pid" 2>/dev/null || true
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Cleaned up failed process for math task [${dataset}], adding to retry queue"
        retry_task_queue+=("${model_name}|${model_path}|${dataset}|math")
        return 1
    fi
}

# Function to check for completed jobs and free up GPUs
check_completed_jobs() {
    # First pass: detect newly completed GPUs and run post_eval immediately for each completed task
    for gpu_id in "${!pids[@]}"; do
        pid="${pids[$gpu_id]}"
        
        # Check if process is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            local model_name="${model_gpu_mapping[$gpu_id]}"
            local model_path="${model_path_mapping[$gpu_id]}"
            
            # Check if we've already processed this GPU
            if [ -z "${task_completed[$gpu_id]}" ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job finished for model [${model_name}] with PID [${pid}]"
                task_completed["$gpu_id"]=1
                
                # Run post_eval immediately for this completed task
                local dataset="${gpu_dataset[$gpu_id]}"
                local task_verified=false
                
                local n_samples=${dataset_n_samples_map[$dataset]}
                local temp=${dataset_temperature_map[$dataset]}
                
                # Base model 保存在全局目录
                local target_save_dir="${eval_saved_path_dir}"
                if [ "$model_name" == "$base_model_name" ]; then
                    target_save_dir="${global_base_model_eval_dir}"
                fi
                
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Running post_eval for [${model_name}] on dataset [${dataset}] with n_samples [${n_samples}] and temperature [${temp}]..."
                python post_eval.py --save_path_dir "${target_save_dir}" --dataset "${dataset}" --model_name "${model_name}" --n_samples "${n_samples}" --temperature "${temp}"
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Post-processing completed for GPU [${gpu_id}] (model [${model_name}], dataset [${dataset}], n_samples [${n_samples}], temperature [${temp}])"
                
                # Verify math task completion
                if check_main_task_completed "$model_name" "$dataset"; then
                    task_verified=true
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ✓ Verified: Math task [${dataset}] for model [${model_name}] completed successfully"
                else
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ✗ Verification failed: Math task [${dataset}] for model [${model_name}] - adding to retry queue"
                    retry_task_queue+=("${model_name}|${model_path}|${dataset}|math")
                fi
                
                # Free this GPU
                free_gpu "$gpu_id"
            fi
        fi
    done
    
    # Second pass: check for stuck processes (original logic)
    for gpu_id in "${!pids[@]}"; do
        pid="${pids[$gpu_id]}"
        
        if kill -0 "$pid" 2>/dev/null; then
            local model_name="${model_gpu_mapping[$gpu_id]}"
            
            # Process is running, but check if it's actually using GPU resources
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            
            # Check GPU utilization
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            # If GPU memory usage AND utilization are very low for a long time, the process might be stuck
            if [ "$memory_percent" -lt 2 ] && [ "$memory_used" -lt 100 ] && [ "$gpu_util" -lt 5 ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Warning: GPU [${gpu_id}] has very low memory usage (${memory_percent}%) and utilization (${gpu_util}%) for model [${model_name}]"
                # Check if this is a stuck process by looking at CPU usage
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

# Array to track which task has completed for each GPU
declare -A task_completed
# Array to track which dataset each GPU is processing
declare -A gpu_dataset

# Function to properly free up a GPU
free_gpu() {
    local gpu_id="$1"
    local model_name="${model_gpu_mapping[$gpu_id]}"
    
    # Clean up tracking variables
    unset pids["$gpu_id"]
    gpu_status["$gpu_id"]="idle"
    unset model_gpu_mapping["$gpu_id"]
    unset model_path_mapping["$gpu_id"]
    unset task_completed["$gpu_id"]
    unset gpu_dataset["$gpu_id"]
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Freed GPU [${gpu_id}] (was running model [${model_name}])"
}

# Function to detect and clean up zombie processes
cleanup_zombie_processes() {
    # Find all eval_all_math_para.py processes that might not be tracked
    local orphaned_pids=()
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            local pid=$(echo "$line" | awk '{print $2}')
            local gpu_used=$(echo "$line" | grep -o 'CUDA_VISIBLE_DEVICES=[0-9]*' | cut -d'=' -f2)
            
            # Check if this PID is not in our tracking
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
    done < <(ps aux | grep -E "eval_all_math_para.py" | grep -v grep)
    
    # Kill orphaned processes
    for pid in "${orphaned_pids[@]}"; do
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Killing orphaned process PID [${pid}]"
        kill -TERM "$pid" 2>/dev/null || true
    done
}

# Function to perform comprehensive GPU state cleanup
comprehensive_cleanup() {
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Performing comprehensive GPU state cleanup..."
    
    # Check all GPUs and their actual usage
    for gpu_id in {0..7}; do
        if nvidia-smi -i "$gpu_id" >/dev/null 2>&1; then
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            
            # Check GPU utilization
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            # If GPU is marked as busy but has very low memory usage AND utilization, it might be stuck
            if [ "${gpu_status[$gpu_id]}" = "busy" ] && [ "$memory_percent" -lt 5 ] && [ "$gpu_util" -lt 5 ]; then
                local model_name="${model_gpu_mapping[$gpu_id]}"
                local pid="${pids[$gpu_id]}"
                
                # Check if the process is actually still running
                if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                    # Process has finished, but don't free GPU here - let check_completed_jobs handle it
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] process finished for model [${model_name}], will be handled by check_completed_jobs..."
                else
                    # Process is still running but stuck, force cleanup
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] marked as busy but has low memory usage (${memory_percent}%) and utilization (${gpu_util}%), forcing cleanup..."
                    free_gpu "$gpu_id"
                fi
            fi
        fi
    done
    
    # Clean up any orphaned processes
    cleanup_zombie_processes
}

# Main execution loop
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting math tasks for ${#model_list[@]} models"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Math dataset: ${TASKS[*]}"

# Build math task queue: array of "model_name|model_path|dataset"
declare -a math_task_queue=()

# Initialize math task queue with all model-dataset combinations
for model_name in "${model_list[@]}"; do
    if [ "$model_name" == "$base_model_name" ]; then
        model_path=${model_dir}/${base_model_name}
    elif [[ $model_name =~ V[0-9]+ ]]; then
        model_path=${solver_path_dir}/${model_name}/ckpts/global_step_${eval_step}/actor/huggingface
    else
        echo "Warning: Model name [${model_name}] does not match any known pattern"
        echo "Skipping model: $model_name"
        continue
    fi
    
    # Verify model path exists
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path does not exist: $model_path"
        echo "Skipping model: $model_name"
        continue
    fi
    
    # Add each math dataset to the queue
    for dataset in "${TASKS[@]}"; do
        # Check if already completed before adding to queue
        if check_main_task_completed "$model_name" "$dataset"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Skipping already completed task: model [${model_name}], dataset [${dataset}]"
            continue
        fi
        math_task_queue+=("${model_name}|${model_path}|${dataset}")
    done
done

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Math task queue initialized with ${#math_task_queue[@]} tasks"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Math Task queue:"
for i in "${!math_task_queue[@]}"; do
  echo "[$i] ${math_task_queue[$i]}"
done

# Initialize cleanup counter
cleanup_counter=0
math_task_index=0

# Process math tasks asynchronously
while [ $math_task_index -lt ${#math_task_queue[@]} ] || [ ${#pids[@]} -gt 0 ]; do
    # Perform periodic comprehensive cleanup every 10 iterations
    cleanup_counter=$((cleanup_counter + 1))
    if [ $((cleanup_counter % 10)) -eq 0 ]; then
        comprehensive_cleanup
    else
        cleanup_zombie_processes
    fi
    
    check_completed_jobs
    available_gpus=($(get_available_gpus))
    
    # Try to start new tasks if there are available GPUs and pending tasks
    while [ $math_task_index -lt ${#math_task_queue[@]} ] && [ ${#available_gpus[@]} -ge 1 ]; do
        # Get next task from queue
        task="${math_task_queue[$math_task_index]}"
        IFS='|' read -r model_name model_path dataset <<< "$task"
        
        # Double-check if completed (in case it was completed by another process)
        if check_main_task_completed "$model_name" "$dataset"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Task already completed, skipping: model [${model_name}], dataset [${dataset}]"
            math_task_index=$((math_task_index + 1))
            continue
        fi
        
        # Validate that GPU ID is a number
        if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]]; then
            # Start the math task job on the first available GPU
            if start_math_task_job "${available_gpus[0]}" "$model_path" "$model_name" "$dataset"; then
                current_task=$((math_task_index + 1))
                total_tasks=${#math_task_queue[@]}
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started math task [${dataset}] for model [${model_name}] - task ${current_task}/${total_tasks}"
                math_task_index=$((math_task_index + 1))
                # Update available GPUs after starting a job
                available_gpus=($(get_available_gpus))
            else
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Failed to start math task [${dataset}] for model [${model_name}], will retry later"
                # Break to wait for GPU availability
                break
            fi
        else
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Invalid GPU ID detected: GPU='${available_gpus[0]}'"
            math_task_index=$((math_task_index + 1))
            continue
        fi
    done
    
    # If there are still running jobs or pending tasks, wait a bit
    if [ ${#pids[@]} -gt 0 ] || [ $math_task_index -lt ${#math_task_queue[@]} ]; then
        if [ ${#pids[@]} -gt 0 ]; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Running: ${#pids[@]} jobs, Pending: $((${#math_task_queue[@]} - math_task_index)) tasks"
        fi
        sleep 30
    fi
done

# Process retry queue for math tasks
if [ ${#retry_task_queue[@]} -gt 0 ]; then
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Processing ${#retry_task_queue[@]} retry tasks for math evaluations..."
    
    # Filter only math tasks from retry queue
    math_retry_tasks=()
    for task in "${retry_task_queue[@]}"; do
        IFS='|' read -r model_name model_path dataset task_type <<< "$task"
        if [ "$task_type" == "math" ]; then
            math_retry_tasks+=("${model_name}|${model_path}|${dataset}")
        fi
    done
    
    if [ ${#math_retry_tasks[@]} -gt 0 ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Found ${#math_retry_tasks[@]} math retry tasks"
        
        retry_index=0
        while [ $retry_index -lt ${#math_retry_tasks[@]} ] || [ ${#pids[@]} -gt 0 ]; do
            check_completed_jobs
            available_gpus=($(get_available_gpus))
            
            while [ $retry_index -lt ${#math_retry_tasks[@]} ] && [ ${#available_gpus[@]} -ge 1 ]; do
                task="${math_retry_tasks[$retry_index]}"
                IFS='|' read -r model_name model_path dataset <<< "$task"
                
                if check_main_task_completed "$model_name" "$dataset"; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Retry task already completed, skipping: model [${model_name}], dataset [${dataset}]"
                    retry_index=$((retry_index + 1))
                    continue
                fi
                
                if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]]; then
                    if start_math_task_job "${available_gpus[0]}" "$model_path" "$model_name" "$dataset"; then
                        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Retry: Successfully started math task [${dataset}] for model [${model_name}]"
                        retry_index=$((retry_index + 1))
                        available_gpus=($(get_available_gpus))
                    else
                        break
                    fi
                else
                    retry_index=$((retry_index + 1))
                fi
            done
            
            if [ ${#pids[@]} -gt 0 ] || [ $retry_index -lt ${#math_retry_tasks[@]} ]; then
                sleep 30
            fi
        done
    fi
fi

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All math tasks completed successfully!"

# Final verification: check if all tasks are truly completed
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Final verification of all tasks..."
all_completed=true
failed_tasks=()

for model_name in "${model_list[@]}"; do
    # Check math tasks
    for dataset in "${TASKS[@]}"; do
        if ! check_main_task_completed "$model_name" "$dataset" 2>/dev/null; then
            all_completed=false
            failed_tasks+=("${model_name}|${dataset}")
        fi
    done
done

if [ "$all_completed" = true ]; then
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ✓ All tasks verified complete!"
else
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ⚠ WARNING: The following tasks are still incomplete:"
    for task in "${failed_tasks[@]}"; do
        echo "    - $task"
    done
fi

# Upload evaluation results to TensorBoard
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Uploading evaluation results to TensorBoard..."

# 使用 se 环境的 Python（确保 verl 可用）

python tb_single_math.py  \
    --prefix=$prefix \
    --dataset=$INPUT_DATASET \
    --eval_results_dir="${eval_saved_path_dir}" \
    --tb_path_dir="${tb_path_dir}" \
    --base_model="${base_model_name}" \
    --base_model_dir="${global_base_model_eval_dir}" \

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All models eval results upload tb successfully!"
