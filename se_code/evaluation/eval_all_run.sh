#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1
project_name=Self-evolving-Agent
dir=/home/ycy/data1
model_dir=${dir}/models
data_dir=${dir}/data
WORKING_DIR=${dir}/${project_name}
saved_results_dir=${dir}/saved_results
save_path_dir=${saved_results_dir}/evaluation
solver_path_dir=${saved_results_dir}/Solver
eval_path=${WORKING_DIR}/evaluation
cd ${eval_path}
prefix=$1
temperature=$2
eval_step=$3
num_iter=$4
model_list=()
for i in $(seq 1 ${num_iter}); do
  model_list+=("${prefix}-V${i}")
  echo "model_list: ${model_list[@]}"
done


TASKS=(
  "temp_data"
  "greedy_data"
)

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
declare -A model_info           # Store model information for pending models

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
            
            # Only consider GPU available if memory usage is low (< 10%)
            if [ "$memory_percent" -lt 5 ]; then
                available_gpus+=("$gpu_id")
            else
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] has high memory usage (${memory_percent}%), skipping..." >&2
            fi
        fi
    done
    echo "${available_gpus[@]}"
}

# Function to start a model job on two GPUs
start_model_job() {
    local gpu_id_0="$1"
    local gpu_id_1="$2"
    local model_path="$3"
    local model_name="$4"
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting model [${model_name}] on GPUs [${gpu_id_0}] and [${gpu_id_1}]"
    
    # Start temp_data task on first GPU
    CUDA_VISIBLE_DEVICES="${gpu_id_0}" python eval_all_math_para.py --model_path "${model_path}" --model_name "${model_name}" --dataset "${TASKS[0]}" --save_path_dir "${save_path_dir}" --n_samples 32 --temperature $temperature  &
    local pid_0=$!
    
    # Start greedy_data task on second GPU
    CUDA_VISIBLE_DEVICES="${gpu_id_1}" python eval_all_math_para.py --model_path "${model_path}" --model_name "${model_name}" --dataset "${TASKS[1]}" --save_path_dir "${save_path_dir}" --n_samples 1 --temperature 0.0  &
    local pid_1=$!
    
    # Wait a moment to check if processes started successfully
    sleep 10
    
    # Check if both processes are still running
    if kill -0 "$pid_0" 2>/dev/null && kill -0 "$pid_1" 2>/dev/null; then
        # Both processes started successfully, now mark GPUs as busy
        gpu_status["${gpu_id_0}"]="busy"
        gpu_status["${gpu_id_1}"]="busy"
        pids["${gpu_id_0}"]="$pid_0"
        pids["${gpu_id_1}"]="$pid_1"
        model_gpu_mapping["${gpu_id_0}"]="${model_name}"
        model_gpu_mapping["${gpu_id_1}"]="${model_name}"
        # Track which dataset each GPU is processing
        gpu_dataset["${gpu_id_0}"]="${TASKS[0]}"
        gpu_dataset["${gpu_id_1}"]="${TASKS[1]}"
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started both processes for model [${model_name}]"
    else
        # One or both processes failed to start, clean up
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Error: Failed to start processes for model [${model_name}]"
        kill -TERM "$pid_0" 2>/dev/null || true
        kill -TERM "$pid_1" 2>/dev/null || true
        sleep 5
        kill -KILL "$pid_0" 2>/dev/null || true
        kill -KILL "$pid_1" 2>/dev/null || true
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Cleaned up failed processes for model [${model_name}]"
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
            
            # Check if we've already processed this GPU
            if [ -z "${task_completed[$gpu_id]}" ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job finished for model [${model_name}] with PID [${pid}]"
                task_completed["$gpu_id"]=1
                
                # Run post_eval immediately for this completed task
                local dataset="${gpu_dataset[$gpu_id]}"
                local n_samples=32
                local temperature=$temperature
                if [ "$dataset" == "${TASKS[1]}" ]; then
                    n_samples=1
                    temperature=0.0
                fi
                
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Running post_eval for [${model_name}] on dataset [${dataset}] with n_samples [${n_samples}] and temperature [${temperature}]..."
                python post_eval.py --save_path_dir "${save_path_dir}" --dataset "${dataset}" --model_name "${model_name}" --n_samples "${n_samples}" --temperature "${temperature}"
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Post-processing completed for GPU [${gpu_id}] (model [${model_name}], dataset [${dataset}], n_samples [${n_samples}], temperature [${temperature}])"
                
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
            
            # If GPU memory usage is very low for a long time, the process might be stuck
            if [ "$memory_percent" -lt 2 ] && [ "$memory_used" -lt 100 ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Warning: GPU [${gpu_id}] has very low memory usage (${memory_percent}%) for model [${model_name}]"
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

# Array to track which task (0 or 1) has completed for each GPU
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
    unset task_completed["$gpu_id"]
    unset gpu_dataset["$gpu_id"]
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Freed GPU [${gpu_id}] (was running model [${model_name}])"
}

# Function to detect and clean up zombie processes
cleanup_zombie_processes() {
    # Find all eval_all_math.py processes that might not be tracked
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
    done < <(ps aux | grep "eval_all_math_para.py" | grep -v grep)
    
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
            
            # If GPU is marked as busy but has very low memory usage, it might be stuck
            if [ "${gpu_status[$gpu_id]}" = "busy" ] && [ "$memory_percent" -lt 5 ]; then
                local model_name="${model_gpu_mapping[$gpu_id]}"
                local pid="${pids[$gpu_id]}"
                
                # Check if the process is actually still running
                if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                    # Process has finished, but don't free GPU here - let check_completed_jobs handle it
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] process finished for model [${model_name}], will be handled by check_completed_jobs..."
                else
                    # Process is still running but stuck, force cleanup
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] marked as busy but has low memory usage (${memory_percent}%), forcing cleanup..."
                    free_gpu "$gpu_id"
                fi
            fi
        fi
    done
    
    # Clean up any orphaned processes
    cleanup_zombie_processes
}

# Main execution loop
# Clean up GPU processes before starting
#cleanup_gpu_processes

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting GPU queue management for ${#model_list[@]} models"

# Add periodic cleanup counter
cleanup_counter=0

# Process each model
for model_name in "${model_list[@]}"; do
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Processing model: ${model_name}"
    
    if [ "$model_name" == "Qwen3-4B-Base" ]; then
        model_path=${model_dir}/Qwen3-4B-Base
    elif [[ $model_name == *"V"[1-9] ]]; then
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
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Using model_path=$model_path, model_name=$model_name"
    
    # Wait for 2 GPUs to be available
    retry_count=0
    max_retries=5
    
    while true; do
        # Perform periodic comprehensive cleanup every 10 iterations
        cleanup_counter=$((cleanup_counter + 1))
        if [ $((cleanup_counter % 10)) -eq 0 ]; then
            comprehensive_cleanup
        else
            cleanup_zombie_processes  # Clean up any orphaned processes first
        fi
        
        check_completed_jobs      # Check for completed jobs
        available_gpus=($(get_available_gpus))
        
        # Debug: Print available GPUs
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Available GPUs: ${available_gpus[*]}"
        
        if [ ${#available_gpus[@]} -ge 2 ]; then
            # Validate that GPU IDs are numbers
            if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]] && [[ "${available_gpus[1]}" =~ ^[0-9]+$ ]]; then
                # Start the model job on the first two available GPUs
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Starting model job with GPU0=${available_gpus[0]} GPU1=${available_gpus[1]} model=$model_name"
                if start_model_job "${available_gpus[0]}" "${available_gpus[1]}" "$model_path" "$model_name"; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started model [${model_name}]"
                    break
                else
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Failed to start model [${model_name}], retrying..."
                    retry_count=$((retry_count + 1))
                    if [ $retry_count -ge $max_retries ]; then
                        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Max retries reached for model [${model_name}], skipping..."
                        break
                    fi
                    sleep 5
                fi
            else
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Invalid GPU IDs detected: GPU0='${available_gpus[0]}' GPU1='${available_gpus[1]}'"
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Skipping model [${model_name}] due to invalid GPU IDs"
                break
            fi
        else
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Waiting for 2 GPUs to be available. Currently available: ${#available_gpus[@]}"
            sleep 60
        fi
    done
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Started model: ${model_name}"
done

# Wait for all models to complete
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All models started. Waiting for completion..."

while [ ${#pids[@]} -gt 0 ]; do
    check_completed_jobs
    if [ ${#pids[@]} -gt 0 ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Still running: ${#pids[@]} jobs"
        sleep 60
    fi
done

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All models completed successfully!"

python tb.py --prefix=$prefix --step=$eval_step --temperature=$temperature

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All models eval results upload tb successfully!"
