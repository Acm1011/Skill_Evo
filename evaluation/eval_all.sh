#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1
eval_path_dir=/root/users/ycy/saved_results/evaluation
base_path_dir=/root/users/ycy/models/shares
solver_path_dir=/root/users/ycy/saved_results/Solver

model_list=(
  "Qwen3-4B-Base"
  "R-Zero_Qwen3-4B-Base-V1"
  "R-Zero_Qwen3-4B-Base-V2"
  "R-Zero_Qwen3-4B-Base-V3"
  "R-Zero_Qwen3-4B-Base-V4"
  "R-Zero_Qwen3-4B-Base-V5"
)

TASKS=(
  "math"
  "gsm8k" 
  "amc"
  "minerva"
  "olympiad"
  "aime2024"
  "aime2025"
)

# Initialize GPU queue - use all available GPUs
GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))
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
            if [ "$memory_percent" -lt 10 ]; then
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
    local model_path="$3"
    local save_path="$4"
    local model_name="$5"
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting model [${model_name}] on GPUs [${gpu_id_0}] and [${gpu_id_1}]"
    
    # Start temp_data task on first GPU
    CUDA_VISIBLE_DEVICES="${gpu_id_0}" python eval_all_math.py --model "${model_path}" --dataset "${TASKS[0]}" --save_path "${save_path}" --n_samples 32 --temperature 1.0 --top_p 0.95 --top_k 50 &
    local pid_0=$!

    
    # Wait a moment to check if processes started successfully
    sleep 20
    
    # Check if both processes are still running
    if kill -0 "$pid_0" 2>/dev/null && kill -0 "$pid_1" 2>/dev/null; then
        # Both processes started successfully, now mark GPUs as busy
        gpu_status["${gpu_id_0}"]="busy"
        gpu_status["${gpu_id_1}"]="busy"
        pids["${gpu_id_0}"]="$pid_0"
        pids["${gpu_id_1}"]="$pid_1"
        model_gpu_mapping["${gpu_id_0}"]="${model_name}"
        model_gpu_mapping["${gpu_id_1}"]="${model_name}"
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
    for gpu_id in "${!pids[@]}"; do
        pid="${pids[$gpu_id]}"
        model_name="${model_gpu_mapping[$gpu_id]}"
        
        # Check if process is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job finished for model [${model_name}] with PID [${pid}]"
            free_gpu "$gpu_id"
        else
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

# Function to properly free up a GPU
free_gpu() {
    local gpu_id="$1"
    local model_name="${model_gpu_mapping[$gpu_id]}"
    
    # Clean up tracking variables
    unset pids["$gpu_id"]
    gpu_status["$gpu_id"]="idle"
    unset model_gpu_mapping["$gpu_id"]
    
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
    done < <(ps aux | grep "eval_all_math.py" | grep -v grep)
    
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
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] marked as busy but has low memory usage (${memory_percent}%), forcing cleanup..."
                free_gpu "$gpu_id"
            fi
        fi
    done
    
    # Clean up any orphaned processes
    cleanup_zombie_processes
}

# Function to get model information
get_model_info() {
    local model_name="$1"
    local model_path=""
    local save_path=""
    
    if [ "$model_name" == "Qwen3-4B-Base" ]; then
        model_path=${base_path_dir}/Qwen3-4B-Base
        save_path=${eval_path_dir}/${model_name}
    elif [[ $model_name == *"V"[1-9] ]]; then
        model_path=${solver_path_dir}/${model_name}/ckpts/global_step_20/actor/huggingface
        save_path=${solver_path_dir}/${model_name}/eval
    fi
    
    echo "${model_path}|${save_path}"
}

# Main execution loop
# Clean up GPU processes before starting
cleanup_gpu_processes

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting GPU queue management for ${#model_list[@]} models"

# Add periodic cleanup counter
cleanup_counter=0

# Process each model
for model_name in "${model_list[@]}"; do
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Processing model: ${model_name}"
    
    # Get model information
    model_info=$(get_model_info "$model_name")
    model_path=$(echo "$model_info" | cut -d'|' -f1)
    save_path=$(echo "$model_info" | cut -d'|' -f2)
    
    # Verify model path exists
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path does not exist: $model_path"
        echo "Skipping model: $model_name"
        continue
    fi
    
    # Create save directory
    mkdir -p "$save_path"
    
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
                if start_model_job "${available_gpus[0]}" "${available_gpus[1]}" "$model_path" "$save_path" "$model_name"; then
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
            sleep 600
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
        sleep 600
    fi
done

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All models completed successfully!"