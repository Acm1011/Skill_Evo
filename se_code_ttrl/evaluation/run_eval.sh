#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1
eval_path_dir=/root/users/ycy/saved_results/evaluation
base_path_dir=/root/users/ycy/models/shares/
solver_path_dir=/root/users/ycy/saved_results/Solver
data_path_dir=/root/users/ycy/data
greedy_data_path=${data_path_dir}/greedy_data.parquet
temp_data_path=${data_path_dir}/temp_data.parquet
model_list=(
  "Qwen3-4B-Base"
  "R-Zero_Qwen3-4B-Base-V1"
  "R-Zero_Qwen3-4B-Base-V2"
  "R-Zero_Qwen3-4B-Base-V3"
  "R-Zero_Qwen3-4B-Base-V4"
  "R-Zero_Qwen3-4B-Base-V5"
)

TASKS=(
  "math500"
  "amc23"
  "minerva"
  "olympiadbench"
  "aime24"
  "aime25"
)

# Initialize GPU queue and process tracking

GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))
if [ ${#GPU_QUEUE[@]} -eq 0 ]; then
    echo "Error: No GPUs detected."
    exit 1
fi
echo "Available GPUs: ${GPU_QUEUE[@]}"

declare -A pids

start_job() {
  local gpu_id="$1"
  local model="$2"
  local task="$3"
  local save_path="$4"
  echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Start task [${task}] with model [${model}] on GPU [${gpu_id}] and save path [${save_path}] ..."

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python eval_math.py --model "${model}" --dataset "${task}" --save_path "${save_path}" &

  pids["${gpu_id}"]=$!
}
for model_name in "${model_list[@]}"; do
    if [ "$model_name" == "Qwen3-4B-Base" ]; then
      model_path=${base_path_dir}/Qwen3-4B-Base
      save_path=${eval_path_dir}/${model_name}
    elif [[ $model_name == *"V"[1-9] ]]; then
      model_path=${solver_path_dir}/${model_name}/ckpts/global_step_20/actor/huggingface
      save_path=${solver_path_dir}/${model_name}/eval
    fi

    mkdir -p $save_path
    echo "==> Processing model: ${model_name}"
    
    
    # Verify model path exists
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path does not exist: $model_path"
        echo "Skipping model: $model_name"
        continue
    fi

    while :; do
        while [ ${#GPU_QUEUE[@]} -gt 0 ] && [ ${TASK_INDEX} -lt ${NUM_TASKS} ]; do
            gpu_id="${GPU_QUEUE[0]}"
            GPU_QUEUE=("${GPU_QUEUE[@]:1}")

            task="${TASKS[${TASK_INDEX}]}"
            ((TASK_INDEX++))

            start_job "$gpu_id" "$model_path" "$task" "$save_path"
        done

        if [ ${TASK_INDEX} -ge ${NUM_TASKS} ] && [ ${#pids[@]} -eq 0 ]; then
            break
        fi

        for gpu_id in "${!pids[@]}"; do
            pid="${pids[$gpu_id]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job finished with PID [${pid}]."
                unset pids["$gpu_id"]
                GPU_QUEUE+=("$gpu_id")
            fi
        done

        sleep 1
    done
done
