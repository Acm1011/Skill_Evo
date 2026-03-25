#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1
export HF_ENDPOINT=https://hf-mirror.com
eval_path_dir=/root/users/ycy/saved_results/rz_eval_results
base_path_dir=/root/users/ycy/models/shares
solver_path_dir=/root/users/ycy/saved_results/Solver
model_name=$1

if [ $model_name == "Qwen3-4B-Base" ]; then
  model_path=${base_path_dir}/Qwen3-4B-Base
elif [ $model_name == "Qwen3-8B-Base" ]; then
  model_path=${base_path_dir}/Qwen3-8B-Base
elif [ $model_name == "Qwen3-14B-Base" ]; then
  model_path=${base_path_dir}/Qwen3-14B-Base
elif [[ $model_name == *"V"[1-9] ]]; then
  model_path=${solver_path_dir}/${model_name}/ckpts/global_step_20/actor/huggingface
else
  echo "错误: 不支持的模型名称: $model_name"
  echo "支持的模型: Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-14B-Base, 或包含V1-V9的模型名"
  exit 1
fi

# 检查model_path是否存在
if [ ! -d "$model_path" ]; then
  echo "错误: 模型路径不存在: $model_path"
  echo "请检查模型名称是否正确: $model_name"
  exit 1
fi

echo "使用模型路径: $model_path"
save_path=${eval_path_dir}/${model_name}
mkdir -p $save_path

MODEL_NAMES=(
  $model_path
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

GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))
echo "Available GPUs: ${GPU_QUEUE[@]}"

declare -A pids

start_job() {
  local gpu_id="$1"
  local model="$2"
  local task="$3"

  echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Start task [${task}] with model [${model}] on GPU [${gpu_id}] ..."

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python generate.py --model "${model}" --dataset "${task}" --save_path "${save_path}" &

  pids["${gpu_id}"]=$!
}

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "==> Processing model: ${MODEL_NAME}"
    TASK_INDEX=0
    NUM_TASKS=${#TASKS[@]}

    while :; do
        while [ ${#GPU_QUEUE[@]} -gt 0 ] && [ ${TASK_INDEX} -lt ${NUM_TASKS} ]; do
            gpu_id="${GPU_QUEUE[0]}"
            GPU_QUEUE=("${GPU_QUEUE[@]:1}")

            task="${TASKS[${TASK_INDEX}]}"
            ((TASK_INDEX++))

            start_job "$gpu_id" "$MODEL_NAME" "$task"
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

python results_recheck.py --model_path $model_path  --save_path $save_path 

# python -m se.evaluation.eval_supergpqa --model_path $model_path
# python -m se.evaluation.eval_bbeh --model_path $model_path
# python -m se.evaluation.eval_mmlupro --model_path $model_path


echo "==> All tasks have finished!"
