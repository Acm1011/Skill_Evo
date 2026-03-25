#!/bin/bash
# 运行所有 math task 的 post_eval 脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

saved_dir=/home/ycy/data3/ttrl_saved
save_path_dir=/home/ycy/data3/ttrl_saved/evaluation/data_AIME24_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B_step10_temperature0.6
tb_path_dir=/home/ycy/data3/ttrl_saved/eval_tb_log
eval_step=10
temperature=0.6
prefix=data_AIME24_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-1.5B

# 定义模型列表
model_list=()
for i in {1..9}; do
  model_list+=("${prefix}-V${i}")
done

# 定义 math 任务
TASKS=(
  "temp_data"
  "greedy_data"
)

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting post_eval for all math tasks..."
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Models: ${#model_list[@]}, Tasks per model: ${#TASKS[@]}"

# 统计信息
total_tasks=0
completed_tasks=0
skipped_tasks=0
failed_tasks=0
pending_tasks=0

# 先统计需要处理的任务数
for model_name in "${model_list[@]}"; do
    model_dir="${save_path_dir}/${model_name}"
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    for dataset in "${TASKS[@]}"; do
        overall_results_file="${model_dir}/${dataset}_Overall_results.jsonl"
        input_file="${model_dir}/${dataset}_responses.parquet"
        if [ ! -f "$overall_results_file" ] && [ -f "$input_file" ]; then
            pending_tasks=$((pending_tasks + 1))
        fi
    done
done

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Found ${pending_tasks} tasks that need post_eval processing"
echo ""

# 遍历所有模型和数据集
current_task=0
for model_name in "${model_list[@]}"; do
    model_dir="${save_path_dir}/${model_name}"
    
    # 检查模型目录是否存在
    if [ ! -d "$model_dir" ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Warning: Model directory does not exist: ${model_dir}, skipping..."
        continue
    fi
    
    for dataset in "${TASKS[@]}"; do
        total_tasks=$((total_tasks + 1))
        
        # 检查 Overall_results.jsonl 文件是否存在（如果存在，说明已经运行过 post_eval）
        overall_results_file="${model_dir}/${dataset}_Overall_results.jsonl"
        
        if [ -f "$overall_results_file" ]; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Skipping [${model_name}] on dataset [${dataset}] - already processed (found: ${overall_results_file})"
            skipped_tasks=$((skipped_tasks + 1))
            continue
        fi
        
        # 检查是否有原始输出文件（parquet 格式）
        input_file="${model_dir}/${dataset}_responses.parquet"
        
        if [ ! -f "$input_file" ]; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Warning: Input file not found for [${model_name}] on dataset [${dataset}] (expected: ${input_file}), skipping..."
            skipped_tasks=$((skipped_tasks + 1))
            continue
        fi
        
        # 确定 n_samples 和 temperature
        n_samples=32
        temp=$temperature
        if [ "$dataset" == "${TASKS[1]}" ]; then
            n_samples=1
            temp=0.0
        fi
        
        current_task=$((current_task + 1))
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [${current_task}/${pending_tasks}] Running post_eval for [${model_name}] on dataset [${dataset}] with n_samples [${n_samples}] and temperature [${temp}]..."
        
        # 运行 post_eval
        if python post_eval.py --save_path_dir "${save_path_dir}" --dataset "${dataset}" --model_name "${model_name}" --n_samples "${n_samples}" --temperature "${temp}"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully completed post_eval for [${model_name}] on dataset [${dataset}]"
            completed_tasks=$((completed_tasks + 1))
        else
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to run post_eval for [${model_name}] on dataset [${dataset}]"
            failed_tasks=$((failed_tasks + 1))
        fi
    done
done

echo ""
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Post_eval summary:"
echo "  Total tasks: ${total_tasks}"
echo "  Completed: ${completed_tasks}"
echo "  Skipped (already processed): ${skipped_tasks}"
echo "  Failed: ${failed_tasks}"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All post_eval tasks finished!"
# python tb.py \
#   --prefix "${prefix}" \
#   --step "${eval_step}" \
#   --temperature "${temperature}" \
#   --eval_results_dir "${save_path_dir}" \
#   --tb_path_dir "${tb_path_dir}"