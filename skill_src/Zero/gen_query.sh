#!/usr/bin/env bash
set -euo pipefail

# 导入资源清理库
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# 从环境变量获取路径配置
project_name="${SE_PROJECT_NAME:-Self-evolving-Agent}"
dir="${SE_BASE_DIR:-/home/ycy/data1}"
WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
CODE_MODULE="${SE_CODE_MODULE:-se_code_auto}"
prompt_path="${SE_PROMPT_DIR:-${WORKING_DIR}/${CODE_MODULE}}"

# 参数验证
exp_name=$1
num_samples=$2
challenger_model_path=$3
challenger_path_dir=$4
solver_path_dir=$5
gen_question_func=$6
hybrid_data=$7
train_file=$8
real_data_ratio=$9

if [ -z "$exp_name" ] || [ -z "$num_samples" ] || [ -z "$challenger_model_path" ] || [ -z "$challenger_path_dir" ] || [ -z "$solver_path_dir" ]; then
    echo "Error: 所有参数都不能为空"
    exit 1
fi

if [ ! -d "$challenger_model_path" ]; then
    echo "Error: challenger_model_path 不存在: $challenger_model_path"
    exit 1
fi

storage_path=${challenger_path_dir}/${exp_name}/gen_data
save_path_dir=${solver_path_dir}/${exp_name}

export VLLM_DISABLE_COMPILE_CACHE=1

# 从环境变量获取 GPU 配置，如果未设置则使用默认值
GEN_QUERY_GPUS="${SE_GEN_QUERY_GPUS:-4,5,6,7}"

# 将逗号分隔的字符串转换为数组
IFS=',' read -ra GPU_ARRAY <<< "$GEN_QUERY_GPUS"

echo "开始生成查询数据..."
echo "  实验名称: $exp_name"
echo "  样本数量: $num_samples"
echo "  存储路径: $storage_path"
echo "  使用 GPU: $GEN_QUERY_GPUS (共 ${#GPU_ARRAY[@]} 张)"
echo " ========================================== train_file: $train_file =========================================="
# 动态启动查询生成进程
GEN_PIDS=()
for i in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$i]}"
    suffix="$i"
    
    echo "启动查询生成进程: GPU=${gpu}, suffix=${suffix}"
    CUDA_VISIBLE_DEVICES=${gpu} python -m ${CODE_MODULE}.challenger_generate_query \
        --model "$challenger_model_path" \
        --suffix "$suffix" \
        --num_samples "$num_samples" \
        --storage_path="$storage_path" \
        --get_prompts_func="$gen_question_func" \
        --train_file="$train_file" &
    GEN_PIDS+=($!)
done

echo "等待所有查询生成进程完成... (共 ${#GEN_PIDS[@]} 个进程)"
for pid in "${GEN_PIDS[@]}"; do
    wait "$pid"
done

echo "查询生成完成，开始数据合并..."
sleep 5
if [ "$hybrid_data" == "True" ]; then
    python -m ${CODE_MODULE}.data_merge --data_path_dir="$storage_path" --save_path_dir="$save_path_dir" --exp_name="$exp_name" --hybrid_data 
else
    python -m ${CODE_MODULE}.data_merge --data_path_dir="$storage_path" --save_path_dir="$save_path_dir" --exp_name="$exp_name" 
fi
#python -m ${CODE_MODULE}.data_merge --data_path_dir="$storage_path" --save_path_dir="$save_path_dir" --exp_name="$exp_name" --hybrid_data
echo "数据合并完成"
