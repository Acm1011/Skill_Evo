#!/usr/bin/env bash
set -euo pipefail

# 导入资源清理库
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

WORKING_DIR=/root/users/ycy/Self-evolving-Agent

# 参数验证
exp_name=$1
num_samples=$2
challenger_model_path=$3
challenger_path_dir=$4
solver_path_dir=$5

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
tmp=${challenger_path_dir}/${exp_name}/tmp
mkdir -p $tmp

export VLLM_DISABLE_COMPILE_CACHE=1

echo "开始生成查询数据..."
echo "  实验名称: $exp_name"
echo "  样本数量: $num_samples"
echo "  存储路径: $storage_path"

# 启动并行查询生成进程
CUDA_VISIBLE_DEVICES=0 python challenger_generate_query.py --model "$challenger_model_path" --suffix 0 --num_samples "$num_samples" --storage_path="$storage_path" &
GEN_PID1=$!

CUDA_VISIBLE_DEVICES=1 python challenger_generate_query.py --model "$challenger_model_path" --suffix 1 --num_samples "$num_samples" --storage_path="$storage_path" &
GEN_PID2=$!

CUDA_VISIBLE_DEVICES=2 python challenger_generate_query.py --model "$challenger_model_path" --suffix 2 --num_samples "$num_samples" --storage_path="$storage_path" &
GEN_PID3=$!

CUDA_VISIBLE_DEVICES=3 python challenger_generate_query.py --model "$challenger_model_path" --suffix 3 --num_samples "$num_samples" --storage_path="$storage_path" &
GEN_PID4=$!

CUDA_VISIBLE_DEVICES=4 python challenger_generate_query.py --model "$challenger_model_path" --suffix 4 --num_samples "$num_samples" --storage_path="$storage_path" &
GEN_PID5=$!

CUDA_VISIBLE_DEVICES=5 python challenger_generate_query.py --model "$challenger_model_path" --suffix 5 --num_samples "$num_samples" --storage_path="$storage_path" &
GEN_PID6=$!

CUDA_VISIBLE_DEVICES=6 python challenger_generate_query.py --model "$challenger_model_path" --suffix 6 --num_samples "$num_samples" --storage_path="$storage_path" &
GEN_PID7=$!

CUDA_VISIBLE_DEVICES=7 python challenger_generate_query.py --model "$challenger_model_path" --suffix 7 --num_samples "$num_samples" --storage_path="$storage_path" &
GEN_PID8=$!

echo "等待所有查询生成进程完成..."
wait $GEN_PID1 $GEN_PID2 $GEN_PID3 $GEN_PID4 $GEN_PID5 $GEN_PID6 $GEN_PID7 $GEN_PID8

echo "查询生成完成，开始数据合并..."
sleep 5
cd ${WORKING_DIR}/se
python data_merge.py --data_path_dir="$storage_path" --save_path_dir="$save_path_dir" --exp_name="$exp_name"
echo "数据合并完成"
