#!/usr/bin/env bash
set -euo pipefail

# 导入资源清理库
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

WORKING_DIR=/home/ycy/sdi/Self-evolving-Agent
solver_model_path=$1

# 验证参数
if [ -z "$solver_model_path" ]; then
    echo "Error: solver_model_path 不能为空"
    exit 1
fi

if [ ! -d "$solver_model_path" ]; then
    echo "Error: solver_model_path 不存在: $solver_model_path"
    exit 1
fi


export VLLM_DISABLE_COMPILE_CACHE=1
cd ${WORKING_DIR}/
echo "启动vLLM奖励服务器..."
#CUDA_VISIBLE_DEVICES=4 python -m  se_code.start_vllm_server --port 5000 --model_path $solver_model_path &

#CUDA_VISIBLE_DEVICES=5 python -m  se_code.start_vllm_server --port 5001 --model_path $solver_model_path &

CUDA_VISIBLE_DEVICES=6 python -m se_code.start_vllm_server --port 5000 --model_path $solver_model_path &

CUDA_VISIBLE_DEVICES=7 python -m se_code.start_vllm_server --port 5001 --model_path $solver_model_path &
