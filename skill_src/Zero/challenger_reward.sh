#!/usr/bin/env bash
set -euo pipefail

# 导入资源清理库
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# 从环境变量获取路径配置
dir="${SE_BASE_DIR:-/home/ycy/data1}"
project_name="${SE_PROJECT_NAME:-Self-evolving-Agent}"
WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
CODE_MODULE="${SE_CODE_MODULE:-se_code_auto}"

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

# 从环境变量获取 GPU 和端口配置，如果未设置则使用默认值
REWARD_GPUS="${SE_REWARD_GPUS:-6,7}"
REWARD_PORTS="${SE_REWARD_PORTS:-5000,5001}"
REWARD_BASE_PORT="${SE_REWARD_BASE_PORT:-5000}"

# 将逗号分隔的字符串转换为数组
IFS=',' read -ra GPU_ARRAY <<< "$REWARD_GPUS"
IFS=',' read -ra PORT_ARRAY <<< "$REWARD_PORTS"

echo "[Reward Server] GPU 配置: ${REWARD_GPUS}"
echo "[Reward Server] 端口配置: ${REWARD_PORTS}"

export VLLM_DISABLE_COMPILE_CACHE=1
cd ${WORKING_DIR}/
echo "启动vLLM奖励服务器..."

# 动态启动多个 reward server
for i in "${!PORT_ARRAY[@]}"; do
    port="${PORT_ARRAY[$i]}"
    # 使用循环索引对 GPU 数组取模，支持 GPU 数量和端口数量不同的情况
    gpu_idx=$((i % ${#GPU_ARRAY[@]}))
    gpu="${GPU_ARRAY[$gpu_idx]}"
    
    echo "启动 Reward Server: GPU=${gpu}, Port=${port}"
    CUDA_VISIBLE_DEVICES=${gpu} python -m ${CODE_MODULE}.start_vllm_server --port ${port} --model_path $solver_model_path &
done
