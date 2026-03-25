#!/usr/bin/env bash
# =============================================================================
# run_with_gpus.sh - GPU-aware launcher for Self-evolving-Agent training
# =============================================================================
#
# 用法:
#   ./run_with_gpus.sh <n_gpus>
#   ./run_with_gpus.sh 4      # 使用 4 张 GPU
#   ./run_with_gpus.sh 8      # 使用 8 张 GPU
#
# 功能:
#   1. 检测当前服务器 GPU 状态（显存占用、利用率）
#   2. 等待直到空闲 GPU 数量满足 n_gpus
#   3. 选择空闲 GPU 并设置环境变量
#   4. 调用 main.sh 启动训练
#
# 环境变量输出:
#   SE_N_GPUS           - 总 GPU 数量
#   SE_GPU_IDS          - 选中的 GPU ID 列表（逗号分隔）
#   SE_CHALLENGER_GPUS  - Challenger 训练使用的 GPU
#   SE_REWARD_GPUS      - Reward Server 使用的 GPU
#   SE_SOLVER_GPUS      - Solver 训练使用的 GPU
#   SE_GEN_QUERY_GPUS   - 查询生成使用的 GPU
#   SE_REWARD_PORTS     - Reward Server 端口列表
#   SE_REWARD_BASE_PORT - Reward Server 基础端口
#
# =============================================================================
# sleep 7200
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# 配置参数
# =============================================================================

# GPU 空闲判定阈值
GPU_MEMORY_THRESHOLD_MB="${GPU_MEMORY_THRESHOLD_MB:-4000}"      # 显存占用低于此值视为空闲(MB)
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-5}"                  # 利用率低于此值视为空闲(%)

# 轮询配置
POLL_INTERVAL="${POLL_INTERVAL:-1800}"                            # 检查间隔（秒）
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-48}"                          # 最大等待时间（小时）

# Reward Server 配置
REWARD_BASE_PORT="${SE_REWARD_BASE_PORT:-5000}"                 # Reward Server 基础端口

# =============================================================================
# 路径配置 - 通过环境变量统一管理
# =============================================================================

# 基础目录 (可通过环境变量覆盖)
SE_BASE_DIR="${SE_BASE_DIR:-/home/ycy/data1}"
SE_PROJECT_NAME="${SE_PROJECT_NAME:-Self-evolving-Agent}"
SE_CODE_MODULE="${SE_CODE_MODULE:-se_code_ttrl}"

# 派生路径
SE_WORKING_DIR="${SE_BASE_DIR}/${SE_PROJECT_NAME}"
SE_MODEL_DIR="${SE_MODEL_DIR:-${SE_BASE_DIR}/models}"
SE_DATA_DIR="${SE_DATA_DIR:-${SE_BASE_DIR}/data}"
SE_SAVED_RESULTS_DIR="${SE_SAVED_RESULTS_DIR:-/home/ycy/data3/ttrl_saved}"
SE_CHALLENGER_DIR="${SE_SAVED_RESULTS_DIR}/Challenger_ttrl"
SE_SOLVER_DIR="${SE_SAVED_RESULTS_DIR}/Solver_ttrl"
SE_TENSORBOARD_DIR="${SE_SAVED_RESULTS_DIR}/tensorboard_log"
SE_PROMPT_DIR="${SE_WORKING_DIR}/${SE_CODE_MODULE}"
mkdir -p ${SE_CHALLENGER_DIR} ${SE_SOLVER_DIR} ${SE_TENSORBOARD_DIR} 
# =============================================================================
# 辅助函数
# =============================================================================

print_banner() {
    echo "=============================================="
    echo "  Self-evolving-Agent GPU Launcher"
    echo "=============================================="
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

print_usage() {
    echo "用法: $0 <n_gpus>"
    echo ""
    echo "参数:"
    echo "  n_gpus    需要的 GPU 数量 (4 或 8)"
    echo ""
    echo "示例:"
    echo "  $0 4      # 使用 4 张 GPU"
    echo "  $0 8      # 使用 8 张 GPU"
    echo ""
    echo "环境变量:"
    echo "  GPU_MEMORY_THRESHOLD_MB   显存空闲阈值 (默认: 500 MB)"
    echo "  GPU_UTIL_THRESHOLD        利用率空闲阈值 (默认: 10%)"
    echo "  POLL_INTERVAL             轮询间隔 (默认: 30 秒)"
    echo "  MAX_WAIT_HOURS            最大等待时间 (默认: 48 小时)"
    echo "  SE_REWARD_BASE_PORT       Reward Server 基础端口 (默认: 5000)"
}

log_info() {
    echo "[INFO] $(date '+%H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date '+%H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%H:%M:%S') - $1" >&2
}

# =============================================================================
# GPU 检测函数
# =============================================================================

get_gpu_info() {
    # 获取所有 GPU 的信息
    # 输出格式: GPU_ID,MEMORY_USED_MB,MEMORY_TOTAL_MB,GPU_UTIL%
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

get_idle_gpus() {
    # 获取空闲 GPU 列表
    # 返回: 空格分隔的 GPU ID 列表
    local mem_threshold=$1
    local util_threshold=$2
    local idle_gpus=""
    
    while IFS=',' read -r gpu_id mem_used mem_total gpu_util; do
        if [ -z "$gpu_id" ]; then
            continue
        fi
        
        # 检查显存占用和利用率
        if [ "$mem_used" -lt "$mem_threshold" ] && [ "$gpu_util" -lt "$util_threshold" ]; then
            if [ -z "$idle_gpus" ]; then
                idle_gpus="$gpu_id"
            else
                idle_gpus="$idle_gpus $gpu_id"
            fi
        fi
    done < <(get_gpu_info)
    
    echo "$idle_gpus"
}

count_idle_gpus() {
    local idle_gpus
    idle_gpus=$(get_idle_gpus "$GPU_MEMORY_THRESHOLD_MB" "$GPU_UTIL_THRESHOLD")
    if [ -z "$idle_gpus" ]; then
        echo "0"
    else
        echo "$idle_gpus" | wc -w | tr -d ' '
    fi
}

select_gpus() {
    # 选择指定数量的空闲 GPU
    local n_needed=$1
    local idle_gpus
    idle_gpus=$(get_idle_gpus "$GPU_MEMORY_THRESHOLD_MB" "$GPU_UTIL_THRESHOLD")
    
    # 选择前 n_needed 个
    echo "$idle_gpus" | tr ' ' '\n' | head -n "$n_needed" | tr '\n' ' ' | sed 's/ $//'
}

print_gpu_status() {
    echo "当前 GPU 状态:"
    echo "----------------------------------------"
    printf "%-5s %-12s %-12s %-10s %-8s\n" "GPU" "显存使用" "显存总量" "利用率" "状态"
    echo "----------------------------------------"
    
    while IFS=',' read -r gpu_id mem_used mem_total gpu_util; do
        if [ -z "$gpu_id" ]; then
            continue
        fi
        
        local status="占用"
        if [ "$mem_used" -lt "$GPU_MEMORY_THRESHOLD_MB" ] && [ "$gpu_util" -lt "$GPU_UTIL_THRESHOLD" ]; then
            status="空闲"
        fi
        
        printf "%-5s %-12s %-12s %-10s %-8s\n" \
            "$gpu_id" "${mem_used}MB" "${mem_total}MB" "${gpu_util}%" "$status"
    done < <(get_gpu_info)
    
    echo "----------------------------------------"
    echo "空闲判定阈值: 显存 < ${GPU_MEMORY_THRESHOLD_MB}MB, 利用率 < ${GPU_UTIL_THRESHOLD}%"
}

# =============================================================================
# 等待 GPU 函数
# =============================================================================

wait_for_gpus() {
    local n_needed=$1
    local max_iterations=$((MAX_WAIT_HOURS * 3600 / POLL_INTERVAL))
    local iteration=0
    
    log_info "等待 $n_needed 张空闲 GPU..."
    log_info "检查间隔: ${POLL_INTERVAL}秒, 最大等待: ${MAX_WAIT_HOURS}小时"
    echo ""
    
    while true; do
        local n_idle
        n_idle=$(count_idle_gpus)
        
        log_info "当前空闲 GPU: $n_idle / 需要: $n_needed"
        
        if [ "$n_idle" -ge "$n_needed" ]; then
            log_info "GPU 资源满足要求!"
            return 0
        fi
        
        iteration=$((iteration + 1))
        if [ "$iteration" -ge "$max_iterations" ]; then
            log_error "等待超时 (${MAX_WAIT_HOURS}小时)，无法获取足够的 GPU"
            return 1
        fi
        
        # 显示详细状态（每10次显示一次）
        if [ $((iteration % 10)) -eq 1 ]; then
            echo ""
            print_gpu_status
            echo ""
        fi
        
        log_info "等待 ${POLL_INTERVAL} 秒后重试..."
        sleep "$POLL_INTERVAL"
    done
}

# =============================================================================
# 环境变量设置函数
# =============================================================================

setup_gpu_env() {
    local n_gpus=$1
    local selected_gpus=$2
    
    # 将空格分隔转换为数组
    local gpu_array=($selected_gpus)
    local n_selected=${#gpu_array[@]}
    
    if [ "$n_selected" -lt "$n_gpus" ]; then
        log_error "选中的 GPU 数量不足: $n_selected < $n_gpus"
        return 1
    fi
    
    # 计算分配
    local half=$((n_gpus / 2))
    
    # Challenger GPUs (前半部分)
    local challenger_gpus=""
    for ((i=0; i<half; i++)); do
        if [ -z "$challenger_gpus" ]; then
            challenger_gpus="${gpu_array[$i]}"
        else
            challenger_gpus="${challenger_gpus},${gpu_array[$i]}"
        fi
    done
    
    # Reward GPUs (后半部分)
    local reward_gpus=""
    for ((i=half; i<n_gpus; i++)); do
        if [ -z "$reward_gpus" ]; then
            reward_gpus="${gpu_array[$i]}"
        else
            reward_gpus="${reward_gpus},${gpu_array[$i]}"
        fi
    done
    
    # Solver GPUs (全部)
    local solver_gpus=""
    for ((i=0; i<n_gpus; i++)); do
        if [ -z "$solver_gpus" ]; then
            solver_gpus="${gpu_array[$i]}"
        else
            solver_gpus="${solver_gpus},${gpu_array[$i]}"
        fi
    done
    
    # Gen Query GPUs (全部)
    local gen_query_gpus="$solver_gpus"
    
    # Reward Ports (与 reward GPU 数量相同)
    local n_reward_servers=$half
    local reward_ports=""
    for ((i=0; i<n_reward_servers; i++)); do
        local port=$((REWARD_BASE_PORT + i))
        if [ -z "$reward_ports" ]; then
            reward_ports="$port"
        else
            reward_ports="${reward_ports},${port}"
        fi
    done
    
    # 设置环境变量
    export SE_N_GPUS="$n_gpus"
    export SE_GPU_IDS="$solver_gpus"
    export SE_CHALLENGER_GPUS="$challenger_gpus"
    export SE_REWARD_GPUS="$reward_gpus"
    export SE_SOLVER_GPUS="$solver_gpus"
    export SE_GEN_QUERY_GPUS="$gen_query_gpus"
    export SE_REWARD_PORTS="$reward_ports"
    export SE_REWARD_BASE_PORT="$REWARD_BASE_PORT"
    export SE_N_CHALLENGER_GPUS="$half"
    export SE_N_REWARD_GPUS="$half"
    export SE_N_SOLVER_GPUS="$n_gpus"
    export SE_N_REWARD_SERVERS="$n_reward_servers"
    
    # 导出路径环境变量
    export SE_BASE_DIR
    export SE_PROJECT_NAME
    export SE_CODE_MODULE
    export SE_WORKING_DIR
    export SE_MODEL_DIR
    export SE_DATA_DIR
    export SE_SAVED_RESULTS_DIR
    export SE_CHALLENGER_DIR
    export SE_SOLVER_DIR
    export SE_TENSORBOARD_DIR
    export SE_PROMPT_DIR
    
    # 打印配置
    echo ""
    echo "GPU 分配配置:"
    echo "=============================================="
    echo "总 GPU 数量:       $SE_N_GPUS"
    echo "选中 GPU IDs:      $SE_GPU_IDS"
    echo ""
    echo "Challenger GPUs:   $SE_CHALLENGER_GPUS (共 $SE_N_CHALLENGER_GPUS 张)"
    echo "Reward GPUs:       $SE_REWARD_GPUS (共 $SE_N_REWARD_GPUS 张)"
    echo "Solver GPUs:       $SE_SOLVER_GPUS (共 $SE_N_SOLVER_GPUS 张)"
    echo "Gen Query GPUs:    $SE_GEN_QUERY_GPUS"
    echo ""
    echo "Reward Ports:      $SE_REWARD_PORTS"
    echo "Reward Base Port:  $SE_REWARD_BASE_PORT"
    echo "=============================================="
    echo ""
    echo "路径配置:"
    echo "=============================================="
    echo "基础目录:          $SE_BASE_DIR"
    echo "项目名称:          $SE_PROJECT_NAME"
    echo "代码模块:          $SE_CODE_MODULE"
    echo "工作目录:          $SE_WORKING_DIR"
    echo "模型目录:          $SE_MODEL_DIR"
    echo "数据目录:          $SE_DATA_DIR"
    echo "结果保存目录:      $SE_SAVED_RESULTS_DIR"
    echo "Prompt目录:        $SE_PROMPT_DIR"
    echo "=============================================="
    echo ""
}

# =============================================================================
# 主函数
# =============================================================================

main() {
    print_banner
    
    # 参数检查
    if [ $# -lt 1 ]; then
        print_usage
        exit 1
    fi
    
    local n_gpus=$1
    
    # 验证参数
    if ! [[ "$n_gpus" =~ ^[0-9]+$ ]]; then
        log_error "n_gpus 必须是正整数: $n_gpus"
        print_usage
        exit 1
    fi
    
    if [ "$n_gpus" -lt 2 ]; then
        log_error "n_gpus 必须至少为 2 (需要分配给 challenger 和 reward)"
        exit 1
    fi
    
    if [ $((n_gpus % 2)) -ne 0 ]; then
        log_error "n_gpus 必须是偶数 (需要平分给 challenger 和 reward): $n_gpus"
        exit 1
    fi
    
    log_info "请求 GPU 数量: $n_gpus"
    echo ""
    
    # 显示当前状态
    print_gpu_status
    echo ""
    
    # 等待足够的 GPU
    if ! wait_for_gpus "$n_gpus"; then
        log_error "无法获取足够的 GPU，退出"
        exit 1
    fi
    
    # 选择 GPU
    local selected_gpus
    selected_gpus=$(select_gpus "$n_gpus")
    log_info "选中 GPU: $selected_gpus"
    
    # 设置环境变量
    setup_gpu_env "$n_gpus" "$selected_gpus"
    
    # 调用 main.sh
    log_info "启动训练..."
    echo ""
    echo "=============================================="
    echo "  开始执行 main.sh"
    echo "=============================================="
    echo ""
    
    # 执行 main.sh
    bash "${SCRIPT_DIR}/main.sh"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "训练完成!"
    else
        log_error "训练失败，退出码: $exit_code"
    fi
    
    exit $exit_code
}

# =============================================================================
# 入口
# =============================================================================

main "$@"
