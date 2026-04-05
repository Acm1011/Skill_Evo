#!/usr/bin/env bash
# =============================================================================
# run_with_gpus.sh - GPU-aware launcher for Self-evolving-Agent training
# =============================================================================
#
# 用法:
#   ./run_with_gpus.sh <n_gpus>
#   ./run_with_gpus.sh 2      # 使用 2 张 GPU
#   ./run_with_gpus.sh 4      # 使用 4 张 GPU
#   ./run_with_gpus.sh 8      # 使用 8 张 GPU
#
# 功能:
#   1. 检测当前服务器 GPU 状态（显存占用、利用率）
#   2. 等待直到空闲 GPU 数量满足 n_gpus
#   3. 选择空闲 GPU 并设置环境变量
#   4. 调用 main.sh 启动训练；或第二参数 test-rollout 时改为运行 test_solver_rollout.sh
#
# 环境变量输出:
#   SE_N_GPUS              - 总 GPU 数量（支持 2、4 或 8）
#   SE_GPU_IDS             - 选中的 GPU ID 列表（逗号分隔）
#   SE_SYNTHESIZER_GPUS    - Synthesizer RL 训练使用的 GPU（前半部分）
#   SE_N_SYNTHESIZER_GPUS  - Synthesizer 使用的 GPU 数量
#   SE_SOLVER_GPUS         - Solver 训练使用的 GPU（全部）
#   SE_GEN_QUERY_GPUS      - 查询生成使用的 GPU
#   Offline Rollout（solver_offline_rollout_server / solver_offline_driver）:
#   ROLLOUT_BASE_PORT / SE_ROLLOUT_BASE_PORT - Rollout HTTP 基础端口（默认 8760）
#   SE_ROLLOUT_N_SERVERS   - 与 GPU 数相同，每张卡对应一个 server
#   SE_ROLLOUT_PORTS       - Rollout 端口列表（逗号分隔）
#   SE_ROLLOUT_HOST        - Rollout URL 主机（默认 127.0.0.1）
#   SE_ROLLOUT_SERVER_URLS - 空格分隔，供 driver: --server-urls $SE_ROLLOUT_SERVER_URLS
#   test-rollout 可选 --model；未传时由 test_solver_rollout.sh 使用 DEFAULT_ROLLOUT_MODEL（见该脚本）
#   SE_SKILL_SAVED_ROOT    - 实验产物根（默认 /home/ycy/sdi/skill_saved）；完整路径由 main.sh 在 exp_name 确定后设置:
#                             \$SE_SKILL_SAVED_ROOT/<exp_name>/Synthesizer|Solver
#   SE_Synthsizer_DIR 等   - 训练时由 main.sh export；test-rollout 不经 main 时见各测试脚本内默认
#   （Ray 临时目录在 main_synthesizer：默认 $SE_RAY_TEMP_ROOT/r-$USER，SE_RAY_TEMP_ROOT 默认 /home/ycy/sdi）
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
GPU_MEMORY_THRESHOLD_MB="${GPU_MEMORY_THRESHOLD_MB:-40000}"      # 显存占用低于此值视为空闲(MB)
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-20}"                  # 利用率低于此值视为空闲(%)

# 轮询配置
POLL_INTERVAL="${POLL_INTERVAL:-1800}"                            # 检查间隔（秒）
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-48}"                          # 最大等待时间（小时）

# Rollout Server 配置
ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT:-8760}"                 # Rollout Server 基础端口

# =============================================================================
# 路径配置 - 通过环境变量统一管理
# =============================================================================

# 基础目录 (可通过环境变量覆盖)
SE_BASE_DIR="${SE_BASE_DIR:-/home/ycy/sdi}"
SE_PROJECT_NAME="${SE_PROJECT_NAME:-Skill_Evo}"
SE_CODE_MODULE="${SE_CODE_MODULE:-skill_src}"

# 派生路径
SE_WORKING_DIR="${SE_BASE_DIR}/${SE_PROJECT_NAME}"
SE_MODEL_DIR="${SE_MODEL_DIR:-${SE_BASE_DIR}/models}"
SE_DATA_DIR="${SE_DATA_DIR:-${SE_BASE_DIR}/data}"
SE_SKILL_SAVED_ROOT="${SE_SKILL_SAVED_ROOT:-/home/ycy/sdi/skill_saved}"
SE_PROMPT_DIR="${SE_WORKING_DIR}/${SE_CODE_MODULE}"
SE_EMBEDDING_CACHE_PATH="${SE_EMBEDDING_CACHE_PATH:-${SE_SKILL_SAVED_ROOT}/embedding_cache}"
mkdir -p "${SE_SKILL_SAVED_ROOT}"
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
    echo "用法: $0 <n_gpus> [test-rollout|--test-rollout] [--model <本地模型目录或 HF id>]"
    echo ""
    echo "参数:"
    echo "  n_gpus       需要的 GPU 数量，支持 2、4 或 8"
    echo "  test-rollout 可选。仅做 GPU 等待与 setup_gpu_env，然后运行 Zero/test_solver_rollout.sh，不执行 main.sh"
    echo "  --model PATH 与 test-rollout 联用，可选；不传则使用 test_solver_rollout.sh 的默认（DEFAULT_ROLLOUT_MODEL / SE_MODEL_DIR）"
    echo ""
    echo "示例:"
    echo "  $0 2      # 使用 2 张 GPU（Synthesizer 1 + Rollout 1）"
    echo "  $0 4      # 使用 4 张 GPU"
    echo "  $0 8      # 使用 8 张 GPU"
    echo "  $0 2 test-rollout                    # 分配 GPU 后跑 rollout 测试（默认模型）"
    echo "  $0 2 test-rollout --model /path/to/model"
    echo ""
    echo "环境变量:"
    echo "  GPU_MEMORY_THRESHOLD_MB   显存空闲阈值 (默认: 500 MB)"
    echo "  GPU_UTIL_THRESHOLD        利用率空闲阈值 (默认: 10%)"
    echo "  POLL_INTERVAL             轮询间隔 (默认: 30 秒)"
    echo "  MAX_WAIT_HOURS            最大等待时间 (默认: 48 小时)"
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
    
    # Synthesizer GPUs (前半部分，用于 RL 训练)
    local synthesizer_gpus=""
    for ((i=0; i<half; i++)); do
        if [ -z "$synthesizer_gpus" ]; then
            synthesizer_gpus="${gpu_array[$i]}"
        else
            synthesizer_gpus="${synthesizer_gpus},${gpu_array[$i]}"
        fi
    done
    
    # Rollout Server GPUs (后半部分，用于 offline rollout + reward 评估)
    local rollout_gpus=""
    for ((i=half; i<n_gpus; i++)); do
        if [ -z "$rollout_gpus" ]; then
            rollout_gpus="${gpu_array[$i]}"
        else
            rollout_gpus="${rollout_gpus},${gpu_array[$i]}"
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
    
    # 设置环境变量
    export SE_N_GPUS="$n_gpus"
    export SE_GPU_IDS="$solver_gpus"
    export SE_SYNTHESIZER_GPUS="$synthesizer_gpus"
    export SE_N_SYNTHESIZER_GPUS="$half"
    export SE_SOLVER_GPUS="$solver_gpus"
    export SE_GEN_QUERY_GPUS="$gen_query_gpus"
    export SE_N_SOLVER_GPUS="$n_gpus"
    
    # Offline Rollout（每张 GPU 一个 solver_offline_rollout_server，端口 ROLLOUT_BASE_PORT + i）
    local rb_port="${ROLLOUT_BASE_PORT:-8760}"
    local rh="${SE_ROLLOUT_HOST:-127.0.0.1}"
    local rollout_ports=""
    local rollout_urls=""
    for ((i=0; i<n_gpus; i++)); do
        local rp=$((rb_port + i))
        if [ -z "$rollout_ports" ]; then
            rollout_ports="$rp"
        else
            rollout_ports="${rollout_ports},${rp}"
        fi
        local url="http://${rh}:${rp}"
        if [ -z "$rollout_urls" ]; then
            rollout_urls="$url"
        else
            rollout_urls="${rollout_urls} $url"
        fi
    done
    export ROLLOUT_BASE_PORT="$rb_port"
    export SE_ROLLOUT_BASE_PORT="$rb_port"
    export SE_ROLLOUT_HOST="$rh"
    export SE_ROLLOUT_N_SERVERS="$n_gpus"
    export SE_ROLLOUT_PORTS="$rollout_ports"
    export SE_ROLLOUT_SERVER_URLS="$rollout_urls"
    # Rollout 模型路径不在此写死；由 test-rollout 的 --model 或用户自行 export 后启动 server
    
    # 导出路径环境变量
    export SE_BASE_DIR
    export SE_PROJECT_NAME
    export SE_CODE_MODULE
    export SE_WORKING_DIR
    export SE_MODEL_DIR
    export SE_DATA_DIR
    export SE_SKILL_SAVED_ROOT
    export SE_PROMPT_DIR
    export SE_EMBEDDING_CACHE_PATH
    # 打印配置
    echo ""
    echo "GPU 分配配置:"
    echo "=============================================="
    echo "总 GPU 数量:       $SE_N_GPUS"
    echo "选中 GPU IDs:      $SE_GPU_IDS"
    echo ""
    echo "Synthesizer GPUs:  $SE_SYNTHESIZER_GPUS (共 $SE_N_SYNTHESIZER_GPUS 张)"
    echo "Solver GPUs:       $SE_SOLVER_GPUS (共 $SE_N_SOLVER_GPUS 张)"
    echo "Gen Query GPUs:    $SE_GEN_QUERY_GPUS"
    echo ""
    echo "Offline Rollout (solver_offline_driver / rollout_server):"
    echo "  SE_ROLLOUT_BASE_PORT=$SE_ROLLOUT_BASE_PORT  SE_ROLLOUT_N_SERVERS=$SE_ROLLOUT_N_SERVERS"
    echo "  SE_ROLLOUT_PORTS=$SE_ROLLOUT_PORTS"
    echo "  SE_ROLLOUT_SERVER_URLS=$SE_ROLLOUT_SERVER_URLS"
    echo "  （Rollout 模型: 未在此设置；test-rollout 请传 --model，或先 export ROLLOUT_SERVER_MODEL）"
    echo "  一键启动（推荐）: export ROLLOUT_SERVER_MODEL=... && bash ${SCRIPT_DIR}/start_rollout_servers.sh"
    echo "  手动示例（每卡一进程，i 与 GPU 顺序对齐）:"
    echo "    IFS=',' read -ra _RGPU <<< \"\$SE_GPU_IDS\""
    echo "    for i in \$(seq 0 \$((SE_N_GPUS - 1))); do"
    echo "      CUDA_VISIBLE_DEVICES=\${_RGPU[\$i]} python -m ${SE_CODE_MODULE}.solver_offline_rollout_server \\"
    echo "        --model \"<你的模型路径>\" --host 0.0.0.0 --port \$((SE_ROLLOUT_BASE_PORT + i)) &"
    echo "    done"
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
    echo "skill_saved 根目录: $SE_SKILL_SAVED_ROOT"
    echo "  （训练时由 main.sh 设置 SE_SAVED_RESULTS_DIR 等: \$SE_SKILL_SAVED_ROOT/<exp_name>/...）"
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
    shift
    local mode="train"
    local rollout_model=""
    while [ $# -gt 0 ]; do
        case "$1" in
            test-rollout|--test-rollout)
                mode="test-rollout"
                shift
                ;;
            --model)
                if [ $# -lt 2 ]; then
                    log_error "--model 需要路径参数"
                    exit 1
                fi
                rollout_model="$2"
                shift 2
                ;;
            *)
                log_error "未知参数: $1（可选: test-rollout, --model <path>）"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # 验证参数
    if ! [[ "$n_gpus" =~ ^[0-9]+$ ]]; then
        log_error "n_gpus 必须是正整数: $n_gpus"
        print_usage
        exit 1
    fi
    
    if [ "$n_gpus" -ne 2 ] && [ "$n_gpus" -ne 4 ] && [ "$n_gpus" -ne 8 ]; then
        log_error "n_gpus 仅支持 2、4 或 8，当前: $n_gpus"
        print_usage
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
    
    if [ "$mode" = "test-rollout" ]; then
        log_info "启动 offline rollout 测试（test_solver_rollout.sh）..."
        echo ""
        echo "=============================================="
        echo "  执行 test_solver_rollout.sh"
        echo "=============================================="
        echo ""
        if [ -n "${rollout_model}" ]; then
            bash "${SCRIPT_DIR}/test_solver_rollout.sh" --model "${rollout_model}"
        else
            bash "${SCRIPT_DIR}/test_solver_rollout.sh"
        fi
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log_info "rollout 测试完成"
        else
            log_error "rollout 测试失败，退出码: $exit_code"
        fi
        exit $exit_code
    fi
    
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
