#!/usr/bin/env bash
# =============================================================================
# start_rollout_servers.sh - 按 run_with_gpus.sh 分配的 GPU 每张卡起一个 offline rollout server
# =============================================================================
#
# 用法:
#   1) 先由 run_with_gpus.sh 选好 GPU 并 export 环境变量（或手动 export 下方变量），再执行本脚本；
#   2) 本脚本前台阻塞，Ctrl+C 会尝试结束已启动的 server 进程。
#   3) 也可由 test_solver_rollout.sh 在后台启动本脚本（bash start_rollout_servers.sh &），
#      任务结束后对其发 SIGTERM，同样会触发 trap 清理子进程。
#
# 依赖环境变量（与 run_with_gpus.sh 一致）:
#   SE_N_GPUS              与选中 GPU 数量一致（4 或 8）
#   SE_GPU_IDS             逗号分隔的物理 GPU ID，与 SE_N_GPUS 一一对应
#   SE_ROLLOUT_BASE_PORT   或 ROLLOUT_BASE_PORT，默认 8760，第 i 个 server 端口为 base+i
#   ROLLOUT_SERVER_MODEL   或 SE_ROLLOUT_MODEL，vLLM 模型路径
#   SE_WORKING_DIR         项目根目录（python -m skill_src...），未设则取当前仓库推断
#   SE_CODE_MODULE         默认 skill_src
#
# 可选:
#   SE_ROLLOUT_LOG_DIR     日志目录，默认 ${SE_ROLLOUT_DIR}/logs/rollout_servers
#   ROLLOUT_SERVER_GPU_UTIL  传给 server 的 --gpu-utilization，默认 0.95
#
# =============================================================================
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 仓库根: skill_src/Zero -> ../..
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SE_CODE_MODULE="${SE_CODE_MODULE:-skill_src}"
SE_WORKING_DIR="${SE_WORKING_DIR:-$REPO_ROOT}"
# 训练流程中由 main.sh 设置 SE_ROLLOUT_DIR；单独起 server 时回退到 skill_saved/<项目>/_standalone
SE_SKILL_SAVED_ROOT="${SE_SKILL_SAVED_ROOT:-/home/ycy/sdi/skill_saved}"
SE_PROJECT_NAME="${SE_PROJECT_NAME:-Skill_Evo}"
SE_ROLLOUT_DIR="${SE_ROLLOUT_DIR:-${SE_Synthsizer_DIR:-${SE_SKILL_SAVED_ROOT}/${SE_PROJECT_NAME}/_standalone/Synthesizer}}"

MODEL_CLI=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "[ERROR] --model 需要路径参数" >&2; exit 1; }
            MODEL_CLI="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [--model <path>]" >&2
            echo "  未传 --model 时使用 ROLLOUT_SERVER_MODEL 或 SE_ROLLOUT_MODEL" >&2
            exit 0
            ;;
        *)
            echo "[ERROR] 未知参数: $1" >&2
            exit 1
            ;;
    esac
done
MODEL="${MODEL_CLI:-${ROLLOUT_SERVER_MODEL:-${SE_ROLLOUT_MODEL:-}}}"
BASE_PORT="${SE_ROLLOUT_BASE_PORT:-${ROLLOUT_BASE_PORT:-8760}}"
N="${SE_N_GPUS:-}"
IDS="${SE_GPU_IDS:-}"
GPU_UTIL="${ROLLOUT_SERVER_GPU_UTIL:-0.7}"

usage() {
    echo "用法: 在已设置 run_with_gpus.sh 导出的环境变量后执行"
    echo "  $0 [--model <path>]"
    echo ""
    echo "必填: SE_N_GPUS, SE_GPU_IDS；模型路径由 --model 或 ROLLOUT_SERVER_MODEL / SE_ROLLOUT_MODEL"
    echo "示例（手动调试）:"
    echo "  export SE_N_GPUS=4 SE_GPU_IDS=0,1,2,3"
    echo "  $0 --model /path/to/model"
}

if [[ -z "${MODEL}" ]]; then
    echo "[ERROR] 请设置 ROLLOUT_SERVER_MODEL 或 SE_ROLLOUT_MODEL" >&2
    usage
    exit 1
fi
if [[ -z "${N}" || -z "${IDS}" ]]; then
    echo "[ERROR] 请设置 SE_N_GPUS 与 SE_GPU_IDS（可先运行 run_with_gpus.sh 的 setup 阶段或手动 export）" >&2
    usage
    exit 1
fi

IFS=',' read -ra _RAW_GPUS <<< "${IDS}"
GPUS=()
for x in "${_RAW_GPUS[@]}"; do
    x="${x// /}"
    [[ -n "$x" ]] && GPUS+=("$x")
done
if [[ ${#GPUS[@]} -ne "$N" ]]; then
    echo "[ERROR] SE_GPU_IDS 解析得到 ${#GPUS[@]} 个 ID，与 SE_N_GPUS=${N} 不一致" >&2
    exit 1
fi

LOG_DIR="${SE_ROLLOUT_LOG_DIR:-${SE_ROLLOUT_DIR}/logs/rollout_servers}"
mkdir -p "${LOG_DIR}"

PYTHON="${PYTHON:-python3}"

cd "${SE_WORKING_DIR}"
if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${SE_WORKING_DIR}"
else
    export PYTHONPATH="${SE_WORKING_DIR}:${PYTHONPATH}"
fi

echo "=============================================="
echo "  start_rollout_servers"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  WORKING_DIR: ${SE_WORKING_DIR}"
echo "  MODEL: ${MODEL}"
echo "  BASE_PORT: ${BASE_PORT}"
echo "  N: ${N}  GPUS: ${GPUS[*]}"
echo "  日志: ${LOG_DIR}/server_<idx>_gpu<gid>_p<port>.log"
echo "=============================================="

PIDS=()
cleanup() {
    local s=$?
    echo ""
    echo "[start_rollout_servers] 收到退出信号，正在停止子进程..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "[start_rollout_servers] 已退出"
    exit "${s:-0}"
}
trap cleanup INT TERM

for ((i = 0; i < N; i++)); do
    gpu="${GPUS[$i]}"
    port=$((BASE_PORT + i))
    log="${LOG_DIR}/server_${i}_gpu${gpu}_p${port}.log"
    echo "[start_rollout_servers] 启动 i=${i} CUDA_VISIBLE_DEVICES=${gpu} port=${port} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m "${SE_CODE_MODULE}.solver_offline_rollout_server" \
        --model "${MODEL}" \
        --host 0.0.0.0 \
        --port "${port}" \
        --gpu-utilization "${GPU_UTIL}" \
        >>"${log}" 2>&1 &
    PIDS+=($!)
done

echo "[start_rollout_servers] 已全部后台启动，共 ${#PIDS[@]} 个进程。Ctrl+C 结束全部。"
echo "[start_rollout_servers] 健康检查: curl -s http://127.0.0.1:${BASE_PORT}/health"

wait
