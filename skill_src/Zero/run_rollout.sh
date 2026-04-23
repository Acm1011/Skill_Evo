#!/usr/bin/env bash
# =============================================================================
# run_rollout.sh - 启动 vLLM server 并执行 DeepMath-103K 独立 rollout
# =============================================================================
#
# 用法:
#   export ROLLOUT_MODEL_PATH=/path/to/model
#   ./run_rollout.sh <step> <batch_size> [output_dir]
#
# 示例:
#   ./run_rollout.sh 2 32 ./results           # step=2, batch_size=32 (处理 0-64 条数据，共 64 条)
#   ./run_rollout.sh 4 32 ./results           # step=4, batch_size=32 (处理 0-128 条数据，共 128 条)
#
# 环境变量:
#   ROLLOUT_MODEL_PATH       - 模型路径（必须）
#   SE_N_GPUS              - GPU 数量（可选，默认 2）
#   SE_GPU_IDS             - GPU ID 列表（可选，默认 0,1）
#   VLLM_HTTP_BASE_PORT    - vLLM 起始端口（可选，默认 8760）
#   ROLLOUT_HTTP_BASE_PORT - rollout_http_client 起始端口（可选，默认 8860）
#   SERVED_MODEL_NAME      - vllm serve 的模型名（可选，默认 default）
#   VLLM_DTYPE             - vllm dtype（可选，默认 auto）
#   VLLM_HTTP_TIMEOUT       - 单条 vLLM completions 读超时秒数（可选，默认 600）
#   VLLM_HTTP_MAX_CONCURRENT - 每 rollout_http_client 进程对 vLLM 的最大并发请求数（可选，默认 32）
#                              多分片并行时勿过大，否则排队过久触发 ReadTimeout
#   VLLM_HTTP_MAX_RETRIES  - HTTP 重试次数（可选，默认 8）
#   VLLM_HTTP_RETRY_DELAY  - HTTP 重试延迟（可选，默认 2）
#
# =============================================================================

set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ========================== 解析位置参数 ==========================
STEP="${1:-}"
BATCH_SIZE="${2:-}"
OUTPUT_DIR="${3:-./rollout_results}"

if [ -z "$STEP" ] || [ -z "$BATCH_SIZE" ]; then
    echo "用法: $0 <step> <batch_size> [output_dir]"
    echo ""
    echo "参数:"
    echo "  step         - 总步数，数据总量 = step * batch_size（每次从索引 0 开始处理）"
    echo "  batch_size   - 每批次大小"
    echo "  output_dir   - 输出目录（可选，默认 ./rollout_results）"
    echo ""
    echo "环境变量:"
    echo "  ROLLOUT_MODEL_PATH       - 模型路径（必须）"
    echo "  SE_N_GPUS                - GPU 数量（可选，默认 2）"
    echo "  SE_GPU_IDS               - GPU ID 列表（可选，默认 0,1）"
    echo "  VLLM_HTTP_BASE_PORT      - vLLM 起始端口（可选，默认 8760）"
    echo "  ROLLOUT_HTTP_BASE_PORT   - rollout_http_client 起始端口（可选，默认 8860）"
    exit 1
fi

# ========================== 验证必要环境变量 ==========================
ROLLOUT_MODEL_PATH="${ROLLOUT_MODEL_PATH:-/home/ycy/sdi/models/Qwen3-4B-Instruct-2507}"
if [ -z "$ROLLOUT_MODEL_PATH" ]; then
    echo "错误: 请设置环境变量 ROLLOUT_MODEL_PATH 为模型路径"
    exit 1
fi

if [ ! -d "$ROLLOUT_MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $ROLLOUT_MODEL_PATH"
    exit 1
fi

# ========================== 配置参数 ==========================
SE_N_GPUS="${SE_N_GPUS:-4}"
SE_GPU_IDS="${SE_GPU_IDS:-0,1,2,3}"
VLLM_HTTP_BASE_PORT="${VLLM_HTTP_BASE_PORT:-8760}"
ROLLOUT_HTTP_BASE_PORT="${ROLLOUT_HTTP_BASE_PORT:-8860}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-default}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_HTTP_TIMEOUT="${VLLM_HTTP_TIMEOUT:-600}"
VLLM_HTTP_MAX_CONCURRENT="${VLLM_HTTP_MAX_CONCURRENT:-32}"
VLLM_HTTP_MAX_RETRIES="${VLLM_HTTP_MAX_RETRIES:-8}"
VLLM_HTTP_RETRY_DELAY="${VLLM_HTTP_RETRY_DELAY:-2}"

SE_CODE_MODULE="${SE_CODE_MODULE:-skill_src}"
PYTHON="${PYTHON:-python3}"

# 输出文件命名（反映总数据量）
TOTAL_SAMPLES=$((STEP * BATCH_SIZE))
OUTPUT_NAME="deepmath_n${TOTAL_SAMPLES}_bs${BATCH_SIZE}.jsonl"

# ========================== 路径配置 ==========================
# 确保 OUTPUT_DIR 是绝对路径（相对于当前工作目录），避免后续 cd 后路径失效
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    # 去掉开头的 ./ 如果存在，然后拼接为绝对路径
    REL_DIR="${OUTPUT_DIR#./}"
    OUTPUT_DIR="$(pwd)/$REL_DIR"
fi
START_VLLM_SH="${PROJECT_DIR}/${SE_CODE_MODULE}/start_vllm_http_servers.sh"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ========================== 启动日志 ==========================
echo "=============================================="
echo "  DeepMath-103K Rollout 启动"
echo "=============================================="
echo "  step:           $STEP"
echo "  batch_size:     $BATCH_SIZE"
echo "  output_dir:     $OUTPUT_DIR"
echo "  output_name:    $OUTPUT_NAME"
echo "  model_path:     $ROLLOUT_MODEL_PATH"
echo "  n_gpus:         $SE_N_GPUS"
echo "  gpu_ids:        $SE_GPU_IDS"
echo "  vllm_port:      $VLLM_HTTP_BASE_PORT"
echo "  rollout_port:   $ROLLOUT_HTTP_BASE_PORT"
echo "=============================================="
echo ""

# 检查必要脚本
if [ ! -f "$START_VLLM_SH" ]; then
    echo "错误: 未找到 start_vllm_http_servers.sh: $START_VLLM_SH"
    exit 1
fi

# ========================== 清理函数 ==========================
VLLM_LAUNCHER_PID=""
HTTP_CLIENT_PIDS=()

cleanup_all() {
    local s=$?
    echo ""
    echo "========== 清理进程 =========="
    # 结束 rollout_http_client
    for pid in "${HTTP_CLIENT_PIDS[@]:-}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    # 结束 vLLM 启动脚本
    if [ -n "${VLLM_LAUNCHER_PID}" ] && kill -0 "${VLLM_LAUNCHER_PID}" 2>/dev/null; then
        echo "清理 vLLM 启动脚本 (PID: ${VLLM_LAUNCHER_PID})..."
        kill "${VLLM_LAUNCHER_PID}" 2>/dev/null || true
        wait "${VLLM_LAUNCHER_PID}" 2>/dev/null || true
    fi
    echo "清理完成"
    exit "${s:-0}"
}
trap cleanup_all EXIT INT TERM

# ========================== Step 1: 启动 vLLM HTTP servers ==========================
echo ""
echo "========== Step 1: 启动 vLLM HTTP servers =========="

cd "$PROJECT_DIR"
if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${PROJECT_DIR}"
else
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
fi

export BASE_PORT="${VLLM_HTTP_BASE_PORT}"
export DTYPE="${VLLM_DTYPE}"
export SERVED_MODEL_NAME

echo "启动 vLLM servers (GPUs: $SE_GPU_IDS, Base port: $VLLM_HTTP_BASE_PORT)..."
bash "$START_VLLM_SH" "$SE_GPU_IDS" "$ROLLOUT_MODEL_PATH" &
VLLM_LAUNCHER_PID=$!
echo "vLLM 启动脚本 PID: ${VLLM_LAUNCHER_PID}"

# 等待 vLLM servers 就绪
MAX_WAIT=300
echo "等待 vLLM servers 就绪..."
IFS=',' read -r -a GPU_ARRAY <<< "$SE_GPU_IDS"
for ((i=0; i<SE_N_GPUS; i++)); do
    port=$((VLLM_HTTP_BASE_PORT + i))
    waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            echo "  ✓ vLLM server $i (port $port) 就绪 (${waited}s)"
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done
    if [ $waited -ge $MAX_WAIT ]; then
        echo "错误: vLLM server $i (port $port) 启动超时"
        exit 1
    fi
done

# 构建 vLLM URLs CSV
VLLM_URLS_CSV=""
for ((i=0; i<SE_N_GPUS; i++)); do
    vport=$((VLLM_HTTP_BASE_PORT + i))
    VLLM_URLS_CSV="${VLLM_URLS_CSV:+$VLLM_URLS_CSV,}http://127.0.0.1:${vport}"
done

echo "vLLM URLs: $VLLM_URLS_CSV"

# ========================== Step 2: 启动 rollout_http_client ==========================
echo ""
echo "========== Step 2: 启动 rollout_http_client =========="

echo "启动 solver_offline_rollout_http_client (每卡一个)..."
for ((i=0; i<SE_N_GPUS; i++)); do
    cport=$((ROLLOUT_HTTP_BASE_PORT + i))
    log="${LOG_DIR}/rollout_http_client_${i}_port${cport}.log"
    echo "  client i=$i listen=:${cport} -> ${log}"
    "$PYTHON" -m "${SE_CODE_MODULE}.solver_offline_rollout_http_client" \
        --vllm-urls "$VLLM_URLS_CSV" \
        --model "$ROLLOUT_MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$cport" \
        --timeout "$VLLM_HTTP_TIMEOUT" \
        --max-concurrent "$VLLM_HTTP_MAX_CONCURRENT" \
        --max-retries "$VLLM_HTTP_MAX_RETRIES" \
        --retry-delay "$VLLM_HTTP_RETRY_DELAY" \
        >>"$log" 2>&1 &
    HTTP_CLIENT_PIDS+=($!)
done

# 等待 rollout_http_client 就绪
echo "等待 rollout_http_client 就绪..."
for ((i=0; i<SE_N_GPUS; i++)); do
    cport=$((ROLLOUT_HTTP_BASE_PORT + i))
    waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if curl -sf "http://127.0.0.1:${cport}/health" > /dev/null 2>&1; then
            echo "  ✓ rollout_http_client $i (port $cport) 就绪 (${waited}s)"
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    if [ $waited -ge $MAX_WAIT ]; then
        echo "错误: rollout_http_client $i (port $cport) 启动超时"
        exit 1
    fi
done

# 构建 rollout server URLs（供 driver 使用）
SE_ROLLOUT_HOST="${SE_ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_URLS=""
for ((i=0; i<SE_N_GPUS; i++)); do
    cport=$((ROLLOUT_HTTP_BASE_PORT + i))
    ROLLOUT_URLS="${ROLLOUT_URLS:+${ROLLOUT_URLS} }http://${SE_ROLLOUT_HOST}:${cport}"
done
export SE_ROLLOUT_SERVER_URLS="$ROLLOUT_URLS"
echo "Rollout server URLs: $SE_ROLLOUT_SERVER_URLS"

# ========================== Step 3: 执行 rollout ==========================
echo ""
echo "========== Step 3: 执行 DeepMath rollout =========="

echo "调用 rollout_deepmath.py..."
echo "  step: $STEP"
echo "  batch_size: $BATCH_SIZE"
echo "  output_dir: $OUTPUT_DIR"
echo "  output_name: $OUTPUT_NAME"

"$PYTHON" -m "${SE_CODE_MODULE}.rollout_deepmath" \
    --step "$STEP" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --output-name "$OUTPUT_NAME" \
    --rollout-n 1 \
    --max-tokens 4096 \
    --temperature 0 \
    --skill-type "skill_generation_v1"
    # --top-p 1.0 \
    

ROLLOUT_EXIT_CODE=$?

if [ $ROLLOUT_EXIT_CODE -ne 0 ]; then
    echo "错误: rollout 失败，退出码: $ROLLOUT_EXIT_CODE"
    exit $ROLLOUT_EXIT_CODE
fi

# ========================== 完成 ==========================
echo ""
echo "=============================================="
echo "  Rollout 完成!"
echo "=============================================="
echo "  输出文件: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "  日志目录: ${LOG_DIR}"
echo "=============================================="
