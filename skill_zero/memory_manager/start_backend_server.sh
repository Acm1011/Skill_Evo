#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════════════════
#  start_backend_server.sh
#  同时启动 retriever_server（embedding 服务）和 memory_server（skill 记忆服务）。
#
#  日志分别写入：
#    logs/retriever_server.log
#    logs/memory_server.log
#
#  停止方式：Ctrl-C 或 kill $$ 均会同时终止两个子进程。
# ════════════════════════════════════════════════════════════════════════════

# ─── retriever_server 配置 ───────────────────────────────────────────────────
RETRIEVER_HOST="127.0.0.1"
RETRIEVER_PORT=8766
EMBEDDING_MODEL="/home/xzs/data/model/Qwen3-Embedding-0.6B"
# 指定 GPU（PCI_BUS_ID 顺序，与 nvidia-smi 索引一致）；留空则使用所有可见 GPU
RETRIEVER_CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE=1
# vLLM 显存占用比例（0~1）；embedding 模型小，0.3 已足够，避免占满整张卡
GPU_MEMORY_UTILIZATION=0.1
INSTRUCT_TASK="Given a question, retrieve relevant skills that help answer it"
IDLE_TIMEOUT=300

# ─── memory_server 配置 ──────────────────────────────────────────────────────
MEMORY_HOST="0.0.0.0"
MEMORY_PORT=8765
MAX_CAPACITY=3
WARN_CAPACITY=2
RETRIEVE_MODE="hybrid"    # embedding | hybrid
RETRIEVE_LAMBDA=0.5
RETRIEVER_URL="http://127.0.0.1:${RETRIEVER_PORT}"
RETRIEVER_TIMEOUT=30
UPDATE_LAM=0.9
UPDATE_TAU=0.2
UPDATE_U_MIN=0.0
UPDATE_U_MAX=1.0
PERSIST_PATH="runs/skills_memory.jsonl"
SKILLS_JSONL=""              # 留空则跳过启动时预加载

# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."   # 切到项目根目录

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 混合 GPU 型号时按 PCI 总线顺序编号，与 nvidia-smi 保持一致
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# ─── 子进程清理 ──────────────────────────────────────────────────────────────
RETRIEVER_PID=""
MEMORY_PID=""

cleanup() {
    echo ""
    echo "[backend] Shutting down..."
    [ -n "${RETRIEVER_PID}" ] && kill "${RETRIEVER_PID}" 2>/dev/null && echo "[backend] retriever_server (pid=${RETRIEVER_PID}) stopped"
    [ -n "${MEMORY_PID}" ]    && kill "${MEMORY_PID}"    2>/dev/null && echo "[backend] memory_server    (pid=${MEMORY_PID}) stopped"
    exit 0
}
trap cleanup INT TERM

# ─── 启动 retriever_server ───────────────────────────────────────────────────
RETRIEVER_CMD=(
    python -m skill_zero.memory_manager.retriever_server
    --host "${RETRIEVER_HOST}"
    --port "${RETRIEVER_PORT}"
    --embedding-model "${EMBEDDING_MODEL}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --instruct-task "${INSTRUCT_TASK}"
    --idle-timeout "${IDLE_TIMEOUT}"
)

echo "[backend] $(date '+%Y-%m-%d %H:%M:%S')"
echo "[backend] Starting retriever_server → log: ${LOG_DIR}/retriever_server.log"
if [ -n "${RETRIEVER_CUDA_VISIBLE_DEVICES}" ]; then
    echo "[backend] retriever CUDA_VISIBLE_DEVICES=${RETRIEVER_CUDA_VISIBLE_DEVICES}"
    CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES}" \
        "${RETRIEVER_CMD[@]}" >> "${LOG_DIR}/retriever_server.log" 2>&1 &
else
    "${RETRIEVER_CMD[@]}" >> "${LOG_DIR}/retriever_server.log" 2>&1 &
fi
RETRIEVER_PID=$!
echo "[backend] retriever_server pid=${RETRIEVER_PID}"

# ─── 等待 retriever_server 就绪 ──────────────────────────────────────────────
echo "[backend] Waiting for retriever_server to be ready..."
for i in $(seq 1 60); do
    if curl -sf "http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health" > /dev/null 2>&1; then
        echo "[backend] retriever_server is ready (${i}s)"
        break
    fi
    if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
        echo "[backend] ERROR: retriever_server exited unexpectedly. Check ${LOG_DIR}/retriever_server.log"
        exit 1
    fi
    sleep 1
done

# ─── 启动 memory_server ──────────────────────────────────────────────────────
MEMORY_CMD=(
    python -m skill_zero.memory_manager.memory_server
    --host "${MEMORY_HOST}"
    --port "${MEMORY_PORT}"
    --max-capacity "${MAX_CAPACITY}"
    --warn-capacity "${WARN_CAPACITY}"
    --retrieve-mode "${RETRIEVE_MODE}"
    --retrieve-lambda "${RETRIEVE_LAMBDA}"
    --retriever-url "${RETRIEVER_URL}"
    --retriever-timeout "${RETRIEVER_TIMEOUT}"
    --update-lam "${UPDATE_LAM}"
    --update-tau "${UPDATE_TAU}"
    --update-u-min "${UPDATE_U_MIN}"
    --update-u-max "${UPDATE_U_MAX}"
)
[ -n "${PERSIST_PATH}" ]  && MEMORY_CMD+=(--persist-path "${PERSIST_PATH}")
[ -n "${SKILLS_JSONL}" ]  && MEMORY_CMD+=(--skills-jsonl "${SKILLS_JSONL}")

echo "[backend] Starting memory_server → log: ${LOG_DIR}/memory_server.log"
"${MEMORY_CMD[@]}" >> "${LOG_DIR}/memory_server.log" 2>&1 &
MEMORY_PID=$!
echo "[backend] memory_server pid=${MEMORY_PID}"

# ─── 守护：任一子进程退出则整体退出 ─────────────────────────────────────────
echo "[backend] Both servers running. Press Ctrl-C to stop."
while true; do
    if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
        echo "[backend] ERROR: retriever_server (pid=${RETRIEVER_PID}) exited. Check ${LOG_DIR}/retriever_server.log"
        cleanup
    fi
    if ! kill -0 "${MEMORY_PID}" 2>/dev/null; then
        echo "[backend] ERROR: memory_server (pid=${MEMORY_PID}) exited. Check ${LOG_DIR}/memory_server.log"
        cleanup
    fi
    sleep 3
done
