#!/usr/bin/env bash
set -euo pipefail

# ─── 服务监听 ────────────────────────────────────────────────────────────────
HOST="127.0.0.1"
PORT=8766

# ─── Embedding 模型（vLLM 后端，自动管理 GPU） ───────────────────────────────
EMBEDDING_MODEL="/home/xzs/data/model/Qwen3-Embedding-0.6B"

# ─── GPU 配置 ────────────────────────────────────────────────────────────────
# 指定使用哪张/哪些 GPU（PCI_BUS_ID 顺序，与 nvidia-smi 索引一致）
# 留空则不设置 CUDA_VISIBLE_DEVICES，使用系统默认（所有可见 GPU）
CUDA_VISIBLE_DEVICES="1"
# tensor 并行 GPU 数，需与 CUDA_VISIBLE_DEVICES 中的卡数一致
TENSOR_PARALLEL_SIZE=1
# vLLM 显存占用比例（0~1）；embedding 模型较小，0.3 已足够
GPU_MEMORY_UTILIZATION=0.3

# ─── Query 侧 Instruct 任务描述 ───────────────────────────────────────────────
INSTRUCT_TASK="Given a question, retrieve relevant skills that help answer it"

# ─── 空闲超时 ────────────────────────────────────────────────────────────────
# 超过该秒数无检索调用则自动卸载模型并退出，释放 GPU 显存
IDLE_TIMEOUT=300

# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."   # 切到项目根目录

CMD=(
    python -m skill_zero.memory_manager.retriever_server
    --host "${HOST}"
    --port "${PORT}"
    --embedding-model "${EMBEDDING_MODEL}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --instruct-task "${INSTRUCT_TASK}"
    --idle-timeout "${IDLE_TIMEOUT}"
)

echo "[start_retriever_server] $(date '+%Y-%m-%d %H:%M:%S')"
echo "[start_retriever_server] cmd: ${CMD[*]}"

# 混合 GPU 型号时必须设置 PCI_BUS_ID，保证 CUDA 设备编号与 nvidia-smi 一致
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "[start_retriever_server] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    exec env CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
else
    exec "${CMD[@]}"
fi
