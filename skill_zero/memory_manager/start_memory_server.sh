#!/usr/bin/env bash
set -euo pipefail

# ─── 服务监听 ────────────────────────────────────────────────────────────────
HOST="0.0.0.0"
PORT=8765
MAX_CAPACITY=3
WARN_CAPACITY=2            # 警告区最大容量（主库满时的降级存储区）

# ─── 检索超参 ────────────────────────────────────────────────────────────────
RETRIEVE_MODE="embedding"    # embedding | hybrid
RETRIEVE_LAMBDA=0.5          # hybrid 模式下有效：sim' = (1-λ)*sim + λ*utility

# ─── Retriever 服务地址（独立 embedding 服务，由 start_retriever_server.sh 启动）
RETRIEVER_URL="http://127.0.0.1:8766"
RETRIEVER_TIMEOUT=30         # 调用 retriever_server 的超时秒数

# ─── Utility 更新超参（/update 接口使用）────────────────────────────────────
UPDATE_LAM=0.9               # EMA 衰减系数 λ
UPDATE_TAU=0.2               # reward 阈值 τ，低于此值不触发更新
UPDATE_U_MIN=0.0             # utility 下界
UPDATE_U_MAX=1.0             # utility 上界

# ─── 持久化路径 ──────────────────────────────────────────────────────────────
# /add 接口成功后追加写入该 jsonl；留空则使用默认路径 runs/skills_memory.jsonl
PERSIST_PATH="memory_manager/memory/skills_memory_test.jsonl"

# ─── 启动时批量导入（可选）──────────────────────────────────────────────────
# 填写已有 skills.jsonl 路径可在启动时预加载；留空则跳过
SKILLS_JSONL=""

# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."   # 切到项目根目录（skill_zero 的上一级）

CMD=(
    python -m skill_zero.memory_manager.memory_server
    --host "${HOST}"
    --port "${PORT}"
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

[ -n "${PERSIST_PATH}" ]  && CMD+=(--persist-path "${PERSIST_PATH}")
[ -n "${SKILLS_JSONL}" ]  && CMD+=(--skills-jsonl "${SKILLS_JSONL}")

echo "[start_memory_server] $(date '+%Y-%m-%d %H:%M:%S')"
echo "[start_memory_server] cmd: ${CMD[*]}"
exec "${CMD[@]}"
