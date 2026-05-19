#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/ReasoningBankMath/scripts/run_build_memory_pipeline.sh [options]

Default behavior:
  1. Start rollout servers with skill_src/Zero/start_rollout_servers.sh
  2. Build memory bank from trajectories jsonl
  3. Build memory embeddings

Required:
  --model <path>                 Model path for rollout teacher servers

Optional:
  --trajectories <path>          Input trajectories jsonl
  --memory-bank <path>           Output memory bank jsonl
  --embeddings <path>            Output memory embeddings jsonl
  --gpu-ids <ids>                GPU ids, comma separated
  --n-gpus <n>                   Number of GPUs/servers
  --rollout-host <host>          Rollout host
  --rollout-base-port <port>     Rollout base port
  --rollout-start-script <path>  Startup script
  --rollout-log-dir <path>       Rollout log dir
  --temperature <f>              Teacher sampling temperature
  --top-p <f>                    Teacher top-p
  --top-k <n>                    Teacher top-k
  --max-tokens <n>               Teacher max tokens
  --timeout <sec>                Teacher request timeout
  --embed-backend <name>         hash or openai
  --embed-base-url <url>         Used when embed-backend=openai
  --embed-api-key <key>          Used when embed-backend=openai
  --embed-model <name>           Used when embed-backend=openai
  --hash-dim <n>                 Used when embed-backend=hash

Example:
  bash baselines/ReasoningBankMath/scripts/run_build_memory_pipeline.sh \
    --model ../models/Qwen3-4B-Instruct-2507
USAGE
}

TRAJ_PATH="${REPO_ROOT}/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl"
MODEL=""
MEMORY_BANK="${REPO_ROOT}/baselines/ReasoningBankMath/outputs/memory_bank_v1_v2.jsonl"
EMBEDDINGS="${REPO_ROOT}/baselines/ReasoningBankMath/outputs/memory_embeddings_v1_v2.jsonl"
ROLLOUT_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_rollout_servers.sh"
ROLLOUT_LOG_DIR="${REPO_ROOT}/baselines/ReasoningBankMath/outputs/logs/rollout_servers"
SE_ROLLOUT_HOST="127.0.0.1"
SE_ROLLOUT_BASE_PORT="8760"
SE_GPU_IDS="4,5,6,7"
SE_N_GPUS="4"
TEMPERATURE="0.7"
TOP_P="0.95"
TOP_K="50"
MAX_TOKENS="2048"
TIMEOUT="600"
EMBED_BACKEND="hash"
EMBED_BASE_URL=""
EMBED_API_KEY=""
EMBED_MODEL=""
HASH_DIM="256"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --trajectories) TRAJ_PATH="$2"; shift 2 ;;
        --memory-bank) MEMORY_BANK="$2"; shift 2 ;;
        --embeddings) EMBEDDINGS="$2"; shift 2 ;;
        --gpu-ids) SE_GPU_IDS="$2"; shift 2 ;;
        --n-gpus) SE_N_GPUS="$2"; shift 2 ;;
        --rollout-host) SE_ROLLOUT_HOST="$2"; shift 2 ;;
        --rollout-base-port) SE_ROLLOUT_BASE_PORT="$2"; shift 2 ;;
        --rollout-start-script) ROLLOUT_START_SCRIPT="$2"; shift 2 ;;
        --rollout-log-dir) ROLLOUT_LOG_DIR="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --embed-backend) EMBED_BACKEND="$2"; shift 2 ;;
        --embed-base-url) EMBED_BASE_URL="$2"; shift 2 ;;
        --embed-api-key) EMBED_API_KEY="$2"; shift 2 ;;
        --embed-model) EMBED_MODEL="$2"; shift 2 ;;
        --hash-dim) HASH_DIM="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[rbm-pipeline] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "[rbm-pipeline] --model is required" >&2
    exit 2
fi
if [[ ! -f "${TRAJ_PATH}" ]]; then
    echo "[rbm-pipeline] trajectories not found: ${TRAJ_PATH}" >&2
    exit 1
fi
if [[ ! -x "${ROLLOUT_START_SCRIPT}" ]]; then
    echo "[rbm-pipeline] rollout start script missing or not executable: ${ROLLOUT_START_SCRIPT}" >&2
    exit 1
fi

mkdir -p "$(dirname "${MEMORY_BANK}")" "$(dirname "${EMBEDDINGS}")" "${ROLLOUT_LOG_DIR}"
if [[ "${MODEL}" != /* ]]; then
    MODEL="${REPO_ROOT}/${MODEL}"
fi
if [[ ! -e "${MODEL}" ]]; then
    echo "[rbm-pipeline] model path not found: ${MODEL}" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export SE_GPU_IDS
export SE_N_GPUS
export SE_ROLLOUT_HOST
export SE_ROLLOUT_BASE_PORT
export SE_ROLLOUT_N_SERVERS="${SE_N_GPUS}"
export SE_ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR}"
export ROLLOUT_SERVER_MODEL="${MODEL}"

echo "[rbm-pipeline] trajectories: ${TRAJ_PATH}"
echo "[rbm-pipeline] memory bank: ${MEMORY_BANK}"
echo "[rbm-pipeline] embeddings: ${EMBEDDINGS}"
echo "[rbm-pipeline] model: ${MODEL}"
echo "[rbm-pipeline] embed backend: ${EMBED_BACKEND}"

echo "[rbm-pipeline] starting rollout servers..."
bash "${ROLLOUT_START_SCRIPT}" --model "${MODEL}" &
ROLLOUT_LAUNCHER_PID=$!

cleanup() {
    local status=$?
    if kill -0 "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null; then
        echo "[rbm-pipeline] stopping rollout servers..."
        kill -TERM "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null || true
        wait "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "[rbm-pipeline] waiting for rollout health checks..."
for ((i = 0; i < SE_ROLLOUT_N_SERVERS; i++)); do
    port=$((SE_ROLLOUT_BASE_PORT + i))
    url="http://${SE_ROLLOUT_HOST}:${port}/health"
    ok=0
    for _ in $(seq 1 120); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            ok=1
            break
        fi
        sleep 2
    done
    if [[ "${ok}" -ne 1 ]]; then
        echo "[rbm-pipeline] rollout health check failed: ${url}" >&2
        exit 1
    fi
    echo "[rbm-pipeline] healthy: ${url}"
done

echo "[rbm-pipeline] building memory bank..."
python -m baselines.ReasoningBankMath build-memory \
    --teacher-backend rollout \
    --trajectories "${TRAJ_PATH}" \
    --output "${MEMORY_BANK}" \
    --rollout-host "${SE_ROLLOUT_HOST}" \
    --rollout-base-port "${SE_ROLLOUT_BASE_PORT}" \
    --rollout-n-servers "${SE_N_GPUS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-tokens "${MAX_TOKENS}" \
    --timeout "${TIMEOUT}"

echo "[rbm-pipeline] building embeddings..."
EMBED_CMD=(
  python -m baselines.ReasoningBankMath build-embeddings
  --memory-bank "${MEMORY_BANK}"
  --output "${EMBEDDINGS}"
  --backend "${EMBED_BACKEND}"
  --timeout "${TIMEOUT}"
  --hash-dim "${HASH_DIM}"
)
if [[ -n "${EMBED_BASE_URL}" ]]; then
  EMBED_CMD+=(--embed-base-url "${EMBED_BASE_URL}")
fi
if [[ -n "${EMBED_API_KEY}" ]]; then
  EMBED_CMD+=(--embed-api-key "${EMBED_API_KEY}")
fi
if [[ -n "${EMBED_MODEL}" ]]; then
  EMBED_CMD+=(--embed-model "${EMBED_MODEL}")
fi
"${EMBED_CMD[@]}"

echo "[rbm-pipeline] done"
