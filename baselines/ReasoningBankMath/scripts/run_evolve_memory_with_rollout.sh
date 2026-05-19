#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/ReasoningBankMath/scripts/run_evolve_memory_with_rollout.sh [options]

Required:
  --memory-bank <path>           Existing memory bank jsonl
  --trajectories <path>          New trajectories jsonl
  --model <path>                 Model path for rollout servers

Optional:
  --output-memory-bank <path>    Output merged memory bank jsonl
  --existing-embeddings <path>   Existing embedding cache
  --output-embeddings <path>     Output embedding cache
  --embed-backend <name>         hash or openai
  --gpu-ids <ids>                GPU ids
  --n-gpus <n>                   Number of GPUs/servers
  --rollout-host <host>
  --rollout-base-port <port>
  --rollout-start-script <path>
  --rollout-log-dir <path>
USAGE
}

MEMORY_BANK=""
TRAJ_PATH=""
MODEL="../models/Qwen3-4B-Instruct-2507"
OUT_MEMORY_BANK="${REPO_ROOT}/baselines/ReasoningBankMath/outputs/memory_bank_v2.jsonl"
EXISTING_EMBEDDINGS=""
OUT_EMBEDDINGS=""
EMBED_BACKEND="hash"
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
HASH_DIM="256"
SIMILARITY_THRESHOLD="0.98"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --memory-bank) MEMORY_BANK="$2"; shift 2 ;;
        --trajectories) TRAJ_PATH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --output-memory-bank) OUT_MEMORY_BANK="$2"; shift 2 ;;
        --existing-embeddings) EXISTING_EMBEDDINGS="$2"; shift 2 ;;
        --output-embeddings) OUT_EMBEDDINGS="$2"; shift 2 ;;
        --embed-backend) EMBED_BACKEND="$2"; shift 2 ;;
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
        --hash-dim) HASH_DIM="$2"; shift 2 ;;
        --similarity-threshold) SIMILARITY_THRESHOLD="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[rbm-evolve-memory] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${MEMORY_BANK}" || -z "${TRAJ_PATH}" ]]; then
    echo "[rbm-evolve-memory] --memory-bank and --trajectories are required" >&2
    exit 2
fi

mkdir -p "$(dirname "${OUT_MEMORY_BANK}")" "${ROLLOUT_LOG_DIR}"
if [[ -n "${OUT_EMBEDDINGS}" ]]; then mkdir -p "$(dirname "${OUT_EMBEDDINGS}")"; fi
if [[ "${MODEL}" != /* ]]; then MODEL="${REPO_ROOT}/${MODEL}"; fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export SE_GPU_IDS
export SE_N_GPUS
export SE_ROLLOUT_HOST
export SE_ROLLOUT_BASE_PORT
export SE_ROLLOUT_N_SERVERS="${SE_N_GPUS}"
export SE_ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR}"
export ROLLOUT_SERVER_MODEL="${MODEL}"

bash "${ROLLOUT_START_SCRIPT}" --model "${MODEL}" &
ROLLOUT_LAUNCHER_PID=$!

cleanup() {
    local status=$?
    if kill -0 "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null; then
        kill -TERM "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null || true
        wait "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

for ((i = 0; i < SE_ROLLOUT_N_SERVERS; i++)); do
    port=$((SE_ROLLOUT_BASE_PORT + i))
    url="http://${SE_ROLLOUT_HOST}:${port}/health"
    ok=0
    for _ in $(seq 1 120); do
        if curl -fsS "${url}" >/dev/null 2>&1; then ok=1; break; fi
        sleep 2
    done
    if [[ "${ok}" -ne 1 ]]; then
        echo "[rbm-evolve-memory] rollout health check failed: ${url}" >&2
        exit 1
    fi
done

CMD=(
  python -m baselines.ReasoningBankMath evolve-memory
  --teacher-backend rollout
  --memory-bank "${MEMORY_BANK}"
  --trajectories "${TRAJ_PATH}"
  --output-memory-bank "${OUT_MEMORY_BANK}"
  --rollout-host "${SE_ROLLOUT_HOST}"
  --rollout-base-port "${SE_ROLLOUT_BASE_PORT}"
  --rollout-n-servers "${SE_N_GPUS}"
  --embed-backend "${EMBED_BACKEND}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --max-tokens "${MAX_TOKENS}"
  --timeout "${TIMEOUT}"
  --hash-dim "${HASH_DIM}"
  --similarity-threshold "${SIMILARITY_THRESHOLD}"
)
if [[ -n "${EXISTING_EMBEDDINGS}" ]]; then
  CMD+=(--existing-embeddings "${EXISTING_EMBEDDINGS}")
fi
if [[ -n "${OUT_EMBEDDINGS}" ]]; then
  CMD+=(--output-embeddings "${OUT_EMBEDDINGS}")
fi
"${CMD[@]}"

