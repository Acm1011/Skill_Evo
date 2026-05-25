#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/MementoMath/scripts/run_build_memory_with_rollout.sh [options]

Required:
  --trajectories <path>          SkillRL-style trajectories jsonl
  --model <path>                 Model path for rollout servers

Optional:
  --output <path>                Output memory bank jsonl
  --case-pool-output <path>      Output Memento-style memory.jsonl
  --dummy-memory-output <path>   Output dummy_memo.jsonl
  --gpu-ids <ids>                GPU ids, comma separated
  --n-gpus <n>                   Number of GPUs/servers
  --rollout-host <host>          Rollout host
  --rollout-base-port <port>     Rollout base port
  --rollout-start-script <path>  Startup script
  --rollout-log-dir <path>       Rollout log dir
  --temperature <f>
  --top-p <f>
  --top-k <n>
  --max-tokens <n>
  --timeout <sec>
USAGE
}

TRAJ_PATH=""
MODEL="../models/Qwen3-4B-Instruct-2507"
OUT_PATH="${REPO_ROOT}/baselines/MementoMath/outputs/memory_bank.jsonl"
CASE_POOL_OUTPUT="${REPO_ROOT}/baselines/MementoMath/outputs/memory.jsonl"
DUMMY_MEMORY_OUTPUT="${REPO_ROOT}/baselines/MementoMath/outputs/dummy_memo.jsonl"
ROLLOUT_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_rollout_servers.sh"
ROLLOUT_LOG_DIR="${REPO_ROOT}/baselines/MementoMath/outputs/logs/rollout_servers"
SE_ROLLOUT_HOST="127.0.0.1"
SE_ROLLOUT_BASE_PORT="8760"
SE_GPU_IDS="4,5,6,7"
SE_N_GPUS="4"
TEMPERATURE="0.7"
TOP_P="0.95"
TOP_K="50"
MAX_TOKENS="1024"
TIMEOUT="600"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trajectories) TRAJ_PATH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --output) OUT_PATH="$2"; shift 2 ;;
        --case-pool-output) CASE_POOL_OUTPUT="$2"; shift 2 ;;
        --dummy-memory-output) DUMMY_MEMORY_OUTPUT="$2"; shift 2 ;;
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
        -h|--help) usage; exit 0 ;;
        *) echo "[mmm-build-memory] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${TRAJ_PATH}" ]]; then
    echo "[mmm-build-memory] --trajectories is required" >&2
    exit 2
fi

mkdir -p "$(dirname "${OUT_PATH}")" "$(dirname "${CASE_POOL_OUTPUT}")" "$(dirname "${DUMMY_MEMORY_OUTPUT}")" "${ROLLOUT_LOG_DIR}"
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
        echo "[mmm-build-memory] rollout health check failed: ${url}" >&2
        exit 1
    fi
done

python -m baselines.MementoMath build-memory \
    --teacher-backend rollout \
    --trajectories "${TRAJ_PATH}" \
    --output "${OUT_PATH}" \
    --case-pool-output "${CASE_POOL_OUTPUT}" \
    --dummy-memory-output "${DUMMY_MEMORY_OUTPUT}" \
    --rollout-host "${SE_ROLLOUT_HOST}" \
    --rollout-base-port "${SE_ROLLOUT_BASE_PORT}" \
    --rollout-n-servers "${SE_N_GPUS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-tokens "${MAX_TOKENS}" \
    --timeout "${TIMEOUT}"
