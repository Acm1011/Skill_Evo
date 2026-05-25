#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/MementoMath/scripts/run_gen_traj.sh [options]

Required:
  --data-path <path>             DeepMath jsonl
  --end <n>                      Exclusive end index
  --model <path>                 Model path for rollout servers

Optional:
  --start <n>                    Start index (default: 0)
  --output <path>                Output trajectories jsonl
  --gpu-ids <ids>                GPU ids, comma separated
  --n-gpus <n>                   Number of GPUs/servers
  --rollout-host <host>          Rollout host
  --rollout-base-port <port>     Rollout base port
  --rollout-start-script <path>  Startup script
  --rollout-log-dir <path>       Rollout log dir
  --served-model-name <name>     Override served model name
  --max-tokens <n>
  --temperature <f>
  --top-p <f>
  --timeout <sec>
USAGE
}

DATA_PATH=""
END_IDX=""
START_IDX="0"
MODEL="../models/Qwen3-4B-Instruct-2507"
OUTPUT="${REPO_ROOT}/baselines/MementoMath/outputs/trajectories.jsonl"
ROLLOUT_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_rollout_servers.sh"
ROLLOUT_LOG_DIR="${REPO_ROOT}/baselines/MementoMath/outputs/logs/rollout_servers"
SE_ROLLOUT_HOST="127.0.0.1"
SE_ROLLOUT_BASE_PORT="8760"
SE_GPU_IDS="4,5,6,7"
SE_N_GPUS="4"
SERVED_MODEL_NAME=""
MAX_TOKENS="4096"
TEMPERATURE="0.7"
TOP_P="0.95"
TIMEOUT="600"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-path) DATA_PATH="$2"; shift 2 ;;
        --start) START_IDX="$2"; shift 2 ;;
        --end) END_IDX="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --gpu-ids) SE_GPU_IDS="$2"; shift 2 ;;
        --n-gpus) SE_N_GPUS="$2"; shift 2 ;;
        --rollout-host) SE_ROLLOUT_HOST="$2"; shift 2 ;;
        --rollout-base-port) SE_ROLLOUT_BASE_PORT="$2"; shift 2 ;;
        --rollout-start-script) ROLLOUT_START_SCRIPT="$2"; shift 2 ;;
        --rollout-log-dir) ROLLOUT_LOG_DIR="$2"; shift 2 ;;
        --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[mmm-gen-traj] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${DATA_PATH}" || -z "${END_IDX}" ]]; then
    echo "[mmm-gen-traj] --data-path and --end are required" >&2
    exit 2
fi

mkdir -p "$(dirname "${OUTPUT}")" "${ROLLOUT_LOG_DIR}"
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
        echo "[mmm-gen-traj] rollout health check failed: ${url}" >&2
        exit 1
    fi
done

CMD=(
  python -m baselines.MementoMath gen-traj
  --data-path "${DATA_PATH}"
  --start "${START_IDX}"
  --end "${END_IDX}"
  --output "${OUTPUT}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --timeout "${TIMEOUT}"
)
if [[ -n "${SERVED_MODEL_NAME}" ]]; then
  CMD+=(--served-model-name "${SERVED_MODEL_NAME}")
fi
"${CMD[@]}"
