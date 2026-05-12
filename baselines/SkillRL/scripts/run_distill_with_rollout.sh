#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/SkillRL/scripts/run_distill_with_rollout.sh [options]

Model:
  --model <path>                 Model path for rollout server (relative path recommended)

Rollout options:
  --gpu-ids <ids>                GPU ids, comma separated (default: 4,5,6)
  --n-gpus <n>                   Number of GPUs/servers (default: 3)
  --rollout-host <host>          Rollout host (default: 127.0.0.1)
  --rollout-base-port <port>     Rollout base port (default: 8760)
  --rollout-n-servers <n>        Rollout server count (default: n-gpus)
  --rollout-start-script <path>  Startup script (default: skill_src/Zero/start_rollout_servers.sh)

Distill IO:
  --trajectories <path>          Trajectory jsonl
  --sample-k <n>                 Use first k trajectory rows for quick testing
  --output-skills <path>         Output skills json
  --raw-dump-dir <path>          Raw teacher output directory
  --rollout-log-dir <path>       Rollout server logs directory

Distill hyperparameters:
  --max-problems-per-call <n>
  --max-chars-per-call <n>
  --mistakes-cap <n>
  --temperature <f>
  --top-p <f>
  --top-k <n>
  --max-tokens <n>
  --timeout <sec>

Example:
  bash baselines/SkillRL/scripts/run_distill_with_rollout.sh \\
    --model ../models/Qwen3-4B-Instruct-2507 \\
    --gpu-ids 4,5,6 --n-gpus 3 \\
    --trajectories baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl \\
    --output-skills baselines/SkillRL/outputs/skills_from_rollout_teacher.json
USAGE
}

MODEL="../models/Qwen3-4B-Instruct-2507"
TRAJ_PATH="${REPO_ROOT}/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl"
SAMPLE_K="0"
OUT_PATH="${REPO_ROOT}/baselines/SkillRL/outputs/skills_from_rollout_teacher.json"
RAW_DUMP_DIR="${REPO_ROOT}/baselines/SkillRL/outputs/distill_raw"
ROLLOUT_LOG_DIR="${REPO_ROOT}/baselines/SkillRL/outputs/logs/rollout_servers"

MAX_PROBLEMS_PER_CALL="8"
MAX_CHARS_PER_CALL="12000"
MISTAKES_CAP="50"
TEMPERATURE="0.7"
TOP_P="0.95"
TOP_K="50"
MAX_TOKENS="8192"
TIMEOUT="600"

ROLLOUT_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_rollout_servers.sh"
SE_ROLLOUT_HOST="127.0.0.1"
SE_ROLLOUT_BASE_PORT="8760"
SE_GPU_IDS="4,5,6,7"
SE_N_GPUS="4"
SE_ROLLOUT_N_SERVERS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --gpu-ids) SE_GPU_IDS="$2"; shift 2 ;;
        --n-gpus) SE_N_GPUS="$2"; shift 2 ;;
        --rollout-host) SE_ROLLOUT_HOST="$2"; shift 2 ;;
        --rollout-base-port) SE_ROLLOUT_BASE_PORT="$2"; shift 2 ;;
        --rollout-n-servers) SE_ROLLOUT_N_SERVERS="$2"; shift 2 ;;
        --rollout-start-script) ROLLOUT_START_SCRIPT="$2"; shift 2 ;;
        --trajectories) TRAJ_PATH="$2"; shift 2 ;;
        --sample-k) SAMPLE_K="$2"; shift 2 ;;
        --output-skills) OUT_PATH="$2"; shift 2 ;;
        --raw-dump-dir) RAW_DUMP_DIR="$2"; shift 2 ;;
        --rollout-log-dir) ROLLOUT_LOG_DIR="$2"; shift 2 ;;
        --max-problems-per-call) MAX_PROBLEMS_PER_CALL="$2"; shift 2 ;;
        --max-chars-per-call) MAX_CHARS_PER_CALL="$2"; shift 2 ;;
        --mistakes-cap) MISTAKES_CAP="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[distill-rollout] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

SE_ROLLOUT_N_SERVERS="${SE_ROLLOUT_N_SERVERS:-${SE_N_GPUS}}"

if [[ ! -f "${TRAJ_PATH}" ]]; then
    echo "[distill-rollout] trajectories not found: ${TRAJ_PATH}" >&2
    exit 1
fi
if [[ ! -x "${ROLLOUT_START_SCRIPT}" ]]; then
    echo "[distill-rollout] rollout start script missing or not executable: ${ROLLOUT_START_SCRIPT}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT_PATH}")" "${RAW_DUMP_DIR}" "${ROLLOUT_LOG_DIR}"

if [[ "${MODEL}" != /* ]]; then
    MODEL="${REPO_ROOT}/${MODEL}"
fi
if [[ ! -e "${MODEL}" ]]; then
    echo "[distill-rollout] model path not found: ${MODEL}" >&2
    exit 1
fi

export SE_GPU_IDS
export SE_N_GPUS
export SE_ROLLOUT_HOST
export SE_ROLLOUT_BASE_PORT
export SE_ROLLOUT_N_SERVERS
export SE_ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR}"
export ROLLOUT_SERVER_MODEL="${MODEL}"

TMP_SAMPLE_TRAJ=""
if [[ "${SAMPLE_K}" != "0" ]]; then
    if ! [[ "${SAMPLE_K}" =~ ^[0-9]+$ ]] || [[ "${SAMPLE_K}" -le 0 ]]; then
        echo "[distill-rollout] --sample-k must be a positive integer" >&2
        exit 2
    fi
    TMP_SAMPLE_TRAJ="${RAW_DUMP_DIR}/sample_k_${SAMPLE_K}.jsonl"
    head -n "${SAMPLE_K}" "${TRAJ_PATH}" > "${TMP_SAMPLE_TRAJ}"
    TRAJ_PATH="${TMP_SAMPLE_TRAJ}"
    echo "[distill-rollout] sample mode enabled: first ${SAMPLE_K} rows -> ${TRAJ_PATH}"
fi


echo "[distill-rollout] starting rollout servers..."
bash "${ROLLOUT_START_SCRIPT}" --model "${MODEL}" &
ROLLOUT_LAUNCHER_PID=$!

cleanup() {
    local status=$?
    if kill -0 "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null; then
        echo "[distill-rollout] stopping rollout servers..."
        kill -TERM "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null || true
        wait "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${TMP_SAMPLE_TRAJ}" && -f "${TMP_SAMPLE_TRAJ}" ]]; then
        rm -f "${TMP_SAMPLE_TRAJ}" || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "[distill-rollout] waiting for rollout health checks..."
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
        echo "[distill-rollout] health check failed: ${url}" >&2
        exit 1
    fi
    echo "[distill-rollout] healthy: ${url}"
done

echo "[distill-rollout] running teacher distill with rollout backend..."
python -m baselines.SkillRL distill \
    --teacher-backend rollout \
    --trajectories "${TRAJ_PATH}" \
    --output-skills "${OUT_PATH}" \
    --raw-dump-dir "${RAW_DUMP_DIR}" \
    --rollout-host "${SE_ROLLOUT_HOST}" \
    --rollout-base-port "${SE_ROLLOUT_BASE_PORT}" \
    --rollout-n-servers "${SE_ROLLOUT_N_SERVERS}" \
    --max-problems-per-call "${MAX_PROBLEMS_PER_CALL}" \
    --max-chars-per-call "${MAX_CHARS_PER_CALL}" \
    --mistakes-cap "${MISTAKES_CAP}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-tokens "${MAX_TOKENS}" \
    --timeout "${TIMEOUT}"

echo "[distill-rollout] done. skills saved to: ${OUT_PATH}"
