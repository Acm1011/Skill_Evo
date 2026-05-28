#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

TRAJ="${REPO_ROOT}/Skill_Evo/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl"
METHOD="all"
SAMPLE_SIZE=5000
MODEL=""
OUT_DIR=""
SE_GPU_IDS="${SE_GPU_IDS:-4,5,6,7}"
SE_N_GPUS="${SE_N_GPUS:-4}"
SE_ROLLOUT_HOST="${SE_ROLLOUT_HOST:-127.0.0.1}"
SE_ROLLOUT_BASE_PORT="${SE_ROLLOUT_BASE_PORT:-8760}"
ROLLOUT_LOG_DIR="${REPO_ROOT}/Skill_Evo/baselines/preliminary/outputs/source_linked_eval/logs/rollout_servers"
TEACHER_API_BASE_URL="${EVAL_TEACHER_API_BASE_URL:-}"
TEACHER_API_MODEL="${EVAL_TEACHER_API_MODEL:-}"
TEACHER_API_KEY="${EVAL_TEACHER_API_KEY:-}"
STUDENT_ROLLOUT_N=4
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-0}"
RESUME=0

usage() {
  cat <<EOF
Usage: $0 --model <model_path> --output-dir <dir> [options]

Options:
  --model <path>
  --output-dir <dir>
  --trajectories <path>
  --method <skillrl|reasoningbank|expelmath|all>
  --sample-size <n>
  --gpu-ids <ids>
  --n-gpus <n>
  --rollout-host <host>
  --rollout-base-port <port>
  --rollout-log-dir <path>
  --teacher-api-base-url <url>
  --teacher-api-model <name>
  --teacher-api-key <key>
  --student-rollout-n <n>
  --eval-max-workers <n>
  --resume
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --trajectories) TRAJ="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --gpu-ids) SE_GPU_IDS="$2"; shift 2 ;;
    --n-gpus) SE_N_GPUS="$2"; shift 2 ;;
    --rollout-host) SE_ROLLOUT_HOST="$2"; shift 2 ;;
    --rollout-base-port) SE_ROLLOUT_BASE_PORT="$2"; shift 2 ;;
    --rollout-log-dir) ROLLOUT_LOG_DIR="$2"; shift 2 ;;
    --teacher-api-base-url) TEACHER_API_BASE_URL="$2"; shift 2 ;;
    --teacher-api-model) TEACHER_API_MODEL="$2"; shift 2 ;;
    --teacher-api-key) TEACHER_API_KEY="$2"; shift 2 ;;
    --student-rollout-n) STUDENT_ROLLOUT_N="$2"; shift 2 ;;
    --eval-max-workers) EVAL_MAX_WORKERS="$2"; shift 2 ;;
    --resume) RESUME=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${MODEL}" || -z "${OUT_DIR}" ]]; then
  usage
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/Skill_Evo:${PYTHONPATH:-}"
export SE_WORKING_DIR="${REPO_ROOT}/Skill_Evo"
export SE_PROJECT_NAME="Skill_Evo"
export SE_ROLLOUT_MODEL="${MODEL}"
export SE_GPU_IDS
export SE_N_GPUS
export SE_ROLLOUT_HOST
export SE_ROLLOUT_BASE_PORT
export SE_ROLLOUT_N_SERVERS="${SE_N_GPUS}"
export SE_ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR}"

mkdir -p "${OUT_DIR}" "${ROLLOUT_LOG_DIR}"

if [[ "${RESUME}" -eq 1 ]]; then
  set +e
  RESUME_STATUS="$(
    python -m baselines.preliminary.eval_source_linked_skills \
      --trajectories "${TRAJ}" \
      --output-dir "${OUT_DIR}" \
      --method "${METHOD}" \
      --sample-size "${SAMPLE_SIZE}" \
      --resume-check-only
  )"
  RESUME_RC=$?
  set -e
  if [[ "${RESUME_RC}" -eq 0 ]]; then
    echo "resume check: outputs already complete, skipping rollout server startup"
    echo "${RESUME_STATUS}"
    exit 0
  fi
  if [[ "${RESUME_RC}" -eq 11 ]]; then
    echo "resume check: outputs incomplete, but no rollout server is needed"
    echo "${RESUME_STATUS}"
    EVAL_CMD=(
      python -m baselines.preliminary.eval_source_linked_skills
      --trajectories "${TRAJ}"
      --output-dir "${OUT_DIR}"
      --method "${METHOD}"
      --sample-size "${SAMPLE_SIZE}"
      --server-urls "http://${SE_ROLLOUT_HOST}:${SE_ROLLOUT_BASE_PORT}"
      --teacher-api-base-url "${TEACHER_API_BASE_URL}"
      --teacher-api-model "${TEACHER_API_MODEL}"
      --teacher-api-key "${TEACHER_API_KEY}"
      --student-rollout-n "${STUDENT_ROLLOUT_N}"
      --eval-max-workers "${EVAL_MAX_WORKERS}"
      --resume
    )
    "${EVAL_CMD[@]}"
    exit 0
  fi
  if [[ "${RESUME_RC}" -ne 10 ]]; then
    echo "resume check failed" >&2
    echo "${RESUME_STATUS}" >&2
    exit "${RESUME_RC}"
  fi
  echo "resume check: outputs incomplete, continuing with rollout server startup"
  echo "${RESUME_STATUS}"
fi

if [[ -z "${TEACHER_API_BASE_URL}" || -z "${TEACHER_API_MODEL}" ]]; then
  echo "teacher api config is required" >&2
  exit 1
fi

bash "${REPO_ROOT}/Skill_Evo/skill_src/Zero/start_rollout_servers.sh" --model "${MODEL}" &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
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
    echo "rollout health check failed: ${url}" >&2
    exit 1
  fi
done

SERVER_URLS=()
for ((i = 0; i < SE_ROLLOUT_N_SERVERS; i++)); do
  port=$((SE_ROLLOUT_BASE_PORT + i))
  SERVER_URLS+=("http://${SE_ROLLOUT_HOST}:${port}")
done

EVAL_CMD=(
  python -m baselines.preliminary.eval_source_linked_skills
  --trajectories "${TRAJ}"
  --output-dir "${OUT_DIR}"
  --method "${METHOD}"
  --sample-size "${SAMPLE_SIZE}"
  --server-urls "${SERVER_URLS[@]}"
  --teacher-api-base-url "${TEACHER_API_BASE_URL}"
  --teacher-api-model "${TEACHER_API_MODEL}"
  --teacher-api-key "${TEACHER_API_KEY}"
  --student-rollout-n "${STUDENT_ROLLOUT_N}"
  --eval-max-workers "${EVAL_MAX_WORKERS}"
)

if [[ "${RESUME}" -eq 1 ]]; then
  EVAL_CMD+=(--resume)
fi

"${EVAL_CMD[@]}"
