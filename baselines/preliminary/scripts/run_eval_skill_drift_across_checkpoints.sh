#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

SKILLS_RUN_DIR=""
CHECKPOINT_ROOT=""
OUT_DIR=""
TRAJ="${REPO_ROOT}/Skill_Evo/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl"
SAMPLE_SIZE=5000
CHECKPOINT_LIMIT=0
METHODS=()
TEACHER_BACKENDS=()
STUDENT_ROLLOUT_N=4
SE_GPU_IDS="${SE_GPU_IDS:-4,5,6,7}"
SE_N_GPUS="${SE_N_GPUS:-4}"
SE_ROLLOUT_HOST="${SE_ROLLOUT_HOST:-127.0.0.1}"
SE_ROLLOUT_BASE_PORT="${SE_ROLLOUT_BASE_PORT:-8760}"

usage() {
  cat <<EOF
Usage: $0 --skills-run-dir <dir> --checkpoint-root <dir> --output-dir <dir> [options]

Options:
  --trajectories <path>
  --sample-size <n>
  --checkpoint-limit <n>
  --methods <m1,m2,...>
  --teacher-backends <b1,b2,...>
  --student-rollout-n <n>
  --gpu-ids <ids>
  --n-gpus <n>
  --rollout-host <host>
  --rollout-base-port <port>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skills-run-dir) SKILLS_RUN_DIR="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --trajectories) TRAJ="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --checkpoint-limit) CHECKPOINT_LIMIT="$2"; shift 2 ;;
    --methods) IFS=',' read -r -a METHODS <<< "$2"; shift 2 ;;
    --teacher-backends) IFS=',' read -r -a TEACHER_BACKENDS <<< "$2"; shift 2 ;;
    --student-rollout-n) STUDENT_ROLLOUT_N="$2"; shift 2 ;;
    --gpu-ids) SE_GPU_IDS="$2"; shift 2 ;;
    --n-gpus) SE_N_GPUS="$2"; shift 2 ;;
    --rollout-host) SE_ROLLOUT_HOST="$2"; shift 2 ;;
    --rollout-base-port) SE_ROLLOUT_BASE_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${SKILLS_RUN_DIR}" || -z "${CHECKPOINT_ROOT}" || -z "${OUT_DIR}" ]]; then
  usage
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/Skill_Evo:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"

CMD=(
  python -m baselines.preliminary.eval_skill_drift_across_checkpoints
  --skills-run-dir "${SKILLS_RUN_DIR}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --output-dir "${OUT_DIR}"
  --trajectories "${TRAJ}"
  --sample-size "${SAMPLE_SIZE}"
  --checkpoint-limit "${CHECKPOINT_LIMIT}"
  --student-rollout-n "${STUDENT_ROLLOUT_N}"
  --gpu-ids "${SE_GPU_IDS}"
  --n-gpus "${SE_N_GPUS}"
  --rollout-host "${SE_ROLLOUT_HOST}"
  --rollout-base-port "${SE_ROLLOUT_BASE_PORT}"
)

if [[ ${#METHODS[@]} -gt 0 ]]; then
  CMD+=(--methods "${METHODS[@]}")
fi
if [[ ${#TEACHER_BACKENDS[@]} -gt 0 ]]; then
  CMD+=(--teacher-backends "${TEACHER_BACKENDS[@]}")
fi

"${CMD[@]}"
