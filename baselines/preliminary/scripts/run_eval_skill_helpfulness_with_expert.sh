#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

SKILLS_RUN_DIR=""
OUT_DIR=""
TRAJ="${REPO_ROOT}/Skill_Evo/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl"
SAMPLE_SIZE=5000
METHODS=()
TEACHER_BACKENDS=()
EXPERT_API_BASE_URL="${EVAL_TEACHER_API_BASE_URL:-}"
EXPERT_API_MODEL="${EVAL_TEACHER_API_MODEL:-}"
EXPERT_API_KEY="${EVAL_TEACHER_API_KEY:-}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-0}"

usage() {
  cat <<EOF
Usage: $0 --skills-run-dir <dir> --output-dir <dir> [options]

Options:
  --trajectories <path>
  --sample-size <n>
  --methods <m1,m2,...>
  --teacher-backends <b1,b2,...>
  --expert-api-base-url <url>
  --expert-api-model <name>
  --expert-api-key <key>
  --eval-max-workers <n>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skills-run-dir) SKILLS_RUN_DIR="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --trajectories) TRAJ="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --methods) IFS=',' read -r -a METHODS <<< "$2"; shift 2 ;;
    --teacher-backends) IFS=',' read -r -a TEACHER_BACKENDS <<< "$2"; shift 2 ;;
    --expert-api-base-url) EXPERT_API_BASE_URL="$2"; shift 2 ;;
    --expert-api-model) EXPERT_API_MODEL="$2"; shift 2 ;;
    --expert-api-key) EXPERT_API_KEY="$2"; shift 2 ;;
    --eval-max-workers) EVAL_MAX_WORKERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${SKILLS_RUN_DIR}" || -z "${OUT_DIR}" ]]; then
  usage
  exit 1
fi
if [[ -z "${EXPERT_API_BASE_URL}" || -z "${EXPERT_API_MODEL}" ]]; then
  echo "expert api config is required" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/Skill_Evo:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"

CMD=(
  python -m baselines.preliminary.eval_skill_helpfulness_with_expert
  --skills-run-dir "${SKILLS_RUN_DIR}"
  --output-dir "${OUT_DIR}"
  --trajectories "${TRAJ}"
  --sample-size "${SAMPLE_SIZE}"
  --expert-api-base-url "${EXPERT_API_BASE_URL}"
  --expert-api-model "${EXPERT_API_MODEL}"
  --expert-api-key "${EXPERT_API_KEY}"
  --eval-max-workers "${EVAL_MAX_WORKERS}"
)

if [[ ${#METHODS[@]} -gt 0 ]]; then
  CMD+=(--methods "${METHODS[@]}")
fi
if [[ ${#TEACHER_BACKENDS[@]} -gt 0 ]]; then
  CMD+=(--teacher-backends "${TEACHER_BACKENDS[@]}")
fi

"${CMD[@]}"
