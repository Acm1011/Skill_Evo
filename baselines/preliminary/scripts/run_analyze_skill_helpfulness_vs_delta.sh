#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DETAILS=""
EXPERT_SCORES=""
OUT_DIR=""

usage() {
  cat <<EOF
Usage: $0 --details <path> --expert-scores <path> --output-dir <dir>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --details) DETAILS="$2"; shift 2 ;;
    --expert-scores) EXPERT_SCORES="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${DETAILS}" || -z "${EXPERT_SCORES}" || -z "${OUT_DIR}" ]]; then
  usage
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/Skill_Evo:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"

python -m baselines.preliminary.analyze_skill_helpfulness_vs_delta \
  --details "${DETAILS}" \
  --expert-scores "${EXPERT_SCORES}" \
  --output-dir "${OUT_DIR}"
