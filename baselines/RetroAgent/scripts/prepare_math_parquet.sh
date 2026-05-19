#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

: "${DEEPMATH_JSONL:?set DEEPMATH_JSONL}"
: "${END:?set END}"

OUT_TRAIN="${OUT_TRAIN:-${ROOT_DIR}/baselines/RetroAgent/outputs/deepmath_rl_train.parquet}"
OUT_VAL="${OUT_VAL:-}"
VAL_RATIO="${VAL_RATIO:-0.0}"
START="${START:-0}"

mkdir -p "$(dirname "${OUT_TRAIN}")"

cmd=(
  python "${ROOT_DIR}/baselines/RetroAgent/build_math_parquet.py"
  --deepmath-jsonl "${DEEPMATH_JSONL}"
  --start "${START}"
  --end "${END}"
  --val-ratio "${VAL_RATIO}"
  --output-train "${OUT_TRAIN}"
)

if [[ -n "${OUT_VAL}" ]]; then
  mkdir -p "$(dirname "${OUT_VAL}")"
  cmd+=(--output-val "${OUT_VAL}")
fi

"${cmd[@]}"

