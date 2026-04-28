#!/usr/bin/env bash
# Build train/val parquet for EvolveR math (DeepMath-103K jsonl).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVOR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

: "${DEEPMATH_JSONL:?set DEEPMATH_JSONL to DeepMath-103K.jsonl path}"
: "${OUT_TRAIN:?set OUT_TRAIN to output train.parquet path}"
START="${START:-0}"
END="${END:?set END (exclusive line index)}"
VAL_RATIO="${VAL_RATIO:-0}"
OUT_VAL="${OUT_VAL:-}"

Args=(--deepmath-jsonl "$DEEPMATH_JSONL" --start "$START" --end "$END" --output-train "$OUT_TRAIN")
if [ -n "$OUT_VAL" ] && [ "$VAL_RATIO" != "0" ] && [ "$VAL_RATIO" != "0.0" ]; then
  Args+=(--val-ratio "$VAL_RATIO" --output-val "$OUT_VAL")
fi

python3 "$EVOR/build_math_parquet.py" "${Args[@]}"
