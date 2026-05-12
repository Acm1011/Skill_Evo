#!/usr/bin/env bash
# Build train/val parquet for EvolveR math (DeepMath-103K jsonl).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common_env.sh"

mkdir -p "$(dirname "$OUT_TRAIN")"
[ -n "${OUT_VAL:-}" ] && mkdir -p "$(dirname "$OUT_VAL")"

if [ ! -f "$DEEPMATH_JSONL" ]; then
  echo "DEEPMATH_JSONL not found: $DEEPMATH_JSONL" >&2
  echo "Override it before running, e.g.:" >&2
  echo "  export DEEPMATH_JSONL=/path/to/DeepMath-103K.jsonl" >&2
  exit 1
fi

Args=(--deepmath-jsonl "$DEEPMATH_JSONL" --start "$START" --end "$END" --output-train "$OUT_TRAIN")
if [ -n "$OUT_VAL" ] && [ "$VAL_RATIO" != "0" ] && [ "$VAL_RATIO" != "0.0" ]; then
  Args+=(--val-ratio "$VAL_RATIO" --output-val "$OUT_VAL")
fi

python3 "$EVOR/build_math_parquet.py" "${Args[@]}"
