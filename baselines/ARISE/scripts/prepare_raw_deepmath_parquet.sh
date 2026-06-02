#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$BASE_DIR/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

DEEPMATH_JSONL="${DEEPMATH_JSONL:-/home/ycy/sdi/data/DeepMath-103K.jsonl}"
OUT_PARQUET="${OUT_PARQUET:-$BASE_DIR/outputs/deepmath_raw_arise.parquet}"
START="${START:-0}"
END="${END:-}"
DATA_SOURCE="${DATA_SOURCE:-math_dapo}"

CMD=(
  python -m baselines.ARISE.build_deepmath_parquet
  --deepmath-jsonl "$DEEPMATH_JSONL"
  --output "$OUT_PARQUET"
  --data-source "$DATA_SOURCE"
  --start "$START"
)

if [[ -n "${END}" ]]; then
  CMD+=(--end "$END")
fi

"${CMD[@]}"
