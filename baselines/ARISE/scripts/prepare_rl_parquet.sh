#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$BASE_DIR/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

DEEPMATH_JSONL="${DEEPMATH_JSONL:-/home/ycy/sdi/data/DeepMath-103K.jsonl}"
ARISE_LIBRARY_JSON="${ARISE_LIBRARY_JSON:?set ARISE_LIBRARY_JSON}"
START="${START:-0}"
END="${END:-}"
TOP_K="${TOP_K:-3}"
MODE="${MODE:-embedding}"
RETRIEVE_LAMBDA="${RETRIEVE_LAMBDA:-0.5}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8766}"
OUT_JSONL="${OUT_JSONL:-$BASE_DIR/outputs/deepmath_arise_rl.jsonl}"
OUT_PARQUET="${OUT_PARQUET:-$BASE_DIR/outputs/deepmath_arise_rl.parquet}"

CMD=(
  python -m baselines.ARISE prepare-rl-data
  --deepmath-jsonl "$DEEPMATH_JSONL"
  --library-json "$ARISE_LIBRARY_JSON"
  --retriever-url "$RETRIEVER_URL"
  --start "$START"
  --top-k "$TOP_K"
  --mode "$MODE"
  --retrieve-lambda "$RETRIEVE_LAMBDA"
  --output-jsonl "$OUT_JSONL"
  --output-parquet "$OUT_PARQUET"
)

if [[ -n "${END}" ]]; then
  CMD+=(--end "$END")
fi
if [[ "${INCLUDE_RESERVOIR:-0}" == "1" ]]; then
  CMD+=(--include-reservoir)
fi

"${CMD[@]}"
