#!/usr/bin/env bash
# Build train/val parquet for verl math RL (skills in user message).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$BASE_DIR/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

DEEPMATH_JSONL="${DEEPMATH_JSONL:?set DEEPMATH_JSONL}"
SKILLS_JSON="${SKILLS_JSON:?set SKILLS_JSON}"
START="${START:-0}"
END="${END:?set END}"
TOP_K="${TOP_K:-6}"
VAL_RATIO="${VAL_RATIO:-0.05}"
OUT_TRAIN="${OUT_TRAIN:-$BASE_DIR/outputs/deepmath_rl_train.parquet}"
OUT_VAL="${OUT_VAL:-$BASE_DIR/outputs/deepmath_rl_val.parquet}"

python -m baselines.SkillRL build-rl-parquet \
  --deepmath-jsonl "$DEEPMATH_JSONL" \
  --skills-json "$SKILLS_JSON" \
  --start "$START" \
  --end "$END" \
  --top-k "$TOP_K" \
  --val-ratio "$VAL_RATIO" \
  --output-train "$OUT_TRAIN" \
  --output-val "$OUT_VAL"
