#!/usr/bin/env bash
# Build Alpaca JSON for LLaMA-Factory, optionally copy into LLaMA-Factory/data/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$BASE_DIR/../.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

DEEPMATH_JSONL="${DEEPMATH_JSONL:?set DEEPMATH_JSONL}"
SKILLS_JSON="${SKILLS_JSON:?set SKILLS_JSON}"
OUT_JSON="${OUT_JSON:-${BASE_DIR}/outputs/deepmath_skills_sft.json}"
START="${START:-0}"
END="${END:?set END (exclusive line index)}"
TOP_K="${TOP_K:-6}"
TRAJ="${TRAJ:-}"

EXTRA=()
if [[ "${ONLY_CORRECT:-0}" == "1" ]]; then EXTRA+=(--only-correct); fi
if [[ "${FALLBACK_BOXED_GT:-0}" == "1" ]]; then EXTRA+=(--fallback-boxed-gt); fi

CMD=(python -m baselines.SkillRL build-sft
  --deepmath-jsonl "$DEEPMATH_JSONL"
  --skills-json "$SKILLS_JSON"
  --start "$START"
  --end "$END"
  --top-k "$TOP_K"
  --output "$OUT_JSON"
)
if [[ -n "$TRAJ" ]]; then CMD+=(--trajectories "$TRAJ"); fi
CMD+=("${EXTRA[@]}")

"${CMD[@]}"

if [[ -n "${LLAMAFACTORY_HOME:-}" ]]; then
  mkdir -p "$LLAMAFACTORY_HOME/data"
  cp -f "$OUT_JSON" "$LLAMAFACTORY_HOME/data/deepmath_skills_sft.json"
  echo "[prepare_sft_deepmath] Copied to $LLAMAFACTORY_HOME/data/deepmath_skills_sft.json"
  echo "[prepare_sft_deepmath] Merge snippet into dataset_info.json:"
  echo "    $BASE_DIR/llamafactory/dataset_info.snippet.json"
  echo "  -> \$LLAMAFACTORY_HOME/data/dataset_info.json (append key deepmath_skills_sft)"
fi
