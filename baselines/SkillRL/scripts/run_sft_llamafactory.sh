#!/usr/bin/env bash
# Run LLaMA-Factory SFT (expects dataset already registered). Execute from Skill_Evo repo root or set PYTHONPATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LLAMAFACTORY_HOME="${LLAMAFACTORY_HOME:?set LLAMAFACTORY_HOME to your LLaMA-Factory clone}"

YAML_SRC="${SFT_YAML:-$BASE_DIR/llamafactory/deepmath_skills_lora.yaml}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$LLAMAFACTORY_HOME/saves/deepmath_skills_lora}"

TMP_YAML="$(mktemp)"
trap 'rm -f "$TMP_YAML"' EXIT
sed \
  -e "s|^model_name_or_path:.*|model_name_or_path: ${MODEL_PATH}|" \
  -e "s|^output_dir:.*|output_dir: ${OUTPUT_DIR}|" \
  "$YAML_SRC" > "$TMP_YAML"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$LLAMAFACTORY_HOME"
if ! command -v llamafactory-cli &>/dev/null; then
  echo "llamafactory-cli not found. Install LLaMA-Factory and ensure its venv is active." >&2
  exit 1
fi

echo "[run_sft_llamafactory] cwd=$LLAMAFACTORY_HOME config=$TMP_YAML"
llamafactory-cli train "$TMP_YAML"
