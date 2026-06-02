#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/MementoMath/scripts/run_prepare_prompt_data.sh [options]

Default behavior:
  Prepare both temp and greedy datasets into jsonl + parquet with retrieved skills.

Required:
  --memory-bank <path>
  --case-pool <path>
  --model-path <path>           best.pt or last.pt from run_train_retriever.sh

Optional:
  --model-name <name>           Encoder name/path used during retriever training
  --device <name>               cuda/cpu, default auto
  --temp-input <path>           Default: data/temp_data.jsonl
  --greedy-input <path>         Default: data/greedy_data.jsonl
  --out-dir <path>              Default: baselines/MementoMath/outputs/prepared
  --top-k <n>
USAGE
}

MEMORY_BANK=""
CASE_POOL=""
MODEL_PATH=""
MODEL_NAME="princeton-nlp/sup-simcse-roberta-base"
DEVICE=""
TEMP_INPUT="${REPO_ROOT}/data/temp_data.jsonl"
GREEDY_INPUT="${REPO_ROOT}/data/greedy_data.jsonl"
OUT_DIR="${REPO_ROOT}/baselines/MementoMath/outputs/prepared"
TOP_K="5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --memory-bank) MEMORY_BANK="$2"; shift 2 ;;
    --case-pool) CASE_POOL="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --temp-input) TEMP_INPUT="$2"; shift 2 ;;
    --greedy-input) GREEDY_INPUT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[mmm-prepare-prompt] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${MEMORY_BANK}" || -z "${CASE_POOL}" || -z "${MODEL_PATH}" ]]; then
  echo "[mmm-prepare-prompt] --memory-bank --case-pool --model-path are required" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

run_one() {
  local input_path="$1"
  local stem="$2"
  if [[ ! -f "${input_path}" ]]; then
    echo "[mmm-prepare-prompt] input not found: ${input_path}" >&2
    exit 1
  fi
  CMD=(
    python -m baselines.MementoMath prepare-prompt-data
    --input-jsonl "${input_path}"
    --memory-bank "${MEMORY_BANK}"
    --case-pool "${CASE_POOL}"
    --model-path "${MODEL_PATH}"
    --model-name "${MODEL_NAME}"
    --top-k "${TOP_K}"
    --data-source "${stem}"
    --output-jsonl "${OUT_DIR}/${stem}.jsonl"
    --output-parquet "${OUT_DIR}/${stem}.parquet"
    --keep-raw-prompt
  )
  if [[ -n "${DEVICE}" ]]; then
    CMD+=(--device "${DEVICE}")
  fi
  "${CMD[@]}"
}

run_one "${TEMP_INPUT}" "temp_data"
run_one "${GREEDY_INPUT}" "greedy_data"
