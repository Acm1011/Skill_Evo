#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/ReasoningBankMath/scripts/run_prepare_prompt_data.sh [options]

Default behavior:
  Prepare both temp and greedy datasets into jsonl + parquet with retrieved memories.

Required:
  --memory-bank <path>
  --embeddings <path>

Optional:
  --temp-input <path>            Default: data/temp_data.jsonl
  --greedy-input <path>          Default: data/greedy_data.jsonl
  --out-dir <path>               Default: baselines/ReasoningBankMath/outputs/prepared
  --top-k <n>
  --embed-backend <name>
  --embed-base-url <url>
  --embed-api-key <key>
  --embed-model <name>
  --hash-dim <n>
USAGE
}

MEMORY_BANK=""
EMBEDDINGS=""
TEMP_INPUT="${REPO_ROOT}/data/temp_data.jsonl"
GREEDY_INPUT="${REPO_ROOT}/data/greedy_data.jsonl"
OUT_DIR="${REPO_ROOT}/baselines/ReasoningBankMath/outputs/prepared"
TOP_K="5"
EMBED_BACKEND="hash"
EMBED_BASE_URL=""
EMBED_API_KEY=""
EMBED_MODEL=""
HASH_DIM="256"
TIMEOUT="600"
TOPIC_BONUS="0.05"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --memory-bank) MEMORY_BANK="$2"; shift 2 ;;
    --embeddings) EMBEDDINGS="$2"; shift 2 ;;
    --temp-input) TEMP_INPUT="$2"; shift 2 ;;
    --greedy-input) GREEDY_INPUT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --embed-backend) EMBED_BACKEND="$2"; shift 2 ;;
    --embed-base-url) EMBED_BASE_URL="$2"; shift 2 ;;
    --embed-api-key) EMBED_API_KEY="$2"; shift 2 ;;
    --embed-model) EMBED_MODEL="$2"; shift 2 ;;
    --hash-dim) HASH_DIM="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --topic-bonus) TOPIC_BONUS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[rbm-prepare-prompt] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${MEMORY_BANK}" || -z "${EMBEDDINGS}" ]]; then
  echo "[rbm-prepare-prompt] --memory-bank and --embeddings are required" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

run_one() {
  local input_path="$1"
  local stem="$2"
  if [[ ! -f "${input_path}" ]]; then
    echo "[rbm-prepare-prompt] input not found: ${input_path}" >&2
    exit 1
  fi
  python -m baselines.ReasoningBankMath prepare-prompt-data \
    --input-jsonl "${input_path}" \
    --memory-bank "${MEMORY_BANK}" \
    --embeddings "${EMBEDDINGS}" \
    --top-k "${TOP_K}" \
    --embed-backend "${EMBED_BACKEND}" \
    --timeout "${TIMEOUT}" \
    --hash-dim "${HASH_DIM}" \
    --topic-bonus "${TOPIC_BONUS}" \
    --data-source "${stem}" \
    --output-jsonl "${OUT_DIR}/${stem}.jsonl" \
    --output-parquet "${OUT_DIR}/${stem}.parquet" \
    --keep-raw-prompt
}

if [[ -n "${EMBED_BASE_URL}" ]]; then export RBM_EMBED_BASE_URL="${EMBED_BASE_URL}"; fi
if [[ -n "${EMBED_API_KEY}" ]]; then export RBM_EMBED_API_KEY="${EMBED_API_KEY}"; fi
if [[ -n "${EMBED_MODEL}" ]]; then export RBM_EMBED_MODEL="${EMBED_MODEL}"; fi

run_one "${TEMP_INPUT}" "temp_data_memory_prompt"
run_one "${GREEDY_INPUT}" "greedy_data_memory_prompt"

