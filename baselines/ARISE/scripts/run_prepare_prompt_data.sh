#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$BASE_DIR/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/ARISE/scripts/run_prepare_prompt_data.sh [options]

Required:
  --library-json <path>

Optional:
  --temp-input <path>            Default: /home/ycy/sdi/data/temp_data.jsonl
  --greedy-input <path>          Default: /home/ycy/sdi/data/greedy_data.jsonl
  --out-dir <path>               Default: baselines/ARISE/outputs/prepared
  --top-k <n>                    Default: 3
  --retriever-url <url>          Default: http://127.0.0.1:8766
  --mode <embedding|hybrid>      Default: embedding
  --retrieve-lambda <float>      Default: 0.5
  --include-reservoir
USAGE
}

LIBRARY_JSON=""
TEMP_INPUT="/home/ycy/sdi/data/temp_data.jsonl"
GREEDY_INPUT="/home/ycy/sdi/data/greedy_data.jsonl"
OUT_DIR="${REPO_ROOT}/baselines/ARISE/outputs/prepared"
TOP_K="3"
RETRIEVER_URL="http://127.0.0.1:8766"
MODE="embedding"
RETRIEVE_LAMBDA="0.5"
INCLUDE_RESERVOIR="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --library-json) LIBRARY_JSON="$2"; shift 2 ;;
    --temp-input) TEMP_INPUT="$2"; shift 2 ;;
    --greedy-input) GREEDY_INPUT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --retrieve-lambda) RETRIEVE_LAMBDA="$2"; shift 2 ;;
    --include-reservoir) INCLUDE_RESERVOIR="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[arise-prepare-prompt] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${LIBRARY_JSON}" ]]; then
  echo "[arise-prepare-prompt] --library-json is required" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

run_one() {
  local input_path="$1"
  local stem="$2"
  local cmd=(
    python -m baselines.ARISE prepare-prompt-data
    --input-jsonl "$input_path"
    --library-json "$LIBRARY_JSON"
    --retriever-url "$RETRIEVER_URL"
    --top-k "$TOP_K"
    --mode "$MODE"
    --retrieve-lambda "$RETRIEVE_LAMBDA"
    --output-jsonl "${OUT_DIR}/${stem}_skill.jsonl"
    --output-parquet "${OUT_DIR}/${stem}_skill.parquet"
  )
  if [[ "${INCLUDE_RESERVOIR}" == "1" ]]; then
    cmd+=(--include-reservoir)
  fi
  "${cmd[@]}"
}

run_one "${TEMP_INPUT}" "temp_data"
run_one "${GREEDY_INPUT}" "greedy_data"
