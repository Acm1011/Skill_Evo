#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SOURCE_DATA_ROOT="/home/ycy/sdi/data"
OUTPUT_DIR="${REPO_ROOT}/baselines/outputs/general_benchmark_data"
SOURCES_CSV="ExpeLMath,ReasoningBankMath,SkillRL"

EXPEL_MEMORY_BANK=""
EXPEL_EMBEDDINGS=""
RBM_MEMORY_BANK=""
SKILLRL_SKILLS_JSON=""

TOP_K="5"
TOP_K_GENERAL="5"
TOP_K_TASK="5"
TOP_K_MISTAKE="2"
RETRIEVER_URL="http://127.0.0.1:8766"
MODE="embedding"
RETRIEVE_LAMBDA="0.5"

usage() {
    cat <<'EOF'
Usage:
  bash baselines/run_prepare_general_benchmark_data.sh [options]

Options:
  --source-data-root <path>      Root dir containing MMLU-Pro/ and SuperGPQA/
  --output-dir <path>            Output root dir
  --sources <csv>                Comma-separated sources: ExpeLMath,ReasoningBankMath,SkillRL
  --top-k <n>                    ExpeLMath / ReasoningBankMath top-k
  --top-k-general <n>            SkillRL general top-k
  --top-k-task <n>               SkillRL task top-k
  --top-k-mistake <n>            SkillRL mistake top-k
  --retriever-url <url>          Retriever URL for ReasoningBankMath / SkillRL
  --mode <embedding|hybrid>      Retriever mode
  --retrieve-lambda <f>          Hybrid retrieve lambda
  --expel-memory-bank <path>     Override ExpeLMath memory bank
  --expel-embeddings <path>      Override ExpeLMath embeddings
  --rbm-memory-bank <path>       Override ReasoningBankMath memory bank
  --skillrl-skills-json <path>   Override SkillRL skills json

Default output layout:
  <output-dir>/ExpeLMath/MMLU-Pro/...
  <output-dir>/ExpeLMath/SuperGPQA/...
  <output-dir>/ReasoningBankMath/...
  <output-dir>/SkillRL/...
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-data-root) SOURCE_DATA_ROOT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --sources) SOURCES_CSV="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --top-k-general) TOP_K_GENERAL="$2"; shift 2 ;;
        --top-k-task) TOP_K_TASK="$2"; shift 2 ;;
        --top-k-mistake) TOP_K_MISTAKE="$2"; shift 2 ;;
        --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --retrieve-lambda) RETRIEVE_LAMBDA="$2"; shift 2 ;;
        --expel-memory-bank) EXPEL_MEMORY_BANK="$2"; shift 2 ;;
        --expel-embeddings) EXPEL_EMBEDDINGS="$2"; shift 2 ;;
        --rbm-memory-bank) RBM_MEMORY_BANK="$2"; shift 2 ;;
        --skillrl-skills-json) SKILLRL_SKILLS_JSON="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[run_prepare_general_benchmark_data] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

IFS=',' read -r -a SOURCES <<< "${SOURCES_CSV}"

CMD=(
    python baselines/prepare_general_benchmark_data.py
    --source-data-root "${SOURCE_DATA_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --top-k "${TOP_K}"
    --top-k-general "${TOP_K_GENERAL}"
    --top-k-task "${TOP_K_TASK}"
    --top-k-mistake "${TOP_K_MISTAKE}"
    --retriever-url "${RETRIEVER_URL}"
    --mode "${MODE}"
    --retrieve-lambda "${RETRIEVE_LAMBDA}"
)

for source in "${SOURCES[@]}"; do
    [[ -n "${source}" ]] && CMD+=(--sources "${source}")
done

[[ -n "${EXPEL_MEMORY_BANK}" ]] && CMD+=(--expel-memory-bank "${EXPEL_MEMORY_BANK}")
[[ -n "${EXPEL_EMBEDDINGS}" ]] && CMD+=(--expel-embeddings "${EXPEL_EMBEDDINGS}")
[[ -n "${RBM_MEMORY_BANK}" ]] && CMD+=(--rbm-memory-bank "${RBM_MEMORY_BANK}")
[[ -n "${SKILLRL_SKILLS_JSON}" ]] && CMD+=(--skillrl-skills-json "${SKILLRL_SKILLS_JSON}")

"${CMD[@]}"
