#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SOURCE_DATA_ROOT="/home/ycy/sdi/data"
OUTPUT_DIR="${REPO_ROOT}/baselines/outputs/general_benchmark_data"
SOURCES_CSV="ExpeLMath,MementoMath,ReasoningBankMath,SkillRL"

EXPEL_MEMORY_BANK=""
EXPEL_EMBEDDINGS=""
RBM_MEMORY_BANK=""
MEMENTO_MEMORY_BANK=""
MEMENTO_EMBEDDINGS=""
SKILLRL_SKILLS_JSON=""

TOP_K="5"
TOP_K_GENERAL="5"
TOP_K_TASK="5"
TOP_K_MISTAKE="2"
RETRIEVER_URL="http://127.0.0.1:8766"
RETRIEVER_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_retriever_server.sh"
START_RETRIEVER="1"
RETRIEVER_HOST=""
RETRIEVER_PORT=""
RETRIEVER_MAX_WAIT="120"
RETRIEVER_DOC_CACHE_DIR=""
MODE="embedding"
RETRIEVE_LAMBDA="0.5"

usage() {
    cat <<'EOF'
Usage:
  bash baselines/run_prepare_general_benchmark_data.sh [options]

Options:
  --source-data-root <path>      Root dir containing MMLU-Pro/ and SuperGPQA/
  --output-dir <path>            Output root dir
  --sources <csv>                Comma-separated sources: ExpeLMath,MementoMath,ReasoningBankMath,SkillRL
  --top-k <n>                    ExpeLMath / ReasoningBankMath top-k
  --top-k-general <n>            SkillRL general top-k
  --top-k-task <n>               SkillRL task top-k
  --top-k-mistake <n>            SkillRL mistake top-k
  --retriever-url <url>          Retriever URL for ReasoningBankMath / SkillRL
  --retriever-start-script <p>   Retriever startup script
  --no-start-retriever           Assume retriever already running; skip startup/cleanup
  --retriever-host <host>        Health-check host (default: from retriever-url)
  --retriever-port <port>        Health-check port (default: from retriever-url)
  --retriever-max-wait <sec>     Health-check timeout seconds
  --retriever-doc-cache-dir <p>  Export RETRIEVER_DOC_CACHE_DIR before startup
  --mode <embedding|hybrid>      Retriever mode
  --retrieve-lambda <f>          Hybrid retrieve lambda
  --expel-memory-bank <path>     Override ExpeLMath memory bank
  --expel-embeddings <path>      Override ExpeLMath embeddings
  --rbm-memory-bank <path>       Override ReasoningBankMath memory bank
  --memento-memory-bank <path>   Override MementoMath memory bank
  --memento-embeddings <path>    Override MementoMath embeddings
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
        --retriever-start-script) RETRIEVER_START_SCRIPT="$2"; shift 2 ;;
        --no-start-retriever) START_RETRIEVER="0"; shift ;;
        --retriever-host) RETRIEVER_HOST="$2"; shift 2 ;;
        --retriever-port) RETRIEVER_PORT="$2"; shift 2 ;;
        --retriever-max-wait) RETRIEVER_MAX_WAIT="$2"; shift 2 ;;
        --retriever-doc-cache-dir) RETRIEVER_DOC_CACHE_DIR="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --retrieve-lambda) RETRIEVE_LAMBDA="$2"; shift 2 ;;
        --expel-memory-bank) EXPEL_MEMORY_BANK="$2"; shift 2 ;;
        --expel-embeddings) EXPEL_EMBEDDINGS="$2"; shift 2 ;;
        --rbm-memory-bank) RBM_MEMORY_BANK="$2"; shift 2 ;;
        --memento-memory-bank) MEMENTO_MEMORY_BANK="$2"; shift 2 ;;
        --memento-embeddings) MEMENTO_EMBEDDINGS="$2"; shift 2 ;;
        --skillrl-skills-json) SKILLRL_SKILLS_JSON="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[run_prepare_general_benchmark_data] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

IFS=',' read -r -a SOURCES <<< "${SOURCES_CSV}"

if [[ -z "${RETRIEVER_HOST}" ]]; then
    RETRIEVER_HOST="$(python - <<'PY' "${RETRIEVER_URL}"
from urllib.parse import urlparse
import sys
u = urlparse(sys.argv[1])
print(u.hostname or "127.0.0.1")
PY
)"
fi

if [[ -z "${RETRIEVER_PORT}" ]]; then
    RETRIEVER_PORT="$(python - <<'PY' "${RETRIEVER_URL}"
from urllib.parse import urlparse
import sys
u = urlparse(sys.argv[1])
print(u.port or (443 if u.scheme == "https" else 80))
PY
)"
fi

needs_retriever=0
for source in "${SOURCES[@]}"; do
    case "${source}" in
        ReasoningBankMath)
            if (( TOP_K > 0 )); then
                needs_retriever=1
                break
            fi
            ;;
        SkillRL)
            if (( TOP_K_GENERAL > 0 || TOP_K_TASK > 0 || TOP_K_MISTAKE > 0 )); then
                needs_retriever=1
                break
            fi
            ;;
    esac
done

RETRIEVER_LAUNCHER_PID=""

cleanup() {
    if [[ -n "${RETRIEVER_LAUNCHER_PID}" ]] && kill -0 "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null; then
        echo "[run_prepare_general_benchmark_data] stopping retriever..."
        kill -TERM "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
        wait "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
    fi
}

if [[ "${needs_retriever}" == "1" && "${START_RETRIEVER}" == "1" ]]; then
    if [[ ! -x "${RETRIEVER_START_SCRIPT}" ]]; then
        echo "[run_prepare_general_benchmark_data] retriever start script missing or not executable: ${RETRIEVER_START_SCRIPT}" >&2
        exit 1
    fi
    if [[ -n "${RETRIEVER_DOC_CACHE_DIR}" ]]; then
        mkdir -p "${RETRIEVER_DOC_CACHE_DIR}"
        export RETRIEVER_DOC_CACHE_DIR
        export SE_RETRIEVER_DOC_CACHE_DIR="${RETRIEVER_DOC_CACHE_DIR}"
        echo "[run_prepare_general_benchmark_data] retriever doc cache dir: ${RETRIEVER_DOC_CACHE_DIR}"
    fi
    trap cleanup EXIT
    echo "[run_prepare_general_benchmark_data] starting retriever..."
    bash "${RETRIEVER_START_SCRIPT}" &
    RETRIEVER_LAUNCHER_PID=$!

    echo "[run_prepare_general_benchmark_data] waiting for retriever health check..."
    HEALTH_URL="http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health"
    for _ in $(seq 1 "${RETRIEVER_MAX_WAIT}"); do
        if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
            echo "[run_prepare_general_benchmark_data] retriever healthy: ${HEALTH_URL}"
            break
        fi
        sleep 1
    done
    if ! curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
        echo "[run_prepare_general_benchmark_data] retriever health check failed: ${HEALTH_URL}" >&2
        exit 1
    fi
elif [[ "${needs_retriever}" == "1" ]]; then
    echo "[run_prepare_general_benchmark_data] retriever required; assuming existing server at ${RETRIEVER_URL}"
else
    echo "[run_prepare_general_benchmark_data] retriever not needed for sources=${SOURCES_CSV}; skip startup"
fi

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
[[ -n "${MEMENTO_MEMORY_BANK}" ]] && CMD+=(--memento-memory-bank "${MEMENTO_MEMORY_BANK}")
[[ -n "${MEMENTO_EMBEDDINGS}" ]] && CMD+=(--memento-embeddings "${MEMENTO_EMBEDDINGS}")
[[ -n "${SKILLRL_SKILLS_JSON}" ]] && CMD+=(--skillrl-skills-json "${SKILLRL_SKILLS_JSON}")

"${CMD[@]}"
