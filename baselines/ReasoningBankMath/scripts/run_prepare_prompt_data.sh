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

Optional:
  --temp-input <path>            Default: data/temp_data.jsonl
  --greedy-input <path>          Default: data/greedy_data.jsonl
  --out-dir <path>               Default: baselines/ReasoningBankMath/outputs/prepared
  --top-k <n>
  --retriever-url <url>
  --retriever-start-script <path>
  --no-start-retriever
  --retriever-host <host>
  --retriever-port <port>
  --retriever-max-wait <sec>
  --retriever-doc-cache-dir <path>
USAGE
}

MEMORY_BANK=""
TEMP_INPUT="${REPO_ROOT}/data/temp_data.jsonl"
GREEDY_INPUT="${REPO_ROOT}/data/greedy_data.jsonl"
OUT_DIR="${REPO_ROOT}/baselines/ReasoningBankMath/outputs/prepared"
TOP_K="5"
RETRIEVER_URL="http://127.0.0.1:8766"
RETRIEVER_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_retriever_server.sh"
START_RETRIEVER="1"
RETRIEVER_HOST=""
RETRIEVER_PORT=""
RETRIEVER_MAX_WAIT="120"
RETRIEVER_DOC_CACHE_DIR="${REPO_ROOT}/baselines/ReasoningBankMath/outputs/retriever_doc_cache"
MODE="embedding"
RETRIEVE_LAMBDA="0.5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --memory-bank) MEMORY_BANK="$2"; shift 2 ;;
    --temp-input) TEMP_INPUT="$2"; shift 2 ;;
    --greedy-input) GREEDY_INPUT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
    --retriever-start-script) RETRIEVER_START_SCRIPT="$2"; shift 2 ;;
    --no-start-retriever) START_RETRIEVER="0"; shift ;;
    --retriever-host) RETRIEVER_HOST="$2"; shift 2 ;;
    --retriever-port) RETRIEVER_PORT="$2"; shift 2 ;;
    --retriever-max-wait) RETRIEVER_MAX_WAIT="$2"; shift 2 ;;
    --retriever-doc-cache-dir) RETRIEVER_DOC_CACHE_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --retrieve-lambda) RETRIEVE_LAMBDA="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[rbm-prepare-prompt] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${MEMORY_BANK}" ]]; then
  echo "[rbm-prepare-prompt] --memory-bank is required" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

if [[ "${RETRIEVER_DOC_CACHE_DIR}" != /* ]]; then
  RETRIEVER_DOC_CACHE_DIR="${REPO_ROOT}/${RETRIEVER_DOC_CACHE_DIR}"
fi
mkdir -p "${RETRIEVER_DOC_CACHE_DIR}"

if [[ -z "${RETRIEVER_HOST}" ]]; then
    RETRIEVER_HOST="$(python - <<'PY' "${RETRIEVER_URL}"
from urllib.parse import urlparse
import sys
u = urlparse(sys.argv[1]); print(u.hostname or "127.0.0.1")
PY
)"
fi
if [[ -z "${RETRIEVER_PORT}" ]]; then
    RETRIEVER_PORT="$(python - <<'PY' "${RETRIEVER_URL}"
from urllib.parse import urlparse
import sys
u = urlparse(sys.argv[1]); print(u.port or 8766)
PY
)"
fi

RETRIEVER_LAUNCHER_PID=""
cleanup() {
    local status=$?
    if [[ -n "${RETRIEVER_LAUNCHER_PID}" ]] && kill -0 "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null; then
        echo "[rbm-prepare-prompt] stopping retriever..."
        kill -TERM "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
        wait "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ "${START_RETRIEVER}" == "1" ]]; then
    if [[ ! -x "${RETRIEVER_START_SCRIPT}" ]]; then
        echo "[rbm-prepare-prompt] retriever start script missing or not executable: ${RETRIEVER_START_SCRIPT}" >&2
        exit 1
    fi
    export RETRIEVER_DOC_CACHE_DIR
    export SE_RETRIEVER_DOC_CACHE_DIR="${RETRIEVER_DOC_CACHE_DIR}"
    echo "[rbm-prepare-prompt] starting retriever..."
    bash "${RETRIEVER_START_SCRIPT}" &
    RETRIEVER_LAUNCHER_PID=$!
fi

HEALTH_URL="http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health"
ok=0
for _ in $(seq 1 "${RETRIEVER_MAX_WAIT}"); do
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then ok=1; break; fi
    sleep 1
done
if [[ "${ok}" -ne 1 ]]; then
    echo "[rbm-prepare-prompt] health check failed: ${HEALTH_URL}" >&2
    exit 1
fi

echo "[rbm-prepare-prompt] syncing memory docs to retriever..."
bash "${REPO_ROOT}/baselines/ReasoningBankMath/scripts/build_retriever_doc_cache.sh" \
  --memory-bank "${MEMORY_BANK}" \
  --retriever-url "${RETRIEVER_URL}" \
  --no-start-retriever \
  --retriever-host "${RETRIEVER_HOST}" \
  --retriever-port "${RETRIEVER_PORT}" \
  --retriever-doc-cache-dir "${RETRIEVER_DOC_CACHE_DIR}"

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
    --retriever-url "${RETRIEVER_URL}" \
    --top-k "${TOP_K}" \
    --mode "${MODE}" \
    --retrieve-lambda "${RETRIEVE_LAMBDA}" \
    --data-source "${stem}" \
    --output-jsonl "${OUT_DIR}/${stem}.jsonl" \
    --output-parquet "${OUT_DIR}/${stem}.parquet" \
    --keep-raw-prompt
}

run_one "${TEMP_INPUT}" "temp_data"
run_one "${GREEDY_INPUT}" "greedy_data"
