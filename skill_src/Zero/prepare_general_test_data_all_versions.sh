#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPARE_SCRIPT="${WORKING_DIR}/prepare_general_test_data.py"
RETRIEVER_START_SCRIPT="${SCRIPT_DIR}/start_retriever_server.sh"

usage() {
    cat <<'USAGE'
Usage:
  bash skill_src/Zero/prepare_general_test_data_all_versions.sh <memory_jsonl> [source_data_dir]

Arguments:
  memory_jsonl      Path to one memory_after_sol_vN.jsonl
  source_data_dir   Raw benchmark root. Defaults to /home/ycy/sdi/data

Environment:
  SE_PREPARE_GENERAL_TOP_K       Retriever top-k, default 5
  SE_PREPARE_GENERAL_OUTPUT_DIR  Output dir, default sibling dir of <memory_jsonl>/general_test_data
  RETRIEVER_HOST                 Default 127.0.0.1
  RETRIEVER_PORT                 Default 8766
  RETRIEVER_MAX_WAIT_S           Default 120
USAGE
}

if [ "$#" -lt 1 ]; then
    usage >&2
    exit 2
fi

MEMORY_JSONL="$1"
SOURCE_DATA_DIR="${2:-/home/ycy/sdi/data}"

if [ ! -f "${MEMORY_JSONL}" ]; then
    echo "Error: memory_jsonl not found: ${MEMORY_JSONL}" >&2
    exit 1
fi
if [ ! -d "${SOURCE_DATA_DIR}" ]; then
    echo "Error: source_data_dir not found: ${SOURCE_DATA_DIR}" >&2
    exit 1
fi
if [ ! -f "${PREPARE_SCRIPT}" ]; then
    echo "Error: prepare script not found: ${PREPARE_SCRIPT}" >&2
    exit 1
fi
if [ ! -f "${RETRIEVER_START_SCRIPT}" ]; then
    echo "Error: retriever start script not found: ${RETRIEVER_START_SCRIPT}" >&2
    exit 1
fi

MEMORY_DIR="$(dirname "${MEMORY_JSONL}")"
MEMORY_NAME="$(basename "${MEMORY_JSONL}")"
OUTPUT_DIR="${SE_PREPARE_GENERAL_OUTPUT_DIR:-${MEMORY_DIR}/general_test_data}"
TOP_K="${SE_PREPARE_GENERAL_TOP_K:-5}"
RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8766}"
RETRIEVER_MAX_WAIT_S="${RETRIEVER_MAX_WAIT_S:-120}"
mkdir -p "${OUTPUT_DIR}"

RETRIEVER_STARTED_BY_SCRIPT=0
RETRIEVER_PID=""
cleanup() {
    if [ "${RETRIEVER_STARTED_BY_SCRIPT}" -eq 1 ] && [ -n "${RETRIEVER_PID}" ]; then
        if kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
            kill "${RETRIEVER_PID}" 2>/dev/null || true
            wait "${RETRIEVER_PID}" 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT

wait_for_retriever() {
    local waited=0
    while [ "${waited}" -lt "${RETRIEVER_MAX_WAIT_S}" ]; do
        if curl -sf "http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

if curl -sf "http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health" >/dev/null 2>&1; then
    echo "[prepare_general_test_data_all_versions] retriever already healthy"
else
    echo "[prepare_general_test_data_all_versions] starting retriever server"
    MEMORY_PATH_DIR="${MEMORY_DIR}" bash "${RETRIEVER_START_SCRIPT}" &
    RETRIEVER_PID=$!
    RETRIEVER_STARTED_BY_SCRIPT=1
    if ! wait_for_retriever; then
        echo "Error: retriever not ready within ${RETRIEVER_MAX_WAIT_S}s" >&2
        exit 1
    fi
fi

if [[ "${MEMORY_NAME}" =~ ^memory_after_sol_v([0-9]+)\.jsonl$ ]]; then
    VERSION="${BASH_REMATCH[1]}"
    VERSION_DIR="${OUTPUT_DIR}/general_skill_v${VERSION}"
else
    VERSION_DIR="${OUTPUT_DIR}/general_skill"
fi

echo "[prepare_general_test_data_all_versions] processing ${MEMORY_JSONL} -> ${VERSION_DIR}"
python3 "${PREPARE_SCRIPT}" \
    --memory-jsonl "${MEMORY_JSONL}" \
    --source-data-dir "${SOURCE_DATA_DIR}" \
    --output-dir "${VERSION_DIR}" \
    --top-k "${TOP_K}"

echo "[prepare_general_test_data_all_versions] completed. output_dir=${OUTPUT_DIR}"
