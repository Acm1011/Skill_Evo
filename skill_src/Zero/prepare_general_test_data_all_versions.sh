#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPARE_SCRIPT="${WORKING_DIR}/prepare_general_test_data.py"
RETRIEVER_START_SCRIPT="${SCRIPT_DIR}/start_retriever_server.sh"

usage() {
    cat <<'USAGE'
Usage:
  bash skill_src/Zero/prepare_general_test_data_all_versions.sh <memory_dir> [source_data_dir]

Arguments:
  memory_dir        Directory containing memory_after_sol_v*.jsonl
  source_data_dir   Raw benchmark root. Defaults to /home/ycy/sdi/data

Environment:
  SE_PREPARE_GENERAL_TOP_K       Retriever top-k, default 5
  SE_PREPARE_GENERAL_OUTPUT_DIR  Output dir, default parent of <memory_dir>/general_test_data
  RETRIEVER_HOST                 Default 127.0.0.1
  RETRIEVER_PORT                 Default 8766
  RETRIEVER_MAX_WAIT_S           Default 120
USAGE
}

if [ "$#" -lt 1 ]; then
    usage >&2
    exit 2
fi

MEMORY_DIR="$1"
SOURCE_DATA_DIR="${2:-/home/ycy/sdi/data}"

if [ ! -d "${MEMORY_DIR}" ]; then
    echo "Error: memory_dir not found: ${MEMORY_DIR}" >&2
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

OUTPUT_DIR="${SE_PREPARE_GENERAL_OUTPUT_DIR:-$(dirname "${MEMORY_DIR}")/general_test_data}"
TOP_K="${SE_PREPARE_GENERAL_TOP_K:-5}"
RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8766}"
RETRIEVER_MAX_WAIT_S="${RETRIEVER_MAX_WAIT_S:-120}"
mkdir -p "${OUTPUT_DIR}"

mapfile -t MEMORY_FILES < <(
    find "${MEMORY_DIR}" -maxdepth 1 -type f -name 'memory_after_sol_v*.jsonl' \
        | sed -E 's#(.*/memory_after_sol_v)([0-9]+)(\.jsonl)$#\2\t\1\2\3#' \
        | sort -n -k1,1 \
        | cut -f2-
)

if [ "${#MEMORY_FILES[@]}" -eq 0 ]; then
    echo "Error: no memory_after_sol_v*.jsonl found under ${MEMORY_DIR}" >&2
    exit 1
fi

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

for memory_jsonl in "${MEMORY_FILES[@]}"; do
    memory_name="$(basename "${memory_jsonl}")"
    if [[ ! "${memory_name}" =~ ^memory_after_sol_v([0-9]+)\.jsonl$ ]]; then
        echo "Error: unexpected memory file name: ${memory_name}" >&2
        exit 1
    fi
    version="${BASH_REMATCH[1]}"
    version_dir="${OUTPUT_DIR}/general_skill_v${version}"
    echo "[prepare_general_test_data_all_versions] processing ${memory_name} -> ${version_dir}"
    python3 "${PREPARE_SCRIPT}" \
        --memory-jsonl "${memory_jsonl}" \
        --source-data-dir "${SOURCE_DATA_DIR}" \
        --output-dir "${version_dir}" \
        --top-k "${TOP_K}"
done

echo "[prepare_general_test_data_all_versions] completed. output_dir=${OUTPUT_DIR}"
