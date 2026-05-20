#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/ReasoningBankMath/scripts/build_retriever_doc_cache.sh [options]

Input:
  --memory-bank <path>              Refined memory bank jsonl

Retriever:
  --retriever-url <url>             Retriever URL (default: http://127.0.0.1:8766)
  --retriever-start-script <path>   Startup script (default: skill_src/Zero/start_retriever_server.sh)
  --no-start-retriever              Assume retriever already running
  --retriever-host <host>
  --retriever-port <port>
  --retriever-max-wait <sec>
  --retriever-doc-cache-dir <path>  Persist emb_*.npy here
USAGE
}

MEMORY_BANK="${REPO_ROOT}/baselines/ReasoningBankMath/outputs/memory_bank_v1_v2_refined.jsonl"
RETRIEVER_URL="http://127.0.0.1:8766"
RETRIEVER_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_retriever_server.sh"
START_RETRIEVER="1"
RETRIEVER_HOST=""
RETRIEVER_PORT=""
RETRIEVER_MAX_WAIT="120"
RETRIEVER_DOC_CACHE_DIR="${REPO_ROOT}/baselines/ReasoningBankMath/outputs/retriever_doc_cache"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --memory-bank) MEMORY_BANK="$2"; shift 2 ;;
        --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
        --retriever-start-script) RETRIEVER_START_SCRIPT="$2"; shift 2 ;;
        --no-start-retriever) START_RETRIEVER="0"; shift ;;
        --retriever-host) RETRIEVER_HOST="$2"; shift 2 ;;
        --retriever-port) RETRIEVER_PORT="$2"; shift 2 ;;
        --retriever-max-wait) RETRIEVER_MAX_WAIT="$2"; shift 2 ;;
        --retriever-doc-cache-dir) RETRIEVER_DOC_CACHE_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[rbm-build-retriever-cache] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"
if [[ ! -f "${MEMORY_BANK}" ]]; then
    echo "[rbm-build-retriever-cache] memory bank not found: ${MEMORY_BANK}" >&2
    exit 1
fi
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
        echo "[rbm-build-retriever-cache] stopping retriever..."
        kill -TERM "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
        wait "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ "${START_RETRIEVER}" == "1" ]]; then
    if [[ ! -x "${RETRIEVER_START_SCRIPT}" ]]; then
        echo "[rbm-build-retriever-cache] retriever start script missing or not executable: ${RETRIEVER_START_SCRIPT}" >&2
        exit 1
    fi
    export RETRIEVER_DOC_CACHE_DIR
    export SE_RETRIEVER_DOC_CACHE_DIR="${RETRIEVER_DOC_CACHE_DIR}"
    echo "[rbm-build-retriever-cache] starting retriever..."
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
    echo "[rbm-build-retriever-cache] health check failed: ${HEALTH_URL}" >&2
    exit 1
fi

python - <<'PY' "${MEMORY_BANK}" "${RETRIEVER_URL}"
import json, sys, urllib.request
from baselines.ReasoningBankMath.prepare_prompt_data import build_memory_candidates
from baselines.ReasoningBankMath.io_utils import read_jsonl

memory_bank, retriever_url = sys.argv[1], sys.argv[2].rstrip("/")
rows = read_jsonl(memory_bank)
cands = build_memory_candidates(rows)
items = [{"id": c["id"], "text": c["problem_type"]} for c in cands]
payload = json.dumps({"items": items}).encode("utf-8")
req = urllib.request.Request(
    f"{retriever_url}/docs/replace",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
if not data.get("ok"):
    raise SystemExit(f"/docs/replace failed: {data}")
print(f"[rbm-build-retriever-cache] docs replaced: {data.get('n', 0)}")
print(f"[rbm-build-retriever-cache] doc_cache_dir: {data.get('doc_cache_dir')}")
PY

echo "[rbm-build-retriever-cache] done."

