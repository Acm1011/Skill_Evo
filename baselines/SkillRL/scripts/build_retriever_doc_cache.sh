#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASE_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/SkillRL/scripts/build_retriever_doc_cache.sh [options]

Input:
  --skills-json <path>              Claude-style skills json

Retriever:
  --retriever-url <url>             Retriever URL (default: http://127.0.0.1:8766)
  --retriever-start-script <path>   Startup script (default: skill_src/Zero/start_retriever_server.sh)
  --no-start-retriever              Assume retriever already running; skip startup/cleanup
  --retriever-host <host>           Health-check host (default: from retriever-url)
  --retriever-port <port>           Health-check port (default: from retriever-url)
  --retriever-max-wait <sec>        Health-check timeout seconds (default: 120)
  --retriever-doc-cache-dir <path>  Persist emb_*.npy here via retriever --doc-cache-dir

Behavior:
  Build candidates from:
    - general_skills
    - all task_specific_skills buckets
    - common_mistakes
  Then call POST /docs/replace once to pre-encode and persist document embeddings.
USAGE
}

SKILLS_JSON="${REPO_ROOT}/baselines/SkillRL/outputs/skills_from_rollout_teacher.json"
RETRIEVER_URL="http://127.0.0.1:8766"
RETRIEVER_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_retriever_server.sh"
START_RETRIEVER="1"
RETRIEVER_HOST=""
RETRIEVER_PORT=""
RETRIEVER_MAX_WAIT="120"
RETRIEVER_DOC_CACHE_DIR="${REPO_ROOT}/baselines/SkillRL/outputs/retriever_doc_cache"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skills-json) SKILLS_JSON="$2"; shift 2 ;;
        --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
        --retriever-start-script) RETRIEVER_START_SCRIPT="$2"; shift 2 ;;
        --no-start-retriever) START_RETRIEVER="0"; shift ;;
        --retriever-host) RETRIEVER_HOST="$2"; shift 2 ;;
        --retriever-port) RETRIEVER_PORT="$2"; shift 2 ;;
        --retriever-max-wait) RETRIEVER_MAX_WAIT="$2"; shift 2 ;;
        --retriever-doc-cache-dir) RETRIEVER_DOC_CACHE_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[build-retriever-cache] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

if [[ ! -f "${SKILLS_JSON}" ]]; then
    echo "[build-retriever-cache] skills json not found: ${SKILLS_JSON}" >&2
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
print(u.port or 8766)
PY
)"
fi

RETRIEVER_LAUNCHER_PID=""
cleanup() {
    local status=$?
    if [[ -n "${RETRIEVER_LAUNCHER_PID}" ]] && kill -0 "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null; then
        echo "[build-retriever-cache] stopping retriever..."
        kill -TERM "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
        wait "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ "${START_RETRIEVER}" == "1" ]]; then
    if [[ ! -x "${RETRIEVER_START_SCRIPT}" ]]; then
        echo "[build-retriever-cache] retriever start script missing or not executable: ${RETRIEVER_START_SCRIPT}" >&2
        exit 1
    fi
    export RETRIEVER_DOC_CACHE_DIR
    export SE_RETRIEVER_DOC_CACHE_DIR="${RETRIEVER_DOC_CACHE_DIR}"
    echo "[build-retriever-cache] retriever doc cache dir: ${RETRIEVER_DOC_CACHE_DIR}"
    echo "[build-retriever-cache] starting retriever..."
    bash "${RETRIEVER_START_SCRIPT}" &
    RETRIEVER_LAUNCHER_PID=$!
fi

echo "[build-retriever-cache] waiting for retriever health check..."
HEALTH_URL="http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health"
ok=0
for _ in $(seq 1 "${RETRIEVER_MAX_WAIT}"); do
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
        ok=1
        break
    fi
    sleep 1
done
if [[ "${ok}" -ne 1 ]]; then
    echo "[build-retriever-cache] health check failed: ${HEALTH_URL}" >&2
    exit 1
fi
echo "[build-retriever-cache] healthy: ${HEALTH_URL}"

python - <<'PY' "${SKILLS_JSON}" "${RETRIEVER_URL}"
import json
import sys
import urllib.request

skills_json, retriever_url = sys.argv[1], sys.argv[2].rstrip("/")

with open(skills_json, "r", encoding="utf-8") as f:
    bank = json.load(f)

items = []
seen = set()

def push(item_id: str, text: str):
    item_id = (item_id or "").strip()
    text = (text or "").strip()
    if not item_id or not text or item_id in seen:
        return
    seen.add(item_id)
    items.append({"id": item_id, "text": text})

for s in bank.get("general_skills", []):
    sid = str(s.get("skill_id") or "")
    when = str(s.get("when_to_apply") or "").strip()
    text = when or " ".join(
        p for p in [str(s.get("title") or "").strip(), str(s.get("principle") or "").strip()] if p
    )
    push(sid, text)

task = bank.get("task_specific_skills", {})
if isinstance(task, dict):
    for _, arr in task.items():
        if not isinstance(arr, list):
            continue
        for s in arr:
            sid = str(s.get("skill_id") or "")
            when = str(s.get("when_to_apply") or "").strip()
            text = when or " ".join(
                p for p in [str(s.get("title") or "").strip(), str(s.get("principle") or "").strip()] if p
            )
            push(sid, text)

for i, m in enumerate(bank.get("common_mistakes", [])):
    desc = str(m.get("description") or "").strip()
    fix = str(m.get("how_to_avoid") or "").strip()
    text = " ".join(p for p in [desc, fix] if p)
    push(f"cm_{i:06d}", text)

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
print(f"[build-retriever-cache] docs replaced: {data.get('n', 0)}")
print(f"[build-retriever-cache] doc_cache_dir: {data.get('doc_cache_dir')}")
PY

echo "[build-retriever-cache] done."
