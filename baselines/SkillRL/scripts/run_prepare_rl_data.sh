#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${BASE_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/SkillRL/scripts/run_prepare_rl_data.sh [options]

Input:
  --deepmath-jsonl <path>        DeepMath jsonl (default: /home/ycy/sdi/data/DeepMath-103K.jsonl)
  --skills-json <path>           Claude-style skills json
  --start <n>                    Start row index (default: 0)
  --end <n>                      End row index, exclusive (default: read to EOF)

Retrieval:
  --retriever-url <url>          Retriever URL (default: http://127.0.0.1:8766)
  --mode <embedding|hybrid>      Retriever mode (default: embedding)
  --retrieve-lambda <f>          Hybrid lambda (default: 0.5)
  --top-k-general <n>            General skill quota (default: 3)
  --top-k-task <n>               Task skill quota (default: 3)
  --top-k-mistake <n>            Mistake quota (default: 2)

Output:
  --output-jsonl <path>          Output RL jsonl
  --output-parquet <path>        Output RL parquet
  --keep-raw-prompt              Preserve original prompt when present
  --fail-on-retrieve-error       Exit immediately on first retrieve failure

Retriever startup:
  --retriever-start-script <path>  Startup script (default: skill_src/Zero/start_retriever_server.sh)
  --retriever-doc-cache-dir <path> Document embedding cache dir (pass to retriever startup env)
  --no-start-retriever             Assume retriever is already running; skip startup/cleanup
  --retriever-host <host>          Health-check host (default: from retriever-url)
  --retriever-port <port>          Health-check port (default: from retriever-url)
  --retriever-max-wait <sec>       Health-check timeout seconds (default: 120)

Example:
  bash baselines/SkillRL/scripts/run_prepare_rl_data.sh \\
    --start 0 --end 1000 \\
    --top-k-general 3 --top-k-task 3 --top-k-mistake 2 \\
    --output-jsonl baselines/SkillRL/outputs/deepmath_skills_rl.jsonl \\
    --output-parquet baselines/SkillRL/outputs/deepmath_skills_rl.parquet
USAGE
}

DEEPMATH_JSONL="/home/ycy/sdi/data/DeepMath-103K.jsonl"
SKILLS_JSON="${REPO_ROOT}/baselines/SkillRL/outputs/skills_from_rollout_teacher.json"
RETRIEVER_URL="http://127.0.0.1:8766"
MODE="embedding"
RETRIEVE_LAMBDA="0.5"
START="0"
END=""
TOP_K_GENERAL="3"
TOP_K_TASK="3"
TOP_K_MISTAKE="2"
OUTPUT_JSONL="${REPO_ROOT}/baselines/SkillRL/outputs/deepmath_skills_rl.jsonl"
OUTPUT_PARQUET="${REPO_ROOT}/baselines/SkillRL/outputs/deepmath_skills_rl.parquet"
KEEP_RAW_PROMPT="0"
FAIL_ON_RETRIEVE_ERROR="0"

RETRIEVER_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_retriever_server.sh"
RETRIEVER_DOC_CACHE_DIR=""
START_RETRIEVER="1"
RETRIEVER_HOST=""
RETRIEVER_PORT=""
RETRIEVER_MAX_WAIT="120"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deepmath-jsonl) DEEPMATH_JSONL="$2"; shift 2 ;;
        --skills-json) SKILLS_JSON="$2"; shift 2 ;;
        --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --retrieve-lambda) RETRIEVE_LAMBDA="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --end) END="$2"; shift 2 ;;
        --top-k-general) TOP_K_GENERAL="$2"; shift 2 ;;
        --top-k-task) TOP_K_TASK="$2"; shift 2 ;;
        --top-k-mistake) TOP_K_MISTAKE="$2"; shift 2 ;;
        --output-jsonl) OUTPUT_JSONL="$2"; shift 2 ;;
        --output-parquet) OUTPUT_PARQUET="$2"; shift 2 ;;
        --keep-raw-prompt) KEEP_RAW_PROMPT="1"; shift ;;
        --fail-on-retrieve-error) FAIL_ON_RETRIEVE_ERROR="1"; shift ;;
        --retriever-start-script) RETRIEVER_START_SCRIPT="$2"; shift 2 ;;
        --retriever-doc-cache-dir) RETRIEVER_DOC_CACHE_DIR="$2"; shift 2 ;;
        --no-start-retriever) START_RETRIEVER="0"; shift ;;
        --retriever-host) RETRIEVER_HOST="$2"; shift 2 ;;
        --retriever-port) RETRIEVER_PORT="$2"; shift 2 ;;
        --retriever-max-wait) RETRIEVER_MAX_WAIT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[prepare-rl-data] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

if [[ ! -f "${DEEPMATH_JSONL}" ]]; then
    echo "[prepare-rl-data] deepmath jsonl not found: ${DEEPMATH_JSONL}" >&2
    exit 1
fi
if [[ ! -f "${SKILLS_JSON}" ]]; then
    echo "[prepare-rl-data] skills json not found: ${SKILLS_JSON}" >&2
    exit 1
fi

if [[ -n "${RETRIEVER_DOC_CACHE_DIR}" ]]; then
    if [[ "${RETRIEVER_DOC_CACHE_DIR}" != /* ]]; then
        RETRIEVER_DOC_CACHE_DIR="${REPO_ROOT}/${RETRIEVER_DOC_CACHE_DIR}"
    fi
fi

mkdir -p "$(dirname "${OUTPUT_JSONL}")" "$(dirname "${OUTPUT_PARQUET}")"

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
        echo "[prepare-rl-data] stopping retriever..."
        kill -TERM "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
        wait "${RETRIEVER_LAUNCHER_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ "${START_RETRIEVER}" == "1" ]]; then
    if [[ ! -x "${RETRIEVER_START_SCRIPT}" ]]; then
        echo "[prepare-rl-data] retriever start script missing or not executable: ${RETRIEVER_START_SCRIPT}" >&2
        exit 1
    fi

    if [[ -n "${RETRIEVER_DOC_CACHE_DIR}" ]]; then
        mkdir -p "${RETRIEVER_DOC_CACHE_DIR}"
        export RETRIEVER_DOC_CACHE_DIR
        export SE_RETRIEVER_DOC_CACHE_DIR="${RETRIEVER_DOC_CACHE_DIR}"
        echo "[prepare-rl-data] retriever doc cache dir: ${RETRIEVER_DOC_CACHE_DIR}"
    fi

    echo "[prepare-rl-data] starting retriever..."
    bash "${RETRIEVER_START_SCRIPT}" &
    RETRIEVER_LAUNCHER_PID=$!
fi

echo "[prepare-rl-data] waiting for retriever health check..."
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
    echo "[prepare-rl-data] health check failed: ${HEALTH_URL}" >&2
    exit 1
fi
echo "[prepare-rl-data] healthy: ${HEALTH_URL}"

CMD=(
    python -m baselines.SkillRL prepare-rl-data
    --deepmath-jsonl "${DEEPMATH_JSONL}"
    --skills-json "${SKILLS_JSON}"
    --retriever-url "${RETRIEVER_URL}"
    --start "${START}"
    --top-k-general "${TOP_K_GENERAL}"
    --top-k-task "${TOP_K_TASK}"
    --top-k-mistake "${TOP_K_MISTAKE}"
    --mode "${MODE}"
    --retrieve-lambda "${RETRIEVE_LAMBDA}"
    --output-jsonl "${OUTPUT_JSONL}"
    --output-parquet "${OUTPUT_PARQUET}"
)

if [[ -n "${END}" ]]; then
    CMD+=(--end "${END}")
fi
if [[ "${KEEP_RAW_PROMPT}" == "1" ]]; then
    CMD+=(--keep-raw-prompt)
fi
if [[ "${FAIL_ON_RETRIEVE_ERROR}" == "1" ]]; then
    CMD+=(--fail-on-retrieve-error)
fi

echo "[prepare-rl-data] running: ${CMD[*]}"
"${CMD[@]}"

echo "[prepare-rl-data] done."
