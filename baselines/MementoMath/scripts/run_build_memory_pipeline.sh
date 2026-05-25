#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/MementoMath/scripts/run_build_memory_pipeline.sh [options]

Default behavior:
  1. Start rollout servers with skill_src/Zero/start_rollout_servers.sh
  2. Build memory bank from trajectories jsonl
  3. Export Memento-style case pool and dummy memory
  4. Build memory embeddings
  5. Build retriever training data

Required:
  --model <path>                 Model path for rollout teacher servers

Optional:
  --trajectories <path>          Input trajectories jsonl
  --memory-bank <path>           Output memory bank jsonl
  --case-pool-output <path>      Output memory.jsonl
  --dummy-memory-output <path>   Output dummy_memo.jsonl
  --embeddings <path>            Output memory embeddings jsonl
  --training-data <path>         Output training_data.jsonl
  --gpu-ids <ids>                GPU ids, comma separated
  --n-gpus <n>                   Number of GPUs/servers
  --rollout-host <host>          Rollout host
  --rollout-base-port <port>     Rollout base port
  --rollout-start-script <path>  Startup script
  --rollout-log-dir <path>       Rollout log dir
  --temperature <f>              Teacher sampling temperature
  --top-p <f>                    Teacher top-p
  --top-k <n>                    Teacher top-k
  --max-tokens <n>               Teacher max tokens
  --timeout <sec>                Teacher request timeout
  --embed-backend <name>         hash or openai
  --embed-base-url <url>         Used when embed-backend=openai
  --embed-api-key <key>          Used when embed-backend=openai
  --embed-model <name>           Used when embed-backend=openai
  --hash-dim <n>                 Used when embed-backend=hash
  --num-positive <n>             Positive pairs per query
  --num-negative <n>             Negative pairs per query
  --seed <n>                     Random seed for pair sampling
  --train-retriever              Train original Memento retriever after data prep
  --retriever-output-dir <path>  Retriever checkpoint dir
  --pretrained-name <name>       HF encoder name for retriever
  --pretrained-local <path>      Local encoder path for retriever
  --batch-size <n>               Retriever training batch size
  --epochs <n>                   Retriever training epochs
  --lr <f>                       Retriever training learning rate
  --fp16                         Enable fp16 for retriever training
  --save-best                    Save best.pt during retriever training
USAGE
}

TRAJ_PATH="${REPO_ROOT}/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl"
MODEL=""
MEMORY_BANK="${REPO_ROOT}/baselines/MementoMath/outputs/memory_bank_v1_v2.jsonl"
CASE_POOL_OUTPUT="${REPO_ROOT}/baselines/MementoMath/outputs/memory_v1_v2.jsonl"
DUMMY_MEMORY_OUTPUT="${REPO_ROOT}/baselines/MementoMath/outputs/dummy_memo_v1_v2.jsonl"
EMBEDDINGS="${REPO_ROOT}/baselines/MementoMath/outputs/memory_embeddings_v1_v2.jsonl"
TRAINING_DATA="${REPO_ROOT}/baselines/MementoMath/outputs/training_data_v1_v2.jsonl"
ROLLOUT_START_SCRIPT="${REPO_ROOT}/skill_src/Zero/start_rollout_servers.sh"
ROLLOUT_LOG_DIR="${REPO_ROOT}/baselines/MementoMath/outputs/logs/rollout_servers"
SE_ROLLOUT_HOST="127.0.0.1"
SE_ROLLOUT_BASE_PORT="8760"
SE_GPU_IDS="4,5,6,7"
SE_N_GPUS="4"
TEMPERATURE="0.7"
TOP_P="0.95"
TOP_K="50"
MAX_TOKENS="1024"
TIMEOUT="600"
EMBED_BACKEND="hash"
EMBED_BASE_URL=""
EMBED_API_KEY=""
EMBED_MODEL=""
HASH_DIM="256"
NUM_POSITIVE="4"
NUM_NEGATIVE="4"
SEED="42"
TRAIN_RETRIEVER="0"
RETRIEVER_OUTPUT_DIR="${REPO_ROOT}/baselines/MementoMath/outputs/retriever_ckpts"
PRETRAINED_NAME="princeton-nlp/sup-simcse-roberta-base"
PRETRAINED_LOCAL=""
TRAIN_BATCH_SIZE="32"
TRAIN_EPOCHS="3"
TRAIN_LR="2e-5"
TRAIN_FP16="0"
TRAIN_SAVE_BEST="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --trajectories) TRAJ_PATH="$2"; shift 2 ;;
        --memory-bank) MEMORY_BANK="$2"; shift 2 ;;
        --case-pool-output) CASE_POOL_OUTPUT="$2"; shift 2 ;;
        --dummy-memory-output) DUMMY_MEMORY_OUTPUT="$2"; shift 2 ;;
        --embeddings) EMBEDDINGS="$2"; shift 2 ;;
        --training-data) TRAINING_DATA="$2"; shift 2 ;;
        --gpu-ids) SE_GPU_IDS="$2"; shift 2 ;;
        --n-gpus) SE_N_GPUS="$2"; shift 2 ;;
        --rollout-host) SE_ROLLOUT_HOST="$2"; shift 2 ;;
        --rollout-base-port) SE_ROLLOUT_BASE_PORT="$2"; shift 2 ;;
        --rollout-start-script) ROLLOUT_START_SCRIPT="$2"; shift 2 ;;
        --rollout-log-dir) ROLLOUT_LOG_DIR="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --embed-backend) EMBED_BACKEND="$2"; shift 2 ;;
        --embed-base-url) EMBED_BASE_URL="$2"; shift 2 ;;
        --embed-api-key) EMBED_API_KEY="$2"; shift 2 ;;
        --embed-model) EMBED_MODEL="$2"; shift 2 ;;
        --hash-dim) HASH_DIM="$2"; shift 2 ;;
        --num-positive) NUM_POSITIVE="$2"; shift 2 ;;
        --num-negative) NUM_NEGATIVE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --train-retriever) TRAIN_RETRIEVER="1"; shift 1 ;;
        --retriever-output-dir) RETRIEVER_OUTPUT_DIR="$2"; shift 2 ;;
        --pretrained-name) PRETRAINED_NAME="$2"; shift 2 ;;
        --pretrained-local) PRETRAINED_LOCAL="$2"; shift 2 ;;
        --batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
        --lr) TRAIN_LR="$2"; shift 2 ;;
        --fp16) TRAIN_FP16="1"; shift 1 ;;
        --save-best) TRAIN_SAVE_BEST="1"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[mmm-pipeline] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "[mmm-pipeline] --model is required" >&2
    exit 2
fi
if [[ ! -f "${TRAJ_PATH}" ]]; then
    echo "[mmm-pipeline] trajectories not found: ${TRAJ_PATH}" >&2
    exit 1
fi
if [[ ! -x "${ROLLOUT_START_SCRIPT}" ]]; then
    echo "[mmm-pipeline] rollout start script missing or not executable: ${ROLLOUT_START_SCRIPT}" >&2
    exit 1
fi

mkdir -p "$(dirname "${MEMORY_BANK}")" "$(dirname "${CASE_POOL_OUTPUT}")" "$(dirname "${DUMMY_MEMORY_OUTPUT}")" "$(dirname "${EMBEDDINGS}")" "$(dirname "${TRAINING_DATA}")" "${ROLLOUT_LOG_DIR}" "${RETRIEVER_OUTPUT_DIR}"
if [[ "${MODEL}" != /* ]]; then
    MODEL="${REPO_ROOT}/${MODEL}"
fi
if [[ ! -e "${MODEL}" ]]; then
    echo "[mmm-pipeline] model path not found: ${MODEL}" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export SE_GPU_IDS
export SE_N_GPUS
export SE_ROLLOUT_HOST
export SE_ROLLOUT_BASE_PORT
export SE_ROLLOUT_N_SERVERS="${SE_N_GPUS}"
export SE_ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR}"
export ROLLOUT_SERVER_MODEL="${MODEL}"

echo "[mmm-pipeline] trajectories: ${TRAJ_PATH}"
echo "[mmm-pipeline] memory bank: ${MEMORY_BANK}"
echo "[mmm-pipeline] case pool: ${CASE_POOL_OUTPUT}"
echo "[mmm-pipeline] dummy memory: ${DUMMY_MEMORY_OUTPUT}"
echo "[mmm-pipeline] embeddings: ${EMBEDDINGS}"
echo "[mmm-pipeline] training data: ${TRAINING_DATA}"
echo "[mmm-pipeline] train retriever: ${TRAIN_RETRIEVER}"
echo "[mmm-pipeline] model: ${MODEL}"
echo "[mmm-pipeline] embed backend: ${EMBED_BACKEND}"

echo "[mmm-pipeline] starting rollout servers..."
bash "${ROLLOUT_START_SCRIPT}" --model "${MODEL}" &
ROLLOUT_LAUNCHER_PID=$!

cleanup() {
    local status=$?
    if kill -0 "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null; then
        echo "[mmm-pipeline] stopping rollout servers..."
        kill -TERM "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null || true
        wait "${ROLLOUT_LAUNCHER_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "[mmm-pipeline] waiting for rollout health checks..."
for ((i = 0; i < SE_ROLLOUT_N_SERVERS; i++)); do
    port=$((SE_ROLLOUT_BASE_PORT + i))
    url="http://${SE_ROLLOUT_HOST}:${port}/health"
    ok=0
    for _ in $(seq 1 120); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            ok=1
            break
        fi
        sleep 2
    done
    if [[ "${ok}" -ne 1 ]]; then
        echo "[mmm-pipeline] rollout health check failed: ${url}" >&2
        exit 1
    fi
    echo "[mmm-pipeline] healthy: ${url}"
done

echo "[mmm-pipeline] building memory bank..."
python -m baselines.MementoMath build-memory \
    --teacher-backend rollout \
    --trajectories "${TRAJ_PATH}" \
    --output "${MEMORY_BANK}" \
    --case-pool-output "${CASE_POOL_OUTPUT}" \
    --dummy-memory-output "${DUMMY_MEMORY_OUTPUT}" \
    --rollout-host "${SE_ROLLOUT_HOST}" \
    --rollout-base-port "${SE_ROLLOUT_BASE_PORT}" \
    --rollout-n-servers "${SE_N_GPUS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-tokens "${MAX_TOKENS}" \
    --timeout "${TIMEOUT}"

echo "[mmm-pipeline] building embeddings..."
EMBED_CMD=(
  python -m baselines.MementoMath build-embeddings
  --memory-bank "${MEMORY_BANK}"
  --output "${EMBEDDINGS}"
  --backend "${EMBED_BACKEND}"
  --timeout "${TIMEOUT}"
  --hash-dim "${HASH_DIM}"
)
if [[ -n "${EMBED_BASE_URL}" ]]; then
  EMBED_CMD+=(--embed-base-url "${EMBED_BASE_URL}")
fi
if [[ -n "${EMBED_API_KEY}" ]]; then
  EMBED_CMD+=(--embed-api-key "${EMBED_API_KEY}")
fi
if [[ -n "${EMBED_MODEL}" ]]; then
  EMBED_CMD+=(--embed-model "${EMBED_MODEL}")
fi
"${EMBED_CMD[@]}"

echo "[mmm-pipeline] building training data..."
TRAIN_CMD=(
  python -m baselines.MementoMath build-training-data
  --memory-bank "${MEMORY_BANK}"
  --embeddings "${EMBEDDINGS}"
  --output "${TRAINING_DATA}"
  --num-positive "${NUM_POSITIVE}"
  --num-negative "${NUM_NEGATIVE}"
  --seed "${SEED}"
  --backend "${EMBED_BACKEND}"
  --timeout "${TIMEOUT}"
  --hash-dim "${HASH_DIM}"
)
if [[ -n "${EMBED_BASE_URL}" ]]; then
  TRAIN_CMD+=(--embed-base-url "${EMBED_BASE_URL}")
fi
if [[ -n "${EMBED_API_KEY}" ]]; then
  TRAIN_CMD+=(--embed-api-key "${EMBED_API_KEY}")
fi
if [[ -n "${EMBED_MODEL}" ]]; then
  TRAIN_CMD+=(--embed-model "${EMBED_MODEL}")
fi
"${TRAIN_CMD[@]}"

if [[ "${TRAIN_RETRIEVER}" == "1" ]]; then
  echo "[mmm-pipeline] training retriever..."
  RETR_CMD=(
    bash baselines/MementoMath/scripts/run_train_retriever.sh
    --train "${TRAINING_DATA}"
    --output-dir "${RETRIEVER_OUTPUT_DIR}"
    --pretrained-name "${PRETRAINED_NAME}"
    --batch-size "${TRAIN_BATCH_SIZE}"
    --epochs "${TRAIN_EPOCHS}"
    --lr "${TRAIN_LR}"
    --use-plan
  )
  if [[ -n "${PRETRAINED_LOCAL}" ]]; then
    RETR_CMD+=(--pretrained-local "${PRETRAINED_LOCAL}")
  fi
  if [[ "${TRAIN_FP16}" == "1" ]]; then
    RETR_CMD+=(--fp16)
  fi
  if [[ "${TRAIN_SAVE_BEST}" == "1" ]]; then
    RETR_CMD+=(--save-best)
  fi
  "${RETR_CMD[@]}"
fi

echo "[mmm-pipeline] done"
