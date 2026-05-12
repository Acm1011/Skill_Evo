#!/usr/bin/env bash
# Shared defaults for EvolveR math data prep, services, and training.
# Override any variable before invoking a script if needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVOR/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${EVOR}:${PYTHONPATH:-}"

# Experiment naming / output layout
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepmath-evolver-rl}"
export WAND_PROJECT="${WAND_PROJECT:-EvolveR-Math}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$EVOR/outputs}"
export CKPTS_DIR="${CKPTS_DIR:-$OUTPUT_ROOT/ckpts}"
export EXPERIENCE_EXPORT_DIR="${EXPERIENCE_EXPORT_DIR:-$OUTPUT_ROOT/experience_runs}"
export VDB_BASE_DIR="${VDB_BASE_DIR:-$OUTPUT_ROOT/milvus_work}"

# Dataset paths
export DEEPMATH_JSONL="${DEEPMATH_JSONL:-$REPO_ROOT/data/DeepMath-103K.jsonl}"
export OUT_TRAIN="${OUT_TRAIN:-$OUTPUT_ROOT/deepmath_rl_train.parquet}"
export OUT_VAL="${OUT_VAL:-$OUTPUT_ROOT/deepmath_rl_val.parquet}"
export TRAIN_FILE="${TRAIN_FILE:-$OUT_TRAIN}"
export VAL_FILE="${VAL_FILE:-$OUT_VAL}"
export START="${START:-0}"
export END="${END:-10000}"
export VAL_RATIO="${VAL_RATIO:-0.05}"

# Model / serving defaults
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
export EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-BAAI/bge-m3}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge_m3}"
export EMBED_PORT="${EMBED_PORT:-8081}"
export EMBEDDING_API_URL="${EMBEDDING_API_URL:-http://127.0.0.1:${EMBED_PORT}/v1}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-empty}"
export EMBED_CUDA="${EMBED_CUDA:-0}"
export GPU_NUM="${GPU_NUM:-1}"

# Experience / retriever services
export VDB_SERVER_URL="${VDB_SERVER_URL:-http://127.0.0.1:8007}"
export RETRIEVER_STUB_PORT="${RETRIEVER_STUB_PORT:-19999}"
export RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:${RETRIEVER_STUB_PORT}/retrieve}"

# Training toggles and sensible defaults
export EVOLVER_KNOWLEDGE_SEARCH="${EVOLVER_KNOWLEDGE_SEARCH:-0}"
export EVOLVER_QA_OUTCOME="${EVOLVER_QA_OUTCOME:-0}"
export USE_EXPERIENCE="${USE_EXPERIENCE:-true}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
