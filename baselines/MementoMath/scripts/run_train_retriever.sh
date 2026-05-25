#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<USAGE
Usage:
  bash baselines/MementoMath/scripts/run_train_retriever.sh [options]

Required:
  --train <path>                training_data.jsonl path

Optional:
  --valid <path>                Optional validation jsonl
  --output-dir <path>           Retriever checkpoint dir
  --pretrained-name <name>      HF model name
  --pretrained-local <path>     Local encoder path
  --seed <n>
  --val-ratio <f>
  --no-stratify
  --use-plan
  --plan-style <name>           pretty or raw
  --batch-size <n>
  --epochs <n>
  --lr <f>
  --weight-decay <f>
  --warmup-ratio <f>
  --max-len <n>
  --grad-clip <f>
  --fp16
  --eval-every <n>
  --save-best
  --class-weight-pos <f>

Example:
  bash baselines/MementoMath/scripts/run_train_retriever.sh \
    --train baselines/MementoMath/outputs/training_data_v1_v2.jsonl \
    --output-dir baselines/MementoMath/outputs/retriever_ckpts \
    --use-plan --save-best --fp16
USAGE
}

TRAIN=""
VALID=""
OUTPUT_DIR="${REPO_ROOT}/baselines/MementoMath/outputs/retriever_ckpts"
PRETRAINED_NAME="princeton-nlp/sup-simcse-roberta-base"
PRETRAINED_LOCAL=""
SEED="42"
VAL_RATIO="0.1"
NO_STRATIFY="0"
USE_PLAN="0"
PLAN_STYLE="pretty"
BATCH_SIZE="32"
EPOCHS="3"
LR="2e-5"
WEIGHT_DECAY="0.01"
WARMUP_RATIO="0.06"
MAX_LEN="256"
GRAD_CLIP="1.0"
FP16="0"
EVAL_EVERY="500"
SAVE_BEST="0"
CLASS_WEIGHT_POS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train) TRAIN="$2"; shift 2 ;;
        --valid) VALID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --pretrained-name) PRETRAINED_NAME="$2"; shift 2 ;;
        --pretrained-local) PRETRAINED_LOCAL="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --val-ratio) VAL_RATIO="$2"; shift 2 ;;
        --no-stratify) NO_STRATIFY="1"; shift 1 ;;
        --use-plan) USE_PLAN="1"; shift 1 ;;
        --plan-style) PLAN_STYLE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --warmup-ratio) WARMUP_RATIO="$2"; shift 2 ;;
        --max-len) MAX_LEN="$2"; shift 2 ;;
        --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
        --fp16) FP16="1"; shift 1 ;;
        --eval-every) EVAL_EVERY="$2"; shift 2 ;;
        --save-best) SAVE_BEST="1"; shift 1 ;;
        --class-weight-pos) CLASS_WEIGHT_POS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[mmm-train-retriever] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${TRAIN}" ]]; then
    echo "[mmm-train-retriever] --train is required" >&2
    exit 2
fi
if [[ ! -f "${TRAIN}" ]]; then
    echo "[mmm-train-retriever] training file not found: ${TRAIN}" >&2
    exit 1
fi
if [[ -n "${VALID}" && ! -f "${VALID}" ]]; then
    echo "[mmm-train-retriever] validation file not found: ${VALID}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CMD=(
  python Memento/memory/train_memory_retriever.py
  --train "${TRAIN}"
  --output_dir "${OUTPUT_DIR}"
  --pretrained_name "${PRETRAINED_NAME}"
  --seed "${SEED}"
  --val_ratio "${VAL_RATIO}"
  --plan_style "${PLAN_STYLE}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --max_len "${MAX_LEN}"
  --grad_clip "${GRAD_CLIP}"
  --eval_every "${EVAL_EVERY}"
)
if [[ -n "${VALID}" ]]; then
  CMD+=(--valid "${VALID}")
fi
if [[ -n "${PRETRAINED_LOCAL}" ]]; then
  CMD+=(--pretrained_local "${PRETRAINED_LOCAL}")
fi
if [[ "${NO_STRATIFY}" == "1" ]]; then
  CMD+=(--no_stratify)
fi
if [[ "${USE_PLAN}" == "1" ]]; then
  CMD+=(--use_plan)
fi
if [[ "${FP16}" == "1" ]]; then
  CMD+=(--fp16)
fi
if [[ "${SAVE_BEST}" == "1" ]]; then
  CMD+=(--save_best)
fi
if [[ -n "${CLASS_WEIGHT_POS}" ]]; then
  CMD+=(--class_weight_pos "${CLASS_WEIGHT_POS}")
fi

echo "[mmm-train-retriever] train: ${TRAIN}"
echo "[mmm-train-retriever] output: ${OUTPUT_DIR}"
"${CMD[@]}"
