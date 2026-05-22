#!/bin/bash
# =============================================================================
# eval_base_model_only.sh - 仅评测 base model（保存为 step_0）
# =============================================================================
#
# 用法:
#   ./eval_base_model_only.sh --exp_name <exp_name> --base_model_name <base_model_name> [options]
#
# 参数:
#   --exp_name          实验名称 (必需)
#   --base_model_name   基础模型名称 (必需)
#   --temperature       temp_data 采样温度 (默认: 0.7)
#   --temp_data_file    temp_data parquet 路径
#   --greedy_data_file  greedy_data parquet 路径
#
# 输出:
#   ${save_path_dir}/${EXP_NAME}_temperature${TEMPERATURE}/step_0
# =============================================================================

set -euo pipefail
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

EXP_NAME="${SB_EXP_NAME:-}"
BASE_MODEL_NAME="${SB_MODEL_NAME:-}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TEMP_DATA_FILE="${TEMP_DATA_FILE:-/home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b/temp_data_skill.parquet}"
GREEDY_DATA_FILE="${GREEDY_DATA_FILE:-/home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b/greedy_data_skill.parquet}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --base_model_name)
            BASE_MODEL_NAME="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --temp_data_file)
            TEMP_DATA_FILE="$2"
            shift 2
            ;;
        --greedy_data_file)
            GREEDY_DATA_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    echo "Error: --exp_name is required"
    exit 1
fi
if [ -z "$BASE_MODEL_NAME" ]; then
    echo "Error: --base_model_name is required"
    exit 1
fi
if [ ! -f "$TEMP_DATA_FILE" ]; then
    echo "Error: temp_data_file 不存在: $TEMP_DATA_FILE"
    exit 1
fi
if [ ! -f "$GREEDY_DATA_FILE" ]; then
    echo "Error: greedy_data_file 不存在: $GREEDY_DATA_FILE"
    exit 1
fi

project_name="${SB_PROJECT_NAME:-Skill_Evo}"
dir="${SB_BASB_DIR:-/home/ycy/sdi}"
model_dir="${SB_MODEL_DIR:-${dir}/models}"
saved_results_dir="${SB_SAVED_RESULTS_DIR:-/home/ycy/sdi/skill_saved/evaluation}"
WORKING_DIR="${SB_WORKING_DIR:-${dir}/${project_name}}"
save_path_dir="${saved_results_dir}/evaluation"
eval_path="${WORKING_DIR}/evaluation"
tb_path_dir="${saved_results_dir}/eval_tb_log"
CUSTOM_EVAL_DATA_DIR="${WORKING_DIR}/evaluation/.eval_custom_data/${EXP_NAME}"

base_model_path="${model_dir}/${BASE_MODEL_NAME}"
eval_saved_path_dir="${save_path_dir}/${EXP_NAME}_temperature${TEMPERATURE}"
step0_dir="${eval_saved_path_dir}/step_0"

if [ ! -d "$base_model_path" ]; then
    echo "Error: base model 目录不存在: $base_model_path"
    exit 1
fi

mkdir -p "${eval_saved_path_dir}" "${step0_dir}" "${tb_path_dir}" "${WORKING_DIR}/eval_logs" "${CUSTOM_EVAL_DATA_DIR}"

normalize_eval_parquet() {
    local src_file="$1"
    local dst_file="$2"

    python - "$src_file" "$dst_file" <<'PY'
import json
import os
import sys
import pandas as pd

src_file, dst_file = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src_file)

def maybe_parse_json(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value

for column in ("prompt", "extra_info", "reward_model"):
    if column in df.columns:
        df[column] = df[column].map(maybe_parse_json)

tmp_file = f"{dst_file}.tmp"
df.to_parquet(tmp_file)
os.replace(tmp_file, dst_file)
print(f"规范化评测数据: {src_file} -> {dst_file}")
PY
}

is_valid_response_parquet() {
    local parquet_file="$1"

    [ -f "$parquet_file" ] || return 1

    python - "$parquet_file" <<'PY'
import sys
import pandas as pd

parquet_file = sys.argv[1]

try:
    df = pd.read_parquet(parquet_file, columns=None)
except Exception:
    raise SystemExit(1)

columns = set(df.columns)
has_required = {"data_source", "problem"} <= columns
has_prompt = "formatted_prompt" in columns or "prompt" in columns
has_responses = "responses" in columns or "response" in columns

raise SystemExit(0 if (has_required and has_prompt and has_responses) else 1)
PY
}

normalize_eval_parquet "${TEMP_DATA_FILE}" "${CUSTOM_EVAL_DATA_DIR}/temp_data.parquet"
normalize_eval_parquet "${GREEDY_DATA_FILE}" "${CUSTOM_EVAL_DATA_DIR}/greedy_data.parquet"

cd "${eval_path}"

now() {
    date '+%Y-%m-%d-%H-%M'
}

exec > >(tee -a "${WORKING_DIR}/eval_logs/eval_base_${EXP_NAME}-$(now).log") 2>&1

echo "=============================================="
echo "  Base Model Evaluation Only"
echo "=============================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "实验名称:       $EXP_NAME"
echo "基础模型:       $BASE_MODEL_NAME"
echo "模型目录:       $base_model_path"
echo "结果目录:       $step0_dir"
echo "temp_data 文件: $TEMP_DATA_FILE"
echo "greedy_data 文件: $GREEDY_DATA_FILE"
echo "=============================================="

run_dataset_eval() {
    local dataset="$1"
    local n_samples="$2"
    local temp="$3"
    local response_file="${step0_dir}/${dataset}_responses.parquet"

    if is_valid_response_parquet "$response_file"; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Skip ${dataset}, valid result exists: ${response_file}"
        return 0
    fi

    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Running ${dataset} for ${BASE_MODEL_NAME}"
    python eval_all_math_step.py \
        --model_path "${base_model_path}" \
        --model_name "${BASE_MODEL_NAME}" \
        --dataset "${dataset}" \
        --save_path_dir "${eval_saved_path_dir}" \
        --n_samples "${n_samples}" \
        --temperature "${temp}" \
        --data_path_dir "${CUSTOM_EVAL_DATA_DIR}" \
        --step 0

    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Post-processing ${dataset}"
    python post_eval_step.py \
        --save_path_dir "${eval_saved_path_dir}" \
        --dataset "${dataset}" \
        --model_name "${BASE_MODEL_NAME}" \
        --n_samples "${n_samples}" \
        --temperature "${temp}" \
        --step 0
}

run_dataset_eval "temp_data" "32" "${TEMPERATURE}"
run_dataset_eval "greedy_data" "1" "0.0"

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Aggregating evaluation results..."
python aggregate_eval_results_step.py \
    --save_path_dir "${eval_saved_path_dir}"

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Uploading to TensorBoard..."
python tb_step.py \
    --exp_name="${EXP_NAME}" \
    --temperature="${TEMPERATURE}" \
    --save_path_dir="${eval_saved_path_dir}" \
    --tb_path_dir="${tb_path_dir}" \
    --generate_table

echo "=============================================="
echo "评测完成"
echo "  step_0: ${step0_dir}"
echo "  TensorBoard: ${tb_path_dir}/${EXP_NAME}-temperature_${TEMPERATURE}"
echo "=============================================="
