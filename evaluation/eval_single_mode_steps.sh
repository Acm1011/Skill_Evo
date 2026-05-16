#!/bin/bash
# =============================================================================
# eval_single_model_steps.sh - 评测单个模型的多个 checkpoint steps
# =============================================================================
#
# 用法:
#   ./eval_single_model_steps.sh --exp_name <exp_name> [options]
#
# 参数:
#   --exp_name          实验名称 (必需)
#   --ckpts_dir         checkpoint 目录 (可选，默认自动根据 exp_name 计算)
#   --steps             可选：手动指定要评测的 steps (空格分隔)，不指定则自动发现
#   --base_model_name   基础模型名称 (默认: Qwen2.5-Math-1.5B)
#   --temperature       采样温度 (默认: 0.6)
#   --sample_ratio      采样比例 (默认: 0.1，用于 bbeh/mmlupro/supergpqa)
#   --skip_base_model   跳过基础模型评测 (默认: false)
#   --temp_data_file    temp_data parquet 路径（默认: 训练后生成的 temp_data_skill.parquet）
#   --greedy_data_file  greedy_data parquet 路径（默认: 训练后生成的 greedy_data_skill.parquet）
#
# 示例:
#   # 只需指定实验名称，自动发现所有 steps
#   ./eval_single_model_steps.sh --exp_name ttrl_Qwen2.5-Math-1.5B_AIME24_bsz8_epoch80
#
#   # 手动指定 ckpts_dir 和 steps
#   ./eval_single_model_steps.sh --exp_name ttrl_Qwen3-4B-Base_bsz128 \
#       --ckpts_dir /path/to/ckpts --steps "20 40 60 80"
#
# =============================================================================

set -euo pipefail
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# =============================================================================
# 参数解析
# =============================================================================

EXP_NAME="${SB_EXP_NAME:-data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1_skillrl}"
CKPTS_DIR="${SB_CKPTS_DIR:-/home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b/skillrl_grpo_qwen3_4b}"
STEPS="${SB_STEPS:-}"
BASB_MODEL_NAME="${SB_MODEL_NAME:-Qwen3-4B-Instruct-2507}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.1}"
SKIP_BASE_MODEL="${SKIP_BASE_MODEL:-false}"
TEMP_DATA_FILE="${TEMP_DATA_FILE:-/home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b/temp_data_skill.parquet}"
GREEDY_DATA_FILE="${GREEDY_DATA_FILE:-/home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b/greedy_data_skill.parquet}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --ckpts_dir)
            CKPTS_DIR="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --base_model_name)
            BASB_MODEL_NAME="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --sample_ratio)
            SAMPLE_RATIO="$2"
            shift 2
            ;;
        --skip_base_model)
            SKIP_BASE_MODEL="true"
            shift
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
            shift
            ;;
    esac
done

# =============================================================================
# 参数验证
# =============================================================================

if [ -z "$EXP_NAME" ]; then
    echo "Error: EXP_NAME 为空，请在脚本内设置或通过 SB_EXP_NAME/--exp_name 传入"
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

# =============================================================================
# 自动从 EXP_NAME 提取 BASE_MODEL_NAME
# =============================================================================
# 格式: ttrl_{model_name}_{dataset}_{bsz}_{epoch}
# 例如: ttrl_Qwen2.5-Math-1.5B_AIME24_bsz8_epoch80 -> Qwen2.5-Math-1.5B
#       ttrl_Qwen3-4B_AIME24_bsz8_epoch80 -> Qwen3-4B

if [ -z "$BASB_MODEL_NAME" ]; then
    # 用 _ 分割，取第 2 个字段（模型名）
    # ttrl_Qwen2.5-Math-1.5B_AIME24_bsz8_epoch80 -> Qwen2.5-Math-1.5B
    BASB_MODEL_NAME=$(echo "$EXP_NAME" | cut -d'_' -f2)
    
    if [ -z "$BASB_MODEL_NAME" ]; then
        echo "Error: 无法从 EXP_NAME 提取模型名，请使用 --base_model_name 手动指定"
        exit 1
    fi
    
    echo "自动提取 BASE_MODEL_NAME: $BASB_MODEL_NAME"
fi

# =============================================================================
# 路径配置
# =============================================================================

project_name="${SB_PROJECT_NAME:-Skill_Evo}"
dir="${SB_BASB_DIR:-/home/ycy/sdi}"
model_dir="${SB_MODEL_DIR:-${dir}/models}"
data_dir="${SB_DATA_DIR:-${dir}/data}"
saved_results_dir="${SB_SAVED_RESULTS_DIR:-/home/ycy/sdi/skill_saved/evaluation}"
WORKING_DIR="${SB_WORKING_DIR:-${dir}/${project_name}}"
save_path_dir=${saved_results_dir}/evaluation
eval_path=${WORKING_DIR}/evaluation
tb_path_dir=${saved_results_dir}/eval_tb_log
eval_model_dir=${saved_results_dir}/Solver_ttrl_Base
CUSTOM_EVAL_DATA_DIR="${WORKING_DIR}/evaluation/.eval_custom_data/${EXP_NAME}"

# 如果用户没有指定 CKPTS_DIR，则根据 EXP_NAME 自动计算
if [ -z "$CKPTS_DIR" ]; then
    CKPTS_DIR=${eval_model_dir}/${EXP_NAME}/ckpts
    echo "自动计算 CKPTS_DIR: $CKPTS_DIR"
fi

# 验证 checkpoint 目录是否存在
if [ ! -d "$CKPTS_DIR" ]; then
    echo "Error: Checkpoint 目录不存在: $CKPTS_DIR"
    exit 1
fi

# =============================================================================
# 自动发现 Steps（必须在 CKPTS_DIR 确定之后）
# =============================================================================

if [ -z "$STEPS" ]; then
    echo "自动发现 checkpoint steps..."

    # 查找所有 global_step_* 目录；避免在 set -euo pipefail 下因无匹配直接退出
    shopt -s nullglob
    step_dirs=("${CKPTS_DIR}"/global_step_*)
    shopt -u nullglob

    if [ ${#step_dirs[@]} -eq 0 ]; then
        echo "Error: 没有找到任何 global_step_* 目录在 $CKPTS_DIR"
        exit 1
    fi

    STEPS=$(printf '%s\n' "${step_dirs[@]}" | sed 's/.*global_step_//' | sort -n | tr '\n' ' ')
    echo "发现的 steps: $STEPS"
fi

# 评测结果保存目录
eval_saved_path_dir=${save_path_dir}/${EXP_NAME}_temperature${TEMPERATURE}

mkdir -p "${eval_saved_path_dir}" "${tb_path_dir}" "${WORKING_DIR}/eval_logs"
mkdir -p "${CUSTOM_EVAL_DATA_DIR}"

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

for column in ("extra_info", "reward_model"):
    if column in df.columns:
        df[column] = df[column].map(maybe_parse_json)

tmp_file = f"{dst_file}.tmp"
df.to_parquet(tmp_file)
os.replace(tmp_file, dst_file)
print(f"规范化评测数据: {src_file} -> {dst_file}")
PY
}

normalize_eval_parquet "${TEMP_DATA_FILE}" "${CUSTOM_EVAL_DATA_DIR}/temp_data.parquet"
normalize_eval_parquet "${GREEDY_DATA_FILE}" "${CUSTOM_EVAL_DATA_DIR}/greedy_data.parquet"

cd "${eval_path}"

# =============================================================================
# 日志设置
# =============================================================================

function now() {
    date '+%Y-%m-%d-%H-%M'
}

exec > >(tee -a "${WORKING_DIR}/eval_logs/eval_steps_${EXP_NAME}-$(now).log") 2>&1

echo "=============================================="
echo "  Single Model Multi-Step Evaluation"
echo "=============================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "配置:"
echo "  实验名称:      $EXP_NAME"
echo "  Checkpoint 目录: $CKPTS_DIR"
echo "  评测 Steps:    $STEPS"
echo "  基础模型:      $BASB_MODEL_NAME"
echo "  跳过基础模型:  $SKIP_BASE_MODEL"
echo "  温度:          $TEMPERATURE"
echo "  采样比例:      $SAMPLE_RATIO"
echo "  temp_data 文件: $TEMP_DATA_FILE"
echo "  greedy_data 文件: $GREEDY_DATA_FILE"
echo "  评测数据目录:  $CUSTOM_EVAL_DATA_DIR"
echo "  结果保存目录:  $eval_saved_path_dir"
echo "=============================================="
echo ""

# =============================================================================
# 任务定义
# =============================================================================

TASKS=(
    "temp_data"
    "greedy_data"
)

# ADDITIONAL_EVAL_DATASETS=(
#     "eval_bbeh_step.py"
#     "eval_mmlupro_step.py"
#     "eval_supergpqa_step.py"
# )

# =============================================================================
# GPU 初始化
# =============================================================================

GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))

if [ ${#GPU_QUEUE[@]} -eq 0 ]; then
    echo "Error: No GPUs detected."
    exit 1
fi

echo "Available GPUs: ${GPU_QUEUE[@]} (Total: ${#GPU_QUEUE[@]})"

# =============================================================================
# 全局跟踪变量
# =============================================================================

declare -A gpu_status
declare -A pids
declare -A model_gpu_mapping
declare -A task_completed
declare -A gpu_dataset

for gpu_id in "${GPU_QUEUE[@]}"; do
    gpu_status["$gpu_id"]="idle"
done

# =============================================================================
# 构建模型列表 (单个模型的多个 steps)
# =============================================================================

model_list=()
model_paths=()

# Base model 评测结果保存在 step_0 目录（只评测一次，作为 step 0）

base_model_eval_results_dir=${eval_saved_path_dir}/step_0
base_model_path="${model_dir}/${BASB_MODEL_NAME}"
BASE_MODEL_NEEDS_EVAL="false"

# 检查是否需要评测基础模型
if [ "$SKIP_BASE_MODEL" != "true" ]; then
    if [ ! -d "${base_model_path}" ]; then
        echo "Warning: 基础模型不存在: ${base_model_path}，跳过..."
    elif [ -d "${base_model_eval_results_dir}" ]; then
        # 检查是否有完整的评测结果
        if [ -f "${base_model_eval_results_dir}/aggregated_eval_results.json" ]; then
            echo "基础模型 ${BASB_MODEL_NAME} 评测结果已存在: ${base_model_eval_results_dir}"
        else
            echo "基础模型 ${BASB_MODEL_NAME} 评测结果不完整，需要重新评测"
            BASE_MODEL_NEEDS_EVAL="true"
            model_list+=("${BASB_MODEL_NAME}")
            model_paths+=("${base_model_path}")
        fi
    else
        echo "基础模型 ${BASB_MODEL_NAME} 需要评测，结果将保存到: ${base_model_eval_results_dir}"
        BASE_MODEL_NEEDS_EVAL="true"
    model_list+=("${BASB_MODEL_NAME}")
        model_paths+=("${base_model_path}")
    fi
else
    echo "跳过基础模型评测 (--skip_base_model)"
fi

# 添加各个 step 的 checkpoint
valid_steps=0
invalid_steps=0

for step in $STEPS; do
    step_name="${EXP_NAME}-step${step}"
    step_path="${CKPTS_DIR}/global_step_${step}/actor/huggingface"
    
    if [ -d "$step_path" ]; then
        # 验证 checkpoint 完整性（检查是否有 HF 推理权重）
        if ls "${step_path}"/*.safetensors >/dev/null 2>&1 || \
           ls "${step_path}"/*.bin >/dev/null 2>&1 || \
           [ -f "${step_path}/model.safetensors.index.json" ] || \
           [ -f "${step_path}/pytorch_model.bin.index.json" ]; then
            model_list+=("$step_name")
            model_paths+=("$step_path")
            echo "✓ 添加 checkpoint: $step_name -> $step_path"
            valid_steps=$((valid_steps + 1))
        elif ls "${CKPTS_DIR}/global_step_${step}/actor"/model_world_size_*_rank_*.pt >/dev/null 2>&1; then
            echo "✗ Warning: 检测到 FSDP shard checkpoint（model_world_size_*_rank_*.pt），但未检测到 HF 权重: $step_path"
            echo "           请先将 global_step_${step}/actor 合并导出为 HuggingFace 权重，再执行评测。"
            invalid_steps=$((invalid_steps + 1))
        else
            echo "✗ Warning: Checkpoint 不完整 (无模型文件): $step_path"
            invalid_steps=$((invalid_steps + 1))
        fi
    else
        echo "✗ Warning: Checkpoint 目录不存在: $step_path"
        invalid_steps=$((invalid_steps + 1))
    fi
done

echo ""
echo "=============================================="
echo "模型统计:"
echo "  有效 checkpoints: $valid_steps"
echo "  无效/跳过: $invalid_steps"
echo "  总模型数量: ${#model_list[@]}"
if [ ${#model_list[@]} -gt 0 ]; then
    echo ""
    echo "最终模型列表:"
    for i in "${!model_list[@]}"; do
        echo "  [$((i+1))] ${model_list[$i]}"
    done
fi
echo "=============================================="
echo ""

if [ ${#model_list[@]} -eq 0 ]; then
    echo "Error: 没有找到任何可评测的模型"
    exit 1
fi

# =============================================================================
# 辅助函数
# =============================================================================

get_available_gpus() {
    local available_gpus=()
    for gpu_id in "${GPU_QUEUE[@]}"; do
        if [ "${gpu_status[$gpu_id]}" = "idle" ]; then
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            if [ "$memory_percent" -lt 5 ] && [ "$gpu_util" -lt 5 ]; then
                available_gpus+=("$gpu_id")
            fi
        fi
    done
    echo "${available_gpus[@]}"
}

check_dataset_completed() {
    local model_name="$1"
    local eval_script="$2"
    local dataset_name=$(basename "$eval_script" .py | sed 's/^eval_//' | sed 's/_step$//')
    
    # 从 model_name 提取 step（格式：xxx-stepN）
    local step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
    
    local result_file
    if [ -n "$step" ]; then
        # 新目录结构: step_N/{dataset}_final_results.json
        result_file="${eval_saved_path_dir}/step_${step}/${dataset_name}_final_results.json"
    elif [ "$model_name" == "$BASB_MODEL_NAME" ]; then
        # Base model 统一保存在 step_0 目录
        result_file="${base_model_eval_results_dir}/${dataset_name}_final_results.json"
    else
        # 兼容旧目录结构
        result_file="${eval_saved_path_dir}/${model_name}/${dataset_name}_final_results.json"
    fi
    
    if [ -f "$result_file" ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Dataset [${dataset_name}] for model [${model_name}] already completed"
        return 0
    else
        return 1
    fi
}

check_main_task_completed() {
    local model_name="$1"
    local dataset="$2"
    
    # 从 model_name 提取 step（格式：xxx-stepN）
    local step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
    
    local result_file
    if [ -n "$step" ]; then
        # 新目录结构: step_N/{dataset}_responses.parquet
        result_file="${eval_saved_path_dir}/step_${step}/${dataset}_responses.parquet"
    elif [ "$model_name" == "$BASB_MODEL_NAME" ]; then
        # Base model 统一保存在 step_0 目录
        result_file="${base_model_eval_results_dir}/${dataset}_responses.parquet"
    else
        # 兼容旧目录结构
        result_file="${eval_saved_path_dir}/${model_name}/${dataset}_Overall_results.jsonl"
    fi
    
    if [ -f "$result_file" ]; then
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Main task [${dataset}] for model [${model_name}] already completed"
        return 0
    else
        return 1
    fi
}

start_additional_eval_job() {
    local gpu_id="$1"
    local model_path="$2"
    local model_name="$3"
    local eval_script="$4"
    
    # 从 model_name 提取 step（格式：xxx-stepN）
    local step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
    local step_arg=""
    local target_save_dir="${eval_saved_path_dir}"
    
    if [ -n "$step" ]; then
        step_arg="--step ${step}"
    elif [ "$model_name" == "$BASB_MODEL_NAME" ]; then
        # Base model 也按 step 目录组织，固定保存到 step_0
        step_arg="--step 0"
    fi
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation [${eval_script}] for model [${model_name}] (step=${step:-base}) on GPU [${gpu_id}]"
    
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${eval_script}" \
        --model_path "${model_path}" \
        --model_name "${model_name}" \
        --save_path_dir "${target_save_dir}" \
        --data_path_dir "${data_dir}" \
        --sample_ratio "${SAMPLE_RATIO}" \
        ${step_arg} &
    local pid=$!
    
    sleep 10
    
    if kill -0 "$pid" 2>/dev/null; then
        gpu_status["${gpu_id}"]="busy"
        pids["${gpu_id}"]="$pid"
        model_gpu_mapping["${gpu_id}"]="${model_name}"
        gpu_dataset["${gpu_id}"]="${eval_script}"
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started evaluation [${eval_script}] for model [${model_name}]"
        return 0
    else
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Error: Failed to start evaluation [${eval_script}] for model [${model_name}]"
        kill -TERM "$pid" 2>/dev/null || true
        return 1
    fi
}

start_math_task_job() {
    local gpu_id="$1"
    local model_path="$2"
    local model_name="$3"
    local dataset="$4"
    
    # 从 model_name 提取 step（格式：xxx-stepN）
    local step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
    local step_arg=""
    local target_save_dir="${eval_saved_path_dir}"
    
    if [ -n "$step" ]; then
        step_arg="--step ${step}"
    elif [ "$model_name" == "$BASB_MODEL_NAME" ]; then
        # Base model 也按 step 目录组织，固定保存到 step_0
        step_arg="--step 0"
    fi
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting math task [${dataset}] for model [${model_name}] (step=${step:-base}) on GPU [${gpu_id}]"
    
    local n_samples=32
    local temp=$TEMPERATURE
    if [ "$dataset" == "greedy_data" ]; then
        n_samples=1
        temp=0.0
    fi
    
    CUDA_VISIBLE_DEVICES="${gpu_id}" python eval_all_math_step.py \
        --model_path "${model_path}" \
        --model_name "${model_name}" \
        --dataset "${dataset}" \
        --save_path_dir "${target_save_dir}" \
        --n_samples "${n_samples}" \
        --temperature "${temp}" \
        --data_path_dir "${CUSTOM_EVAL_DATA_DIR}" \
        ${step_arg} &
    local pid=$!
    
    sleep 10
    
    if kill -0 "$pid" 2>/dev/null; then
        gpu_status["${gpu_id}"]="busy"
        pids["${gpu_id}"]="$pid"
        model_gpu_mapping["${gpu_id}"]="${model_name}"
        gpu_dataset["${gpu_id}"]="${dataset}"
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Successfully started math task [${dataset}] for model [${model_name}]"
        return 0
    else
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Error: Failed to start math task [${dataset}] for model [${model_name}]"
        kill -TERM "$pid" 2>/dev/null || true
        return 1
    fi
}

free_gpu() {
    local gpu_id="$1"
    local model_name="${model_gpu_mapping[$gpu_id]:-unknown}"
    
    unset pids["$gpu_id"]
    gpu_status["$gpu_id"]="idle"
    unset model_gpu_mapping["$gpu_id"]
    unset task_completed["$gpu_id"]
    unset gpu_dataset["$gpu_id"]
    
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Freed GPU [${gpu_id}] (was running model [${model_name}])"
}

# 清理僵尸进程
cleanup_zombie_processes() {
    local orphaned_pids=()
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            local pid=$(echo "$line" | awk '{print $2}')
            local gpu_used=$(echo "$line" | grep -o 'CUDA_VISIBLE_DEVICES=[0-9]*' | cut -d'=' -f2)
            
            local is_tracked=false
            for tracked_gpu in "${!pids[@]}"; do
                if [ "${pids[$tracked_gpu]}" = "$pid" ]; then
                    is_tracked=true
                    break
                fi
            done
            
            if [ "$is_tracked" = false ] && [ -n "$gpu_used" ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Found orphaned process PID [${pid}] on GPU [${gpu_used}]"
                orphaned_pids+=("$pid")
            fi
        fi
    done < <(ps aux | grep -E "eval_all_math_step.py|eval_bbeh_step.py|eval_mmlupro_step.py|eval_supergpqa_step.py" | grep -v grep)
    
    for pid in "${orphaned_pids[@]}"; do
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Killing orphaned process PID [${pid}]"
        kill -TERM "$pid" 2>/dev/null || true
    done
}

# 综合清理
comprehensive_cleanup() {
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Performing comprehensive GPU state cleanup..."
    
    for gpu_id in "${GPU_QUEUE[@]}"; do
        if nvidia-smi -i "$gpu_id" >/dev/null 2>&1; then
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            if [ "${gpu_status[$gpu_id]}" = "busy" ] && [ "$memory_percent" -lt 5 ] && [ "$gpu_util" -lt 5 ]; then
                local model_name="${model_gpu_mapping[$gpu_id]}"
                local pid="${pids[$gpu_id]}"
                
                if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] process finished for model [${model_name}]"
                else
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] marked as busy but has low usage (${memory_percent}%), forcing cleanup..."
                    free_gpu "$gpu_id"
                fi
            fi
        fi
    done
    
    cleanup_zombie_processes
}

check_completed_jobs() {
    # 第一阶段：检测已完成的任务
    for gpu_id in "${!pids[@]}"; do
        pid="${pids[$gpu_id]}"
        
        if ! kill -0 "$pid" 2>/dev/null; then
            local model_name="${model_gpu_mapping[$gpu_id]}"
            
            if [ -z "${task_completed[$gpu_id]:-}" ]; then
                local dataset="${gpu_dataset[$gpu_id]}"
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [DONE] GPU [${gpu_id}] finished: [${dataset}] for [${model_name}] (PID ${pid})"
                task_completed["$gpu_id"]=1
                
                if [[ " ${TASKS[@]} " =~ " ${dataset} " ]]; then
                    local n_samples=32
                    local temp=$TEMPERATURE
                    if [ "$dataset" == "greedy_data" ]; then
                        n_samples=1
                        temp=0.0
                    fi
                    
                    # 从 model_name 提取 step（格式：xxx-stepN）
                    local step=$(echo "$model_name" | sed -n 's/.*-step\([0-9]*\)$/\1/p')
                    local step_arg=""
                    local target_save_dir="${eval_saved_path_dir}"
                    
                    if [ -n "$step" ]; then
                        step_arg="--step ${step}"
                    elif [ "$model_name" == "$BASB_MODEL_NAME" ]; then
                        # Base model 也按 step 目录组织，固定保存到 step_0
                        step_arg="--step 0"
                    fi
                    
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Running post_eval_step for [${model_name}] (step=${step:-base}) on dataset [${dataset}]..."
                    python post_eval_step.py \
                        --save_path_dir "${target_save_dir}" \
                        --dataset "${dataset}" \
                        --model_name "${model_name}" \
                        --n_samples "${n_samples}" \
                        --temperature "${temp}" \
                        ${step_arg}
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [POST] Post-processing completed for [${model_name}] on [${dataset}]"
                else
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [DONE] Additional eval [${dataset}] completed for [${model_name}], no post-processing needed"
                fi
                
                free_gpu "$gpu_id"
            fi
        fi
    done
    
    # 第二阶段：检测卡住的进程
    for gpu_id in "${!pids[@]}"; do
        pid="${pids[$gpu_id]}"
        
        if kill -0 "$pid" 2>/dev/null; then
            local model_name="${model_gpu_mapping[$gpu_id]}"
            
            # 检查 GPU 资源使用情况
            local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local memory_percent=$((memory_used * 100 / memory_total))
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            
            # 如果 GPU 内存和利用率都很低，可能是卡住了
            if [ "$memory_percent" -lt 2 ] && [ "$memory_used" -lt 100 ] && [ "$gpu_util" -lt 5 ]; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Warning: GPU [${gpu_id}] has very low memory usage (${memory_percent}%) and utilization (${gpu_util}%) for model [${model_name}]"
                
                # 检查 CPU 使用率来确认是否卡住
                local cpu_usage=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
                if [ "$cpu_usage" = "0.0" ] || [ -z "$cpu_usage" ]; then
                    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Detected stuck process PID [${pid}] on GPU [${gpu_id}], terminating..."
                    kill -TERM "$pid" 2>/dev/null || true
                    sleep 2
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -KILL "$pid" 2>/dev/null || true
                    fi
                    free_gpu "$gpu_id"
                fi
            fi
        fi
    done
}

# =============================================================================
# 主执行循环 - 额外评测数据集（已禁用，仅保留数学评测）
# =============================================================================

# echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting additional evaluation datasets"
#
# declare -a task_queue=()
#
# for i in "${!model_list[@]}"; do
#     model_name="${model_list[$i]}"
#     model_path="${model_paths[$i]}"
#
#     if [ ! -d "$model_path" ]; then
#         echo "Warning: Model path does not exist: $model_path, skipping..."
#         continue
#     fi
#
#     for eval_script in "${ADDITIONAL_EVAL_DATASETS[@]}"; do
#         if check_dataset_completed "$model_name" "$eval_script"; then
#             continue
#         fi
#         task_queue+=("${model_name}|${model_path}|${eval_script}")
#     done
# done
#
# echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Additional eval task queue: ${#task_queue[@]} tasks"
# echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Task queue details:"
# for i in "${!task_queue[@]}"; do
#     echo "  [$i] ${task_queue[$i]}"
# done
#
# # 如果没有任务，跳过此循环
# if [ ${#task_queue[@]} -eq 0 ]; then
#     echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] No additional eval tasks to run"
# else
#
# cleanup_counter=0
# task_index=0
#
# while [ $task_index -lt ${#task_queue[@]} ] || [ ${#pids[@]} -gt 0 ]; do
#     # 定期综合清理
#     cleanup_counter=$((cleanup_counter + 1))
#     if [ $((cleanup_counter % 10)) -eq 0 ]; then
#         comprehensive_cleanup
#     else
#         cleanup_zombie_processes
#     fi
#
#     check_completed_jobs
#     available_gpus=($(get_available_gpus))
#
#     while [ $task_index -lt ${#task_queue[@]} ] && [ ${#available_gpus[@]} -ge 1 ]; do
#         task="${task_queue[$task_index]}"
#         IFS='|' read -r model_name model_path eval_script <<< "$task"
#
#         if check_dataset_completed "$model_name" "$eval_script"; then
#             echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] Task $((task_index + 1))/${#task_queue[@]}: [${eval_script}] for [${model_name}] already completed"
#             task_index=$((task_index + 1))
#             continue
#         fi
#
#         if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]]; then
#             if start_additional_eval_job "${available_gpus[0]}" "$model_path" "$model_name" "$eval_script"; then
#                 echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [START] Task $((task_index + 1))/${#task_queue[@]}: [${eval_script}] for [${model_name}] on GPU ${available_gpus[0]}"
#                 task_index=$((task_index + 1))
#                 available_gpus=($(get_available_gpus))
#             else
#                 echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [RETRY] Task $((task_index + 1))/${#task_queue[@]}: [${eval_script}] for [${model_name}] failed to start, will retry"
#                 break
#             fi
#         else
#             echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [WARN] Invalid GPU ID, skipping task $((task_index + 1))/${#task_queue[@]}"
#             task_index=$((task_index + 1))
#         fi
#     done
#
#     if [ ${#pids[@]} -gt 0 ] || [ $task_index -lt ${#task_queue[@]} ]; then
#         completed=$((task_index - ${#pids[@]}))
#         total_tasks=${#task_queue[@]}
#         progress=0
#         if [ $total_tasks -gt 0 ]; then
#             progress=$((completed * 100 / total_tasks))
#         fi
#         echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [STATUS] Additional eval: Running ${#pids[@]} jobs, Completed ${completed}/${total_tasks} (${progress}%), Pending $((total_tasks - task_index))"
#         sleep 30
#     fi
# done
#
# fi  # 结束 task_queue 非空检查
#
# echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [COMPLETE] All additional evaluations completed! (${#task_queue[@]} tasks)"

# =============================================================================
# 重置状态
# =============================================================================

for gpu_id in "${GPU_QUEUE[@]}"; do
    gpu_status["$gpu_id"]="idle"
done
pids=()
model_gpu_mapping=()
task_completed=()
gpu_dataset=()

# =============================================================================
# 主执行循环 - 数学任务
# =============================================================================

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Starting math tasks"

declare -a math_task_queue=()

for i in "${!model_list[@]}"; do
    model_name="${model_list[$i]}"
    model_path="${model_paths[$i]}"
    
    if [ ! -d "$model_path" ]; then
        continue
    fi
    
    for dataset in "${TASKS[@]}"; do
        if check_main_task_completed "$model_name" "$dataset"; then
            continue
        fi
        math_task_queue+=("${model_name}|${model_path}|${dataset}")
    done
done

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Math task queue: ${#math_task_queue[@]} tasks"
echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Task queue details:"
for i in "${!math_task_queue[@]}"; do
    echo "  [$i] ${math_task_queue[$i]}"
done

# 如果没有任务，跳过此循环
if [ ${#math_task_queue[@]} -eq 0 ]; then
    echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] No math tasks to run"
else

cleanup_counter=0
math_task_index=0

while [ $math_task_index -lt ${#math_task_queue[@]} ] || [ ${#pids[@]} -gt 0 ]; do
    # 定期综合清理
    cleanup_counter=$((cleanup_counter + 1))
    if [ $((cleanup_counter % 10)) -eq 0 ]; then
        comprehensive_cleanup
    else
        cleanup_zombie_processes
    fi
    
    check_completed_jobs
    available_gpus=($(get_available_gpus))
    
    while [ $math_task_index -lt ${#math_task_queue[@]} ] && [ ${#available_gpus[@]} -ge 1 ]; do
        task="${math_task_queue[$math_task_index]}"
        IFS='|' read -r model_name model_path dataset <<< "$task"
        
        if check_main_task_completed "$model_name" "$dataset"; then
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] Math task $((math_task_index + 1))/${#math_task_queue[@]}: [${dataset}] for [${model_name}] already completed"
            math_task_index=$((math_task_index + 1))
            continue
        fi
        
        if [[ "${available_gpus[0]}" =~ ^[0-9]+$ ]]; then
            if start_math_task_job "${available_gpus[0]}" "$model_path" "$model_name" "$dataset"; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [START] Math task $((math_task_index + 1))/${#math_task_queue[@]}: [${dataset}] for [${model_name}] on GPU ${available_gpus[0]}"
                math_task_index=$((math_task_index + 1))
                available_gpus=($(get_available_gpus))
            else
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [RETRY] Math task $((math_task_index + 1))/${#math_task_queue[@]}: [${dataset}] for [${model_name}] failed to start, will retry"
                break
            fi
        else
            echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [WARN] Invalid GPU ID, skipping math task $((math_task_index + 1))/${#math_task_queue[@]}"
            math_task_index=$((math_task_index + 1))
        fi
    done
    
    if [ ${#pids[@]} -gt 0 ] || [ $math_task_index -lt ${#math_task_queue[@]} ]; then
        completed=$((math_task_index - ${#pids[@]}))
        total_tasks=${#math_task_queue[@]}
        progress=0
        if [ $total_tasks -gt 0 ]; then
            progress=$((completed * 100 / total_tasks))
        fi
        echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [STATUS] Math tasks: Running ${#pids[@]} jobs, Completed ${completed}/${total_tasks} (${progress}%), Pending $((total_tasks - math_task_index))"
        sleep 30
    fi
done

fi  # 结束 math_task_queue 非空检查

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] [COMPLETE] All math tasks completed! (${#math_task_queue[@]} tasks)"

# =============================================================================
# 聚合结果
# =============================================================================

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Aggregating evaluation results..."

# 使用新的聚合脚本
python aggregate_eval_results_step.py \
    --save_path_dir "${eval_saved_path_dir}"

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Uploading to TensorBoard..."

# 使用新的 TensorBoard 脚本
python tb_step.py \
    --exp_name="${EXP_NAME}" \
    --temperature="${TEMPERATURE}" \
    --save_path_dir="${eval_saved_path_dir}" \
    --tb_path_dir="${tb_path_dir}" \
    --generate_table

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] All evaluations completed successfully!"
echo ""
echo "=============================================="
echo "评测完成！"
echo "  结果目录: ${eval_saved_path_dir}"
echo "  TensorBoard: ${tb_path_dir}/${EXP_NAME}-temperature_${TEMPERATURE}"
if [ "$SKIP_BASE_MODEL" != "true" ] && [ -d "${base_model_eval_results_dir}" ]; then
    echo "  Base model (step 0): ${base_model_eval_results_dir}"
fi
echo "=============================================="
