#!/usr/bin/env bash
# =============================================================================
# main.sh - Skill Synthesizer 训练核心调度器
# wandb: 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
# =============================================================================
# 由 run_with_gpus.sh 调用。所有可调参数在此配置，然后调用 Synthesizer.sh 执行。
#
# 依赖环境变量（由 run_with_gpus.sh 导出）:
#   SE_N_GPUS / SE_GPU_IDS / SE_SYNTHESIZER_GPUS / SE_N_SYNTHESIZER_GPUS
#   SE_ROLLOUT_BASE_PORT / SE_Synthsizer_DIR / SE_ROLLOUT_DIR
#   SE_MODEL_DIR / SE_DATA_DIR / SE_SAVED_RESULTS_DIR / SE_WORKING_DIR 等
# =============================================================================

set -xeuo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Ray 会话目录默认在 $SE_RAY_TEMP_ROOT/r-$USER（main_synthesizer）；避免用满 /tmp
export SE_RAY_TEMP_ROOT="${SE_RAY_TEMP_ROOT:-/home/ycy/sdi/tmp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# 模型配置
# =============================================================================
base_model_name="${SE_BASE_MODEL_NAME:-Qwen3-4B-Instruct-2507}"
base_model_path="${SE_MODEL_DIR}/${base_model_name}"

# Synthesizer 模型（skill 蒸馏 RL 训练）
synthesizer_model_name="${SE_SYNTHESIZER_MODEL_NAME:-${base_model_name}}"
synthesizer_model_path="${SE_MODEL_DIR}/${synthesizer_model_name}"

# Solver 模型（用于 rollout server：离线 rollout + reward 评估）
solver_model_name="${SE_SOLVER_MODEL_NAME:-${base_model_name}}"
solver_model_path="${SE_MODEL_DIR}/${solver_model_name}"

# =============================================================================
# 数据配置
# =============================================================================
data_name="${SE_DATA_NAME:-DeepMath-103K}"
data_file="${SE_DATA_DIR}/${data_name}.jsonl"

# =============================================================================
# 实验命名
# =============================================================================
variant="data_${data_name}_model_${base_model_name}"
exp_name="${variant}-V1"

# =============================================================================
# Offline Rollout 参数
# =============================================================================
export SE_OFFLINE_ROLLOUT_STEPS="${SE_OFFLINE_ROLLOUT_STEPS:-4}"
export SE_OFFLINE_ROLLOUT_BATCH_SIZE="${SE_OFFLINE_ROLLOUT_BATCH_SIZE:-16}"
export SE_OFFLINE_ROLLOUT_N="${SE_OFFLINE_ROLLOUT_N:-4}"
export SE_OFFLINE_NUM_RANDOM_Q="${SE_OFFLINE_NUM_RANDOM_Q:-4}"
export SE_OFFLINE_SKILL_TYPE="${SE_OFFLINE_SKILL_TYPE:-skill_generation_v1}"

# =============================================================================
# Synthesizer RL 训练超参数
# =============================================================================
synthesizer_training_steps="${SE_SYNTHESIZER_TRAINING_STEPS:-4}"

export SYNTH_BATCH_SIZE="${SYNTH_BATCH_SIZE:-16}"
export SYNTH_ROLLOUT_QUERY_NUM="${SYNTH_ROLLOUT_QUERY_NUM:-4}"
export SYNTH_QUERY_TOP_P="${SYNTH_QUERY_TOP_P:-0.99}"
export SYNTH_QUERY_TOP_K="${SYNTH_QUERY_TOP_K:--1}"
export SYNTH_KL_LOSS_COEF="${SYNTH_KL_LOSS_COEF:-0.01}"
export SYNTH_QUERY_TEMPERATURE="${SYNTH_QUERY_TEMPERATURE:-1.0}"
export SYNTH_TP="${SYNTH_TP:-1}"
export SYNTH_MAX_PROMPT_LENGTH="${SYNTH_MAX_PROMPT_LENGTH:-8192}"
export SYNTH_MAX_RESPONSE_LENGTH="${SYNTH_MAX_RESPONSE_LENGTH:-4096}"
export SYNTH_GPU_MEM_UTIL="${SYNTH_GPU_MEM_UTIL:-0.60}"
export SYNTH_RANDOM_Q_COEF="${SYNTH_RANDOM_Q_COEF:-0.5}"
export SYNTH_USE_SKILL_TYPE="${SYNTH_USE_SKILL_TYPE:-skill_use_v1}"

# =============================================================================
# 日志
# =============================================================================
WORKING_DIR="${SE_WORKING_DIR}"
mkdir -p "${WORKING_DIR}/logs"

function now() { date '+%Y-%m-%d-%H-%M'; }
exec > >(tee -a "${WORKING_DIR}/logs/train_synth_${variant}-$(now).log") 2>&1

# =============================================================================
# 打印配置总览
# =============================================================================
echo "=============================================="
echo "  main.sh - 参数配置总览"
echo "=============================================="
echo "  exp_name:             ${exp_name}"
echo "  synthesizer_model:    ${synthesizer_model_path}"
echo "  solver_model:         ${solver_model_path}"
echo "  data_file:            ${data_file}"
echo "  training_steps:       ${synthesizer_training_steps}"
echo ""
echo "  offline rollout:"
echo "    steps=${SE_OFFLINE_ROLLOUT_STEPS}  batch=${SE_OFFLINE_ROLLOUT_BATCH_SIZE}"
echo "    rollout_n=${SE_OFFLINE_ROLLOUT_N}  random_q=${SE_OFFLINE_NUM_RANDOM_Q}"
echo "    skill_type=${SE_OFFLINE_SKILL_TYPE}"
echo ""
echo "  RL hyper-params:"
echo "    batch=${SYNTH_BATCH_SIZE}  rollout_n=${SYNTH_ROLLOUT_QUERY_NUM}"
echo "     kl=${SYNTH_KL_LOSS_COEF}  temp=${SYNTH_QUERY_TEMPERATURE}"
echo "    max_prompt=${SYNTH_MAX_PROMPT_LENGTH}  max_resp=${SYNTH_MAX_RESPONSE_LENGTH}"
echo "=============================================="

# =============================================================================
# 调用 Synthesizer.sh
# =============================================================================
bash "${SCRIPT_DIR}/Synthesizer.sh" \
    "${exp_name}" \
    "${synthesizer_model_path}" \
    "${solver_model_path}" \
    "${synthesizer_training_steps}" \
    "${data_file}" || {
    echo "Error: Synthesizer 训练失败"
    exit 1
}

echo "main.sh 完成！"
