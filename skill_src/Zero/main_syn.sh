#!/usr/bin/env bash
# =============================================================================
# main.sh - Skill Synthesizer 训练核心调度器
# wandb: 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
# =============================================================================
# 由 run_with_gpus.sh 调用。所有可调参数在此配置，然后调用 Synthesizer 执行脚本。
#
# SE_SYNTH_USE_VLLM_HTTP：未设置时默认 Synthesizer_vllm_http.sh（全卡 vLLM 离线 rollout）。
#   设为 0 / false / no 时使用经典 Synthesizer.sh（仅后半卡 rollout + 进程内 vLLM）。
#   设为 1 / true / yes 时显式使用 Synthesizer_vllm_http.sh。
#
# 依赖环境变量（由 run_with_gpus.sh 导出）:
#   SE_N_GPUS / SE_GPU_IDS / SE_SYNTHESIZER_GPUS / SE_N_SYNTHESIZER_GPUS
#   SE_ROLLOUT_BASE_PORT / SE_SKILL_SAVED_ROOT
#   SE_MODEL_DIR / SE_DATA_DIR / SE_WORKING_DIR 等
# 本脚本在 exp_name 确定后设置: SE_SAVED_RESULTS_DIR / SE_Synthsizer_DIR / SE_ROLLOUT_DIR / SE_TENSORBOARD_DIR / SE_SOLVER_DIR
#   SE_EMBEDDING_CACHE_PATH（可选）：embedding cache 目录，传给 Synthesizer.sh 第 6 参；未设则空（离线 rollout 随机采样）
# =============================================================================

set -xeuo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Ray 会话目录默认在 $SE_RAY_TEMP_ROOT/r-$USER（main_synthesizer）；避免用满 /tmp
export SE_RAY_TEMP_ROOT="${SE_RAY_TEMP_ROOT:-/home/ycy/sdi/tmp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SE_SYNTH_USE_VLLM_HTTP="${SE_SYNTH_USE_VLLM_HTTP:-1}"

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
embedding_cache_path="${SE_EMBEDDING_CACHE_PATH}"

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
# 实验产物路径（skill_saved/<exp_name>/Synthesizer|Solver）
# =============================================================================
SE_SKILL_SAVED_ROOT="${SE_SKILL_SAVED_ROOT:-/home/ycy/sdi/skill_saved}"
export SE_SAVED_RESULTS_DIR="${SE_SAVED_RESULTS_DIR:-${SE_SKILL_SAVED_ROOT}/${exp_name}}"
export SE_Synthsizer_DIR="${SE_Synthsizer_DIR:-${SE_SAVED_RESULTS_DIR}/Synthesizer}"
export SE_SOLVER_DIR="${SE_SOLVER_DIR:-${SE_SAVED_RESULTS_DIR}/Solver}"
export SE_TENSORBOARD_DIR="${SE_TENSORBOARD_DIR:-${SE_Synthsizer_DIR}/tensorboard_log}"
export SE_ROLLOUT_DIR="${SE_ROLLOUT_DIR:-${SE_Synthsizer_DIR}}"
mkdir -p "${SE_Synthsizer_DIR}" "${SE_SOLVER_DIR}" "${SE_TENSORBOARD_DIR}" "${SE_Synthsizer_DIR}/logs"

# =============================================================================
# Offline Rollout 参数
# =============================================================================
export SE_OFFLINE_ROLLOUT_STEPS="${SE_OFFLINE_ROLLOUT_STEPS:-2}"
export SE_OFFLINE_ROLLOUT_BATCH_SIZE="${SE_OFFLINE_ROLLOUT_BATCH_SIZE:-16}"
export SE_OFFLINE_ROLLOUT_N="${SE_OFFLINE_ROLLOUT_N:-4}"
export SE_OFFLINE_NUM_RANDOM_Q="${SE_OFFLINE_NUM_RANDOM_Q:-4}"
export SE_OFFLINE_SKILL_TYPE="${SE_OFFLINE_SKILL_TYPE:-skill_generation_v1}"

# =============================================================================
# Synthesizer RL 训练超参数
# =============================================================================
synthesizer_training_steps="${SE_SYNTHESIZER_TRAINING_STEPS:-2}"

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
function now() { date '+%Y-%m-%d-%H-%M'; }
exec > >(tee -a "${SE_Synthsizer_DIR}/logs/train_synth_${variant}-$(now).log") 2>&1

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
echo "    embedding_cache_path=${embedding_cache_path:-<empty>}"
echo ""
echo "  RL hyper-params:"
echo "    batch=${SYNTH_BATCH_SIZE}  rollout_n=${SYNTH_ROLLOUT_QUERY_NUM}"
echo "     kl=${SYNTH_KL_LOSS_COEF}  temp=${SYNTH_QUERY_TEMPERATURE}"
echo "    max_prompt=${SYNTH_MAX_PROMPT_LENGTH}  max_resp=${SYNTH_MAX_RESPONSE_LENGTH}"
echo ""
case "${SE_SYNTH_USE_VLLM_HTTP}" in
    0|false|FALSE|no|NO) synth_script="${SCRIPT_DIR}/Synthesizer.sh" ;;
    *) synth_script="${SCRIPT_DIR}/Synthesizer_vllm_http.sh" ;;
esac
echo "  synthesizer_runner:   ${synth_script}"
echo "  prebuilt_rollout:     ${SE_PREBUILT_ROLLOUT_PATH:-<none>}"
echo "=============================================="

# =============================================================================
# 导出可选环境变量（供 Synthesizer_vllm_http.sh 使用）
# =============================================================================
export SE_PREBUILT_ROLLOUT_PATH="${SE_PREBUILT_ROLLOUT_PATH:-}"

# =============================================================================
# 调用 Synthesizer.sh 或 Synthesizer_vllm_http.sh
# =============================================================================
bash "${synth_script}" \
    "${exp_name}" \
    "${synthesizer_model_path}" \
    "${solver_model_path}" \
    "${synthesizer_training_steps}" \
    "${data_file}" \
    "${embedding_cache_path}" || {
    echo "Error: Synthesizer 训练失败"
    exit 1
}

echo "main.sh 完成！"
