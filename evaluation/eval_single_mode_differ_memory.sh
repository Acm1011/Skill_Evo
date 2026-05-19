#!/bin/bash
# =============================================================================
# eval_single_mode_differ_memory.sh - 固定 checkpoint/step，批量评测不同 memory 版本
# =============================================================================
#
# 设计目标:
#   1. 固定同一组 ckpts 和 steps
#   2. 只切换 temp_data / greedy_data 的 memory 版本
#   3. 结果目录必须能区分不同 memory 版本，避免只按 step 覆盖
#
# 典型用法:
#   1. 直接修改脚本顶部变量:
#      - MEMORY_VERSIONS
#      - EXP_NAME_PREFIX
#      - CKPTS_DIR
#      - FIXED_STEPS
#      - DATA_DIR
#   2. 然后执行:
#      bash Skill_Evo/evaluation/eval_single_mode_differ_memory.sh
#
# 默认数据规则:
#   temp_data:   ${DATA_DIR}/temp_data_skill_v${version}.parquet
#   greedy_data: ${DATA_DIR}/greedy_data_skill_v${version}.parquet
#
# 为避免不同 memory 版本的评测结果混在一起，
# 每次调用都会使用:
#   EXP_NAME=${EXP_NAME_PREFIX}_memory_v${version}
#
# 这样 eval_single_mode_steps.sh 最终会保存到:
#   .../evaluation/${EXP_NAME}_temperature${TEMPERATURE}/step_${step}/...
#
# 也支持命令行覆盖:
#   --versions "1 2 3"
#   --steps "20"
#   --ckpts_dir /path/to/ckpts
#   --data_dir /path/to/data_dir
#   --exp_name_prefix my_exp
#   --temp_pattern "temp_data_skill_v%s.parquet"
#   --greedy_pattern "greedy_data_skill_v%s.parquet"
#
# 其它参数会原样透传给 eval_single_mode_steps.sh，例如:
#   --skip_base_model
#   --temperature 0.7
#   --base_model_name Qwen3-4B-Instruct-2507
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/eval_single_mode_steps.sh"

# =============================================================================
# 优先修改这里
# =============================================================================

MEMORY_VERSIONS=(1)
EXP_NAME_PREFIX="data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1_differ_memory"
CKPTS_DIR="/home/yangchengyi/data/skill_saved/Skill_Evo/data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1/Solver/V6/ckpts"
FIXED_STEPS="20"
DATA_DIR="/home/yangchengyi/data/skill_saved/Skill_Evo/data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1"
TEMP_PATTERN="temp_data_skill_v%s.parquet"
GREEDY_PATTERN="greedy_data_skill_v%s.parquet"

# =============================================================================
# 可选参数
# =============================================================================

VERSION_LIST_OVERRIDE=""
STEPS_OVERRIDE=""
CKPTS_DIR_OVERRIDE=""
DATA_DIR_OVERRIDE=""
EXP_NAME_PREFIX_OVERRIDE=""
TEMP_PATTERN_OVERRIDE=""
GREEDY_PATTERN_OVERRIDE=""
declare -a FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --versions)
            VERSION_LIST_OVERRIDE="${2:-}"
            shift 2
            ;;
        --steps)
            STEPS_OVERRIDE="${2:-}"
            shift 2
            ;;
        --ckpts_dir)
            CKPTS_DIR_OVERRIDE="${2:-}"
            shift 2
            ;;
        --data_dir)
            DATA_DIR_OVERRIDE="${2:-}"
            shift 2
            ;;
        --exp_name_prefix)
            EXP_NAME_PREFIX_OVERRIDE="${2:-}"
            shift 2
            ;;
        --temp_pattern)
            TEMP_PATTERN_OVERRIDE="${2:-}"
            shift 2
            ;;
        --greedy_pattern)
            GREEDY_PATTERN_OVERRIDE="${2:-}"
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    echo "Error: target script not found: ${TARGET_SCRIPT}"
    exit 1
fi

declare -a RUN_VERSIONS=()
if [[ -n "${VERSION_LIST_OVERRIDE}" ]]; then
    read -r -a RUN_VERSIONS <<< "${VERSION_LIST_OVERRIDE}"
else
    RUN_VERSIONS=("${MEMORY_VERSIONS[@]}")
fi

RUN_STEPS="${STEPS_OVERRIDE:-${FIXED_STEPS}}"
RUN_CKPTS_DIR="${CKPTS_DIR_OVERRIDE:-${CKPTS_DIR}}"
RUN_DATA_DIR="${DATA_DIR_OVERRIDE:-${DATA_DIR}}"
RUN_EXP_NAME_PREFIX="${EXP_NAME_PREFIX_OVERRIDE:-${EXP_NAME_PREFIX}}"
RUN_TEMP_PATTERN="${TEMP_PATTERN_OVERRIDE:-${TEMP_PATTERN}}"
RUN_GREEDY_PATTERN="${GREEDY_PATTERN_OVERRIDE:-${GREEDY_PATTERN}}"

if [[ ${#RUN_VERSIONS[@]} -eq 0 ]]; then
    echo "Error: no memory versions specified"
    exit 1
fi

if [[ -z "${RUN_STEPS}" ]]; then
    echo "Error: steps is empty"
    exit 1
fi

if [[ ! -d "${RUN_CKPTS_DIR}" ]]; then
    echo "Error: checkpoint directory not found: ${RUN_CKPTS_DIR}"
    exit 1
fi

if [[ ! -d "${RUN_DATA_DIR}" ]]; then
    echo "Error: data directory not found: ${RUN_DATA_DIR}"
    exit 1
fi

echo "=============================================="
echo "  Batch Eval By Memory Version"
echo "=============================================="
echo "Target script: ${TARGET_SCRIPT}"
echo "Memory versions: ${RUN_VERSIONS[*]}"
echo "EXP_NAME_PREFIX: ${RUN_EXP_NAME_PREFIX}"
echo "CKPTS_DIR: ${RUN_CKPTS_DIR}"
echo "STEPS: ${RUN_STEPS}"
echo "DATA_DIR: ${RUN_DATA_DIR}"
echo "TEMP_PATTERN: ${RUN_TEMP_PATTERN}"
echo "GREEDY_PATTERN: ${RUN_GREEDY_PATTERN}"
if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    echo "Forward args: ${FORWARD_ARGS[*]}"
fi
echo "=============================================="
echo ""

for idx in "${!RUN_VERSIONS[@]}"; do
    version="${RUN_VERSIONS[$idx]}"
    exp_name="${RUN_EXP_NAME_PREFIX}_memory_v${version}"
    temp_data_file="${RUN_DATA_DIR}/$(printf "${RUN_TEMP_PATTERN}" "${version}")"
    greedy_data_file="${RUN_DATA_DIR}/$(printf "${RUN_GREEDY_PATTERN}" "${version}")"

    echo "==> [$((idx + 1))/${#RUN_VERSIONS[@]}] memory_version=${version}"
    echo "    EXP_NAME=${exp_name}"
    echo "    TEMP_DATA_FILE=${temp_data_file}"
    echo "    GREEDY_DATA_FILE=${greedy_data_file}"

    if [[ ! -f "${temp_data_file}" ]]; then
        echo "Error: temp data file not found: ${temp_data_file}"
        exit 1
    fi

    if [[ ! -f "${greedy_data_file}" ]]; then
        echo "Error: greedy data file not found: ${greedy_data_file}"
        exit 1
    fi

    bash "${TARGET_SCRIPT}" \
        --exp_name "${exp_name}" \
        --ckpts_dir "${RUN_CKPTS_DIR}" \
        --steps "${RUN_STEPS}" \
        --temp_data_file "${temp_data_file}" \
        --greedy_data_file "${greedy_data_file}" \
        "${FORWARD_ARGS[@]}"

    echo ""
done

echo "==> All memory-version evaluations completed."
