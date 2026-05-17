#!/bin/bash
# =============================================================================
# eval_single_mode_versions.sh - 按版本号批量调用 eval_single_mode_steps.sh
# =============================================================================
#
# 典型用法:
#   1. 直接修改脚本顶部 3 个变量:
#      - VERSIONS
#      - EXP_NAME_PREFIX
#      - CKPTS_BASE_DIR
#   2. 然后执行:
#      ./eval_single_mode_versions.sh
#
# 也支持命令行覆盖:
#   ./eval_single_mode_versions.sh --versions "6 7 8"
#
# 版本号拼接规则:
#   EXP_NAME = ${EXP_NAME_PREFIX}${version}
#   CKPTS_DIR = ${CKPTS_BASE_DIR}/V${version}/ckpts
#
# 例如:
#   version=6
#   EXP_NAME=data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1_skill_evo_v6
#   CKPTS_DIR=/home/yangchengyi/data/skill_saved/Skill_Evo/data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1/Solver/V6/ckpts
#
# 其它参数会原样透传给 eval_single_mode_steps.sh，例如:
#   ./eval_single_mode_versions.sh --versions "6 7" --skip_base_model --steps "10 20 30"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/eval_single_mode_steps.sh"

# =============================================================================
# 只需优先修改这 3 处
# =============================================================================

VERSIONS=(6)
EXP_NAME_PREFIX="data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1_skill_evo_v"
CKPTS_BASE_DIR="/home/yangchengyi/data/skill_saved/Skill_Evo/data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1/Solver"

# =============================================================================
# 可选参数
# =============================================================================

VERSION_LIST_OVERRIDE=""
declare -a FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --versions)
            VERSION_LIST_OVERRIDE="${2:-}"
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
    RUN_VERSIONS=("${VERSIONS[@]}")
fi

if [[ ${#RUN_VERSIONS[@]} -eq 0 ]]; then
    echo "Error: no versions specified"
    exit 1
fi

echo "=============================================="
echo "  Batch Eval By Version"
echo "=============================================="
echo "Target script: ${TARGET_SCRIPT}"
echo "Versions: ${RUN_VERSIONS[*]}"
echo "EXP_NAME_PREFIX: ${EXP_NAME_PREFIX}"
echo "CKPTS_BASE_DIR: ${CKPTS_BASE_DIR}"
if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    echo "Forward args: ${FORWARD_ARGS[*]}"
fi
echo "=============================================="
echo ""

for idx in "${!RUN_VERSIONS[@]}"; do
    version="${RUN_VERSIONS[$idx]}"
    exp_name="${EXP_NAME_PREFIX}${version}"
    ckpts_dir="${CKPTS_BASE_DIR}/V${version}/ckpts"

    echo "==> [$((idx + 1))/${#RUN_VERSIONS[@]}] version=${version}"
    echo "    EXP_NAME=${exp_name}"
    echo "    CKPTS_DIR=${ckpts_dir}"

    if [[ ! -d "${ckpts_dir}" ]]; then
        echo "Error: checkpoint directory not found: ${ckpts_dir}"
        exit 1
    fi

    bash "${TARGET_SCRIPT}" \
        --exp_name "${exp_name}" \
        --ckpts_dir "${ckpts_dir}" \
        "${FORWARD_ARGS[@]}"

    echo ""
done

echo "==> All version evaluations completed."
