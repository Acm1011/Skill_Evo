#!/usr/bin/env bash
# Solver 训练结束后：用 reward jsonl 更新各 skill 的 utility，保存 Memory/memory_after_sol_vN.jsonl
#
# 用法: memory_func_after_solver.sh <exp_version> [reward_jsonl_path]
# 未指定第二参数时，在 SOLVER_PATH_DIR/<version>/reward_info/ 下查找：
#   优先 expdata/step_*.jsonl（SolverRewardManager 训练写入），否则 train_data/step_*.jsonl
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="${SCRIPT_DIR}/../skill_manager/memory_hook.py"
exp_version="${1:?exp_version e.g. V1 required}"
reward_path="${2:-}"

export PYTHONPATH="${WORKING_DIR:?}/skill_src${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "${reward_path}" ]; then
  python3 "${HOOK}" after-solver "${exp_version}" "${reward_path}" \
    --solver-path-dir "${SOLVER_PATH_DIR:?}" \
    --memory-path-dir "${MEMORY_PATH_DIR:?}"
else
  python3 "${HOOK}" after-solver "${exp_version}" \
    --solver-path-dir "${SOLVER_PATH_DIR:?}" \
    --memory-path-dir "${MEMORY_PATH_DIR:?}"
fi
