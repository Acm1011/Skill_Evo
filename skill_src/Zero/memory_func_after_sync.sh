#!/usr/bin/env bash
# Synthesizer 训练结束后：从 synthesizer 目录的 synth reward jsonl ingest skill，
# 将 memory 写入 Memory/memory_after_syn_vN.jsonl，并生成当前 solver 版本目录下的 train_data.parquet。
#
# 用法: memory_func_after_sync.sh <exp_version>
# 依赖环境（与 main_o.sh 一致）: WORKING_DIR, SYNTHESIZER_PATH_DIR, SOLVER_PATH_DIR,
#   MEMORY_PATH_DIR；数据路径可用 SE_DATA_FILE 或调用方在环境中导出 data_file
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="${SCRIPT_DIR}/memory_hook.py"
exp_version="${1:?exp_version e.g. V1 required}"

if [ -z "${data_file:-}" ] && [ -n "${SE_DATA_FILE:-}" ]; then
  data_file="${SE_DATA_FILE}"
fi
if [ -z "${data_file:-}" ]; then
  echo "data_file or SE_DATA_FILE must be set" >&2
  exit 1
fi

synth_step="${SE_SYNTHESIZER_STEPS:-${synthesizer_training_steps:-20}}"

export PYTHONPATH="${WORKING_DIR:?}/skill_src${PYTHONPATH:+:$PYTHONPATH}"

python3 "${HOOK}" after-sync "${exp_version}" \
  --synth-step "${synth_step}" \
  --synthesizer-path-dir "${SYNTHESIZER_PATH_DIR:?}" \
  --solver-path-dir "${SOLVER_PATH_DIR:?}" \
  --memory-path-dir "${MEMORY_PATH_DIR:?}" \
  --data-file "${data_file}"
