#!/usr/bin/env bash
# 在已成功跑过 after-sync、已有 memory_after_syn_vN.jsonl 的前提下，仅生成
# {SOLVER_PATH_DIR}/{exp}/test_data.parquet，不推进 data_cursor、不重写 train_data.parquet。
#
# 用法: memory_func_prepare_test.sh <exp_version>
# 环境: WORKING_DIR, SOLVER_PATH_DIR, MEMORY_PATH_DIR；可选 SE_TEST_DATA_FILE（默认 /home/ycy/sdi/data/test.jsonl）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="${SCRIPT_DIR}/../skill_manager/memory_hook.py"
exp_version="${1:?exp_version e.g. V1 required}"

if [ -n "${SE_TEST_DATA_FILE+x}" ]; then
  test_data_file="${SE_TEST_DATA_FILE}"
else
  test_data_file="/home/ycy/sdi/data/test.jsonl"
fi
if [ -z "${test_data_file}" ]; then
  echo "SE_TEST_DATA_FILE is empty; set it to a test.jsonl path" >&2
  exit 1
fi

export PYTHONPATH="${WORKING_DIR:?}/skill_src${PYTHONPATH:+:$PYTHONPATH}"

python3 "${HOOK}" prepare-test "${exp_version}" \
  --solver-path-dir "${SOLVER_PATH_DIR:?}" \
  --memory-path-dir "${MEMORY_PATH_DIR:?}" \
  --test-data-file "${test_data_file}"
