#!/usr/bin/env bash
# Synthesizer 训练结束后：从 synthesizer 目录的 synth reward jsonl ingest skill，
# 将 memory 写入 Memory/memory_after_syn_vN.jsonl，并生成当前 solver 版本目录下的 train_data.parquet，
# 以及（默认）同目录下的 test_data.parquet（整文件来自 SE_TEST_DATA_FILE，不推进 data_cursor）。
#
# 用法: memory_func_after_sync.sh <exp_version>
# 依赖环境（与 main_o.sh 一致）: WORKING_DIR, SYNTHESIZER_PATH_DIR, SOLVER_PATH_DIR,
#   MEMORY_PATH_DIR；SE_DATA_FILE 为原始全量语料。after_sync 用 SOLVER 目录下 data_cursor.txt
#   与 SE_SOLVER_BATCH_SIZE、SE_SOLVER_RETRAIN_STEPS（或 solver_retrain_steps），
#   自该下标起取连续 (batch×steps) 条做 retrieve 并写 train_data.parquet，再前推游标。
#   SE_TEST_DATA_FILE：未设置时默认为 /home/ycy/sdi/data/test.jsonl；设置为空字符串则跳过 test_data.parquet。
#   SE_SOLVER_RETRIEVE_TOP_K：传给 retriever /rank 的 top_k（train 与 test 共用）；默认 3。
#   若 after-sync 已成功但曾跳过 test，勿重复跑 after-sync（会推进游标）；请用同目录下
#   memory_func_prepare_test.sh 仅补写 test_data.parquet。
#   SE_MEMORY_MIN_UTILITY：跨文件合并 synth 行之后，同一题仍只入 reward（utility）全局最大的一条，
#   且其 reward 须 >= 本值；默认 0。
#   入库来源：默认整合 ``Synthesizer/<exp>/reward_info/exp_data_step_*.jsonl`` **全部文件**（按 step
#   升序合并后再按题择优）。若仅需单步 jsonl：`export SE_SYNTH_REWARD_INGEST_SINGLE_STEP_ONLY=1`
#   并照常传 ``SE_SYNTHESIZER_STEPS`` / ``--synth-step``。
#   SE_SKILL_MEMORY_MAX_CAPACITY：入库与后续 load_jsonl 时 SkillMemory 容量；默认 8192（见 memory_hook）。
#   少于唯一题个数时 ingest 会因淘汰未满额、load 会截断（旧默认 1000 时可显式设为 1000）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="${SCRIPT_DIR}/../skill_manager/memory_hook.py"
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

if [ -n "${SE_TEST_DATA_FILE+x}" ]; then
  test_data_file="${SE_TEST_DATA_FILE}"
else
  test_data_file="/home/ycy/sdi/data/test_new.jsonl"
fi
TEST_ARGS=()
if [ -n "${test_data_file}" ]; then
  TEST_ARGS=(--test-data-file "${test_data_file}")
fi

python3 "${HOOK}" after-sync "${exp_version}" \
  --synth-step "${synth_step}" \
  --synthesizer-path-dir "${SYNTHESIZER_PATH_DIR:?}" \
  --solver-path-dir "${SOLVER_PATH_DIR:?}" \
  --memory-path-dir "${MEMORY_PATH_DIR:?}" \
  --data-file "${data_file}" \
  "${TEST_ARGS[@]}"
