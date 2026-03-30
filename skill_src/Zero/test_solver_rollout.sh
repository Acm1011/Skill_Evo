#!/usr/bin/env bash
# =============================================================================
# test_solver_rollout.sh - 端到端测试 solver_offline_driver + solver_offline_rollout_server
#
# Server 启动复用同目录下的 start_rollout_servers.sh（后台运行，结束后对其发 SIGTERM 触发其 trap 清理子进程）。
#
# 用法:
#   ./test_solver_rollout.sh [--model <本地模型目录或 HF id>] [--data <jsonl>]
#   # 或与 run_with_gpus 统一分配 GPU:
#   ./run_with_gpus.sh 2 test-rollout
#   ./run_with_gpus.sh 2 test-rollout --model /path/to/model
#
#   SE_N_GPUS=8 ... ./test_solver_rollout.sh
#
# 说明:
#   - 多卡且 NUM_RANDOM_QUESTIONS>0 时，需 STEPS*BATCH_SIZE 足够大（脚本内会校验）
#   - 单卡默认: STEPS=1 BATCH_SIZE=16 NUM_RANDOM_QUESTIONS=10
#   - 模型优先级: --model > MODEL_PATH > DEFAULT_ROLLOUT_MODEL（默认 ${SE_MODEL_DIR}/Qwen2.5-3B-Instruct）
# =============================================================================
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 与 run_with_gpus 路径约定一致；可用 DEFAULT_ROLLOUT_MODEL 覆盖整条默认路径
dir="${SE_BASE_DIR:-/home/ycy/sdi}"
SE_MODEL_DIR="${SE_MODEL_DIR:-${dir}/models}"
DEFAULT_ROLLOUT_MODEL="${DEFAULT_ROLLOUT_MODEL:-${SE_MODEL_DIR}/Qwen2.5-3B-Instruct}"

usage_test_rollout() {
  echo "用法: $(basename "$0") [--model <模型路径或 HF id>] [--data <jsonl>]" >&2
  echo "  优先级: --model > MODEL_PATH > DEFAULT_ROLLOUT_MODEL（当前默认: ${DEFAULT_ROLLOUT_MODEL}）" >&2
}

model_path=""
data_path_cli=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { echo "[ERROR] --model 需要参数" >&2; exit 1; }
      model_path="$2"
      shift 2
      ;;
    --data)
      [[ $# -ge 2 ]] || { echo "[ERROR] --data 需要参数" >&2; exit 1; }
      data_path_cli="$2"
      shift 2
      ;;
    -h|--help)
      usage_test_rollout
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      usage_test_rollout
      exit 1
      ;;
  esac
done

if [[ -z "${model_path}" ]]; then
  model_path="${MODEL_PATH:-}"
fi
if [[ -z "${model_path}" ]]; then
  model_path="${DEFAULT_ROLLOUT_MODEL}"
fi

WORKING_DIR="${SE_WORKING_DIR:-$REPO_ROOT}"
SE_CODE_MODULE="${SE_CODE_MODULE:-skill_src}"
SE_Synthsizer_DIR="${SE_Synthsizer_DIR:-${dir}/Synthsizer}"
SE_ROLLOUT_DIR="${SE_ROLLOUT_DIR:-${SE_Synthsizer_DIR}}"

if [[ -n "${data_path_cli}" ]]; then
  data_path="${data_path_cli}"
else
  data_path="${DATA_PATH:-/mnt/sdi/ycy/data/DeepMath-103K.jsonl}"
fi

SE_N_GPUS="${SE_N_GPUS:-1}"
SE_GPU_IDS="${SE_GPU_IDS:-0}"
ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT:-${SE_ROLLOUT_BASE_PORT:-8760}}"
SE_ROLLOUT_HOST="${SE_ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_SERVER_GPU_UTIL="${ROLLOUT_SERVER_GPU_UTIL:-0.9}"
PYTHON="${PYTHON:-python3}"

STEPS="${STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
# random_q 由 driver 在合并各 shard 后按「本 round 全局」采样，不依赖单分片大小；单卡默认 10，多卡默认 3
if [[ "${SE_N_GPUS:-1}" -le 1 ]]; then
  NUM_RANDOM_QUESTIONS="${NUM_RANDOM_QUESTIONS:-10}"
else
  NUM_RANDOM_QUESTIONS="${NUM_RANDOM_QUESTIONS:-4}"
fi

TEST_TAG="solver_rollout_test_$(date '+%Y%m%d_%H%M%S')"
WORK_DIR="${WORK_DIR:-${SE_ROLLOUT_DIR}/${TEST_TAG}}"
# 游标状态由 driver 默认写入 ${WORK_DIR}/train_cursor_state.json（无需预先存在）；自定义可 export STATE_PATH 并加 --state-path
MERGE_OUT="${MERGE_OUT:-${WORK_DIR}/merged_train}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/server_logs}"

IFS=',' read -ra GPU_ARR <<< "${SE_GPU_IDS// /}"
if [[ "${#GPU_ARR[@]}" -ne "${SE_N_GPUS}" ]]; then
  echo "[ERROR] SE_GPU_IDS 解析为 ${#GPU_ARR[@]} 个 ID，与 SE_N_GPUS=${SE_N_GPUS} 不一致" >&2
  exit 1
fi

need=$((STEPS * BATCH_SIZE))
if [[ "${NUM_RANDOM_QUESTIONS}" -gt 0 && "${need}" -lt $((NUM_RANDOM_QUESTIONS + 1)) ]]; then
  echo "[ERROR] NUM_RANDOM_QUESTIONS=${NUM_RANDOM_QUESTIONS} 需要 STEPS*BATCH_SIZE >= num_random+1（当前 need=${need}）。" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${MERGE_OUT}"

echo "=============================================="
echo "  test_solver_rollout"
echo "  WORKING_DIR: ${WORKING_DIR}"
echo "  model_path:  ${model_path}"
echo "  data_path:   ${data_path}"
echo "  SE_N_GPUS:   ${SE_N_GPUS}  SE_GPU_IDS: ${SE_GPU_IDS}"
echo "  driver:      steps=${STEPS} batch_size=${BATCH_SIZE} num_random_q=${NUM_RANDOM_QUESTIONS}"
echo "  work_dir:    ${WORK_DIR}"
echo "  server 脚本: ${SCRIPT_DIR}/start_rollout_servers.sh"
echo "=============================================="

# 供 start_rollout_servers.sh 使用（与 run_with_gpus.sh 命名一致）
export SE_WORKING_DIR="${WORKING_DIR}"
export PYTHON
export ROLLOUT_SERVER_MODEL="${model_path}"
export SE_ROLLOUT_MODEL="${model_path}"
export SE_ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT}"
export ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT}"
export SE_ROLLOUT_LOG_DIR="${LOG_DIR}"
export SE_N_GPUS
export SE_GPU_IDS

STARTER_PID=""
cleanup_test() {
  if [[ -n "${STARTER_PID}" ]] && kill -0 "${STARTER_PID}" 2>/dev/null; then
    echo ""
    echo "[test] 结束 start_rollout_servers（PID ${STARTER_PID}），将触发其内部清理..."
    kill -TERM "${STARTER_PID}" 2>/dev/null || true
    wait "${STARTER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_test EXIT INT TERM

echo "[test] 后台启动: bash ${SCRIPT_DIR}/start_rollout_servers.sh --model <...>"
bash "${SCRIPT_DIR}/start_rollout_servers.sh" --model "${model_path}" &
STARTER_PID=$!

echo "[test] 等待 /health 就绪（vLLM 加载可能较久，最长约 2 小时）..."
for i in $(seq 0 $((SE_N_GPUS - 1))); do
  port=$((ROLLOUT_BASE_PORT + i))
  url="http://${SE_ROLLOUT_HOST}:${port}/health"
  ok=0
  for _ in $(seq 1 2400); do
    if curl -sf "${url}" | "${PYTHON}" -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('ok') else 1)" 2>/dev/null; then
      echo "[test]  ${url} ok"
      ok=1
      break
    fi
    sleep 3
  done
  if [[ "${ok}" -ne 1 ]]; then
    echo "[ERROR] 超时未就绪: ${url}，请查看 ${LOG_DIR}" >&2
    exit 1
  fi
done

export SE_ROLLOUT_N_SERVERS="${SE_N_GPUS}"
export SE_ROLLOUT_HOST="${SE_ROLLOUT_HOST}"
rollout_urls=""
for i in $(seq 0 $((SE_N_GPUS - 1))); do
  port=$((ROLLOUT_BASE_PORT + i))
  u="http://${SE_ROLLOUT_HOST}:${port}"
  rollout_urls="${rollout_urls}${rollout_urls:+ }${u}"
done
export SE_ROLLOUT_SERVER_URLS="${rollout_urls}"

cd "${WORKING_DIR}"
export PYTHONPATH="${WORKING_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[test] 运行 solver_offline_driver run..."
DRIVER_EXTRA=()
if [[ -n "${STATE_PATH:-}" ]]; then
  DRIVER_EXTRA+=(--state-path "${STATE_PATH}")
fi
"${PYTHON}" -m "${SE_CODE_MODULE}.solver_offline_driver" run \
  --data-files "${data_path}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --work-dir "${WORK_DIR}" \
  "${DRIVER_EXTRA[@]}" \
  --merge-output-dir "${MERGE_OUT}" \
  --exp-name "${TEST_TAG}" \
  --num-random-questions "${NUM_RANDOM_QUESTIONS}" \
  --reset-state \
  --max-tokens 4096 \
  --rollout-n 4 \
  --request-timeout 300 \
  --skill-type "skill_generation_v1" \

echo ""
echo "[test] 完成。合并结果: ${MERGE_OUT}/train_data.jsonl"
echo "[test] 游标状态（默认）: ${WORK_DIR}/train_cursor_state.json"
echo "=============================================="
