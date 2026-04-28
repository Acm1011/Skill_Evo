#!/usr/bin/env bash
# =============================================================================
# Synthesizer_vllm_http.sh - Synthesizer 全流程（vLLM HTTP + rollout_http_client）
# =============================================================================
# 用法与 Synthesizer.sh 相同:
#   ./Synthesizer_vllm_http.sh <exp_name> <synthesizer_model_path> <solver_model_path> \
#                            <training_steps> <data_file> [<embedding_cache_path>]
#
# GPU 角色与 Synthesizer.sh 一致（run_with_gpus / SE_GPU_IDS 顺序不变）:
#   - 前半 SE_GPU_IDS: Step 3 上 Synthesizer RL
#   - 后半 SE_GPU_IDS: Step 3 上保留 vLLM + solver_offline_rollout_http_client，供 reward
#
# 流程:
#   Step 1a. 全部 GPU 启动 vllm serve（skill_src/start_vllm_http_servers.sh）
#   Step 1b. 每 GPU 一个 solver_offline_rollout_http_client（/rollout -> 对应 vLLM）
#   Step 2.  solver_offline_driver 离线 rollout（SE_ROLLOUT_SERVER_URLS 指向全部 client）
#   Step 2.5. 释放前半 GPU：按端口结束前半卡上的 vLLM 与 http client（与 Synthesizer 训练用卡一致）
#   Step 3.  前半 GPU 上 main_synthesizer；仅后半 client URL 写入 SE_ROLLOUT_SERVER_URLS 供 reward
#
# 环境变量（额外）:
#   VLLM_HTTP_BASE_PORT   vLLM 起始端口，默认同 SE_ROLLOUT_BASE_PORT 或 8760
#   ROLLOUT_HTTP_BASE_PORT  每卡 rollout_http_client 起始端口，默认 8860（勿与 vLLM 端口区间重叠）
#   VLLM_HTTP_TIMEOUT / VLLM_HTTP_MAX_RETRIES / VLLM_HTTP_RETRY_DELAY
#       传给每卡 rollout_http_client（调 vLLM /v1/completions），默认 600s / 8 / 2
#   SERVED_MODEL_NAME     与 vllm serve --served-model-name 一致，默认 default（须与 vllm_http_client 一致）
#   VLLM_DTYPE            传给 vllm serve，默认 auto
#   SE_PREBUILT_ROLLOUT_PATH  预生成的 rollout 文件路径（jsonl 或 parquet）
#       若设置且文件存在，将跳过离线 rollout 阶段（Step 1a/1b/2），直接使用该文件进行 RL 训练
#       此时仅后半卡启动 vLLM + client 供 reward 评估使用
#
# 注意:
#   - 勿对 solver_offline_driver 使用 --shutdown-servers-after-run：会按 URL 端口误杀 http client。
#   - 需要本机有 lsof（按端口结束进程）。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# ========================== 解析位置参数 ==========================
exp_name="$1"
synthesizer_model_path="$2"
solver_model_path="$3"
synthesizer_training_steps="$4"
data_file="$5"
embedding_cache_path="${6-${SE_EMBEDDING_CACHE_PATH:-}}"
prebuilt_rollout_path="${SE_PREBUILT_ROLLOUT_PATH:-}"
echo "[exp_name]: ${exp_name}"
echo "[synthesizer_model_path]: ${synthesizer_model_path}"
echo "[solver_model_path]: ${solver_model_path}"
echo "[synthesizer_training_steps]: ${synthesizer_training_steps}"
echo "[data_file]: ${data_file}"
echo "[embedding_cache_path]: ${embedding_cache_path}"
echo "[prebuilt_rollout_path]: ${prebuilt_rollout_path}"

# ========================== 验证参数 ==========================
for var_name in exp_name synthesizer_model_path solver_model_path synthesizer_training_steps data_file; do
    if [ -z "${!var_name}" ]; then
        echo "Error: ${var_name} 不能为空"; exit 1
    fi
done
if ! [[ "$synthesizer_training_steps" =~ ^[0-9]+$ ]]; then
    echo "Error: synthesizer_training_steps 必须是数字，当前值: $synthesizer_training_steps"; exit 1
fi
SYNTH_PPO_STEPS="${synthesizer_training_steps}"
[ -d "$synthesizer_model_path" ] || { echo "Error: synthesizer_model_path 不存在: $synthesizer_model_path"; exit 1; }
[ -d "$solver_model_path" ]      || { echo "Error: solver_model_path 不存在: $solver_model_path"; exit 1; }
[ -f "$data_file" ]              || { echo "Error: data_file 不存在: $data_file"; exit 1; }

if ! command -v lsof >/dev/null 2>&1; then
    echo "Error: 需要 lsof（用于按端口释放 GPU 对应进程）"; exit 1
fi

if [ "${SE_N_GPUS:-0}" -lt 2 ]; then
    echo "Error: SE_N_GPUS 至少为 2（需前后半各至少 1 张 GPU）"; exit 1
fi

# ========================== 路径配置 ==========================
storage_path="${SE_Synthsizer_DIR}/workspace/${exp_name}"
CKPTS_DIR="${SE_Synthsizer_DIR}/ckpts/"
tensorboard_path="${SE_TENSORBOARD_DIR}"
mkdir -p "${CKPTS_DIR}" "${tensorboard_path}" "${storage_path}"
export TENSORBOARD_DIR="${tensorboard_path}"

echo "[路径配置] 工作目录: ${SE_WORKING_DIR}"
echo "[路径配置] 存储路径: ${storage_path}"

# ========================== GPU 分配（与 Synthesizer.sh 相同） ==========================
HALF=$((SE_N_GPUS / 2))
IFS=',' read -ra ALL_GPUS <<< "${SE_GPU_IDS}"

SYNTH_GPU_IDS=""
for ((i=0; i<HALF; i++)); do
    SYNTH_GPU_IDS="${SYNTH_GPU_IDS:+${SYNTH_GPU_IDS},}${ALL_GPUS[$i]}"
done

ROLLOUT_GPU_IDS=""
for ((i=HALF; i<SE_N_GPUS; i++)); do
    ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:+${ROLLOUT_GPU_IDS},}${ALL_GPUS[$i]}"
done

N_SYNTH_GPUS="${HALF}"
N_ROLLOUT_SERVERS="${HALF}"

VLLM_HTTP_BASE_PORT="${VLLM_HTTP_BASE_PORT:-${SE_ROLLOUT_BASE_PORT:-${ROLLOUT_BASE_PORT:-8760}}}"
ROLLOUT_HTTP_BASE_PORT="${ROLLOUT_HTTP_BASE_PORT:-8860}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-default}"
export SERVED_MODEL_NAME
SE_ROLLOUT_HOST="${SE_ROLLOUT_HOST:-127.0.0.1}"

SE_CODE_MODULE="${SE_CODE_MODULE:-skill_src}"
START_VLLM_SH="${SE_WORKING_DIR}/${SE_CODE_MODULE}/start_vllm_http_servers.sh"
PYTHON="${PYTHON:-python3}"

echo "[GPU配置] Synthesizer GPUs: ${SYNTH_GPU_IDS} (${N_SYNTH_GPUS} 张)"
echo "[GPU配置] Rollout GPUs:     ${ROLLOUT_GPU_IDS} (${N_ROLLOUT_SERVERS} 张)"
echo "[端口] vLLM BASE=${VLLM_HTTP_BASE_PORT}  rollout_http_client BASE=${ROLLOUT_HTTP_BASE_PORT}  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"

[ -f "${START_VLLM_SH}" ] || { echo "Error: 未找到 ${START_VLLM_SH}"; exit 1; }

LOG_DIR="${SE_ROLLOUT_LOG_DIR:-${SE_ROLLOUT_DIR}/logs/rollout_vllm_http}"
mkdir -p "${LOG_DIR}"

VLLM_LAUNCHER_PID=""
HTTP_CLIENT_PIDS=()

cleanup_all() {
    local s=$?
    echo ""
    echo "========== 清理 vLLM / rollout_http_client =========="
    for pid in "${HTTP_CLIENT_PIDS[@]:-}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    if [ -n "${VLLM_LAUNCHER_PID}" ] && kill -0 "${VLLM_LAUNCHER_PID}" 2>/dev/null; then
        echo "清理 vLLM 启动脚本 (PID: ${VLLM_LAUNCHER_PID})..."
        kill "${VLLM_LAUNCHER_PID}" 2>/dev/null || true
        wait "${VLLM_LAUNCHER_PID}" 2>/dev/null || true
    fi
    echo "清理完成"
    exit "${s:-0}"
}
trap cleanup_all EXIT INT TERM

# ===========================================================================
# 判断是否使用预生成 rollout 文件
# ===========================================================================
USE_PREBUILT_ROLLOUT=false
if [ -n "$prebuilt_rollout_path" ] && [ -f "$prebuilt_rollout_path" ]; then
    USE_PREBUILT_ROLLOUT=true
    echo ""
    echo "========== 检测到预生成 rollout 文件，将跳过离线 rollout 阶段 =========="
    echo "  预生成文件: ${prebuilt_rollout_path}"
fi

# ===========================================================================
# Step 1a: 启动 vllm serve
# ===========================================================================
echo ""
if [ "$USE_PREBUILT_ROLLOUT" = true ]; then
    echo "========== Step 1a: 启动 vLLM HTTP servers（仅后半卡，供 reward 评估）=========="
    # 仅后半 GPU 需要启动
    ROLLOUT_ONLY_GPU_IDS="${ROLLOUT_GPU_IDS}"
    N_VLLM_SERVERS="${N_ROLLOUT_SERVERS}"
else
    echo "========== Step 1a: 启动 vLLM HTTP servers（全卡）=========="
    # 全部 GPU 需要启动
    ROLLOUT_ONLY_GPU_IDS="${SE_GPU_IDS}"
    N_VLLM_SERVERS="${SE_N_GPUS}"
fi

cd "${SE_WORKING_DIR}"
if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${SE_WORKING_DIR}"
else
    export PYTHONPATH="${SE_WORKING_DIR}:${PYTHONPATH}"
fi

export BASE_PORT="${VLLM_HTTP_BASE_PORT}"
export DTYPE="${VLLM_DTYPE:-auto}"
bash "${START_VLLM_SH}" "${ROLLOUT_ONLY_GPU_IDS}" "${solver_model_path}" &
VLLM_LAUNCHER_PID=$!
echo "vLLM 启动脚本 PID: ${VLLM_LAUNCHER_PID}"

MAX_WAIT=600
for ((i=0; i<N_VLLM_SERVERS; i++)); do
    port=$((VLLM_HTTP_BASE_PORT + i))
    waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            echo "  vLLM server ${i} (port ${port}) 就绪 (${waited}s)"
            break
        fi
        sleep 5; waited=$((waited + 5))
    done
    if [ $waited -ge $MAX_WAIT ]; then
        echo "Error: vLLM server ${i} (port ${port}) 启动超时"; exit 1
    fi
done

# 构建 vLLM URLs CSV（仅后半卡或全卡）
VLLM_URLS_CSV=""
for ((i=0; i<N_VLLM_SERVERS; i++)); do
    vport=$((VLLM_HTTP_BASE_PORT + i))
    VLLM_URLS_CSV="${VLLM_URLS_CSV:+$VLLM_URLS_CSV,}http://127.0.0.1:${vport}"
done
VLLM_HTTP_TIMEOUT="${VLLM_HTTP_TIMEOUT:-600}"
VLLM_HTTP_MAX_RETRIES="${VLLM_HTTP_MAX_RETRIES:-8}"
VLLM_HTTP_RETRY_DELAY="${VLLM_HTTP_RETRY_DELAY:-2}"

# ===========================================================================
# Step 1b: 启动 solver_offline_rollout_http_client
# ===========================================================================
echo ""
if [ "$USE_PREBUILT_ROLLOUT" = true ]; then
    echo "========== Step 1b: 启动 solver_offline_rollout_http_client（仅后半卡，供 reward）=========="
else
    echo "========== Step 1b: 启动 solver_offline_rollout_http_client（每卡）=========="
fi
echo "  [vLLM HTTP] 每进程 --vllm-urls 相同（故障转移） timeout=${VLLM_HTTP_TIMEOUT}s max_retries=${VLLM_HTTP_MAX_RETRIES} retry_delay=${VLLM_HTTP_RETRY_DELAY}s"

for ((i=0; i<N_VLLM_SERVERS; i++)); do
    vport=$((VLLM_HTTP_BASE_PORT + i))
    cport=$((ROLLOUT_HTTP_BASE_PORT + i))
    log="${LOG_DIR}/rollout_http_client_${i}_v${vport}_p${cport}.log"
    echo "  client i=${i} vLLM_urls=${VLLM_URLS_CSV} listen=:${cport} -> ${log}"
    "${PYTHON}" -m "${SE_CODE_MODULE}.solver_offline_rollout_http_client" \
        --vllm-urls "${VLLM_URLS_CSV}" \
        --model "${solver_model_path}" \
        --host 0.0.0.0 \
        --port "${cport}" \
        --timeout "${VLLM_HTTP_TIMEOUT}" \
        --max-retries "${VLLM_HTTP_MAX_RETRIES}" \
        --retry-delay "${VLLM_HTTP_RETRY_DELAY}" \
        >>"${log}" 2>&1 &
    HTTP_CLIENT_PIDS+=($!)
done

for ((i=0; i<N_VLLM_SERVERS; i++)); do
    cport=$((ROLLOUT_HTTP_BASE_PORT + i))
    waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if curl -sf "http://127.0.0.1:${cport}/health" > /dev/null 2>&1; then
            echo "  rollout_http_client ${i} (port ${cport}) 就绪 (${waited}s)"
            break
        fi
        sleep 2; waited=$((waited + 2))
    done
    if [ $waited -ge $MAX_WAIT ]; then
        echo "Error: rollout_http_client ${i} (port ${cport}) 启动超时"; exit 1
    fi
done

# driver / reward：显式 URL 列表；勿用整机 SE_ROLLOUT_N_SERVERS（会与真实进程数不一致）
unset SE_ROLLOUT_N_SERVERS
unset SE_ROLLOUT_PORTS
export ROLLOUT_SERVER_MODEL="${solver_model_path}"

# ===========================================================================
# Step 2: 离线 rollout（仅在未使用预生成文件时执行）
# ===========================================================================
if [ "$USE_PREBUILT_ROLLOUT" = true ]; then
    echo ""
    echo "========== Step 2: 使用预生成 rollout 文件，跳过离线 rollout =========="
    TRAIN_DATA_FILE="${prebuilt_rollout_path}"
    echo "  训练数据: ${TRAIN_DATA_FILE}"
    
    # 设置 SE_ROLLOUT_SERVER_URLS 为后半卡（用于 reward 评估）
    _rollout_urls_tail=""
    for ((i=0; i<N_ROLLOUT_SERVERS; i++)); do
        cport=$((ROLLOUT_HTTP_BASE_PORT + i))
        _rollout_urls_tail="${_rollout_urls_tail:+${_rollout_urls_tail} }http://${SE_ROLLOUT_HOST}:${cport}"
    done
    export SE_ROLLOUT_SERVER_URLS="${_rollout_urls_tail}"
    echo "[env] Step 3 SE_ROLLOUT_SERVER_URLS=${SE_ROLLOUT_SERVER_URLS}"
else
    # 原有完整流程
    _rollout_urls_all=""
    for ((i=0; i<SE_N_GPUS; i++)); do
        cport=$((ROLLOUT_HTTP_BASE_PORT + i))
        _rollout_urls_all="${_rollout_urls_all:+${_rollout_urls_all} }http://${SE_ROLLOUT_HOST}:${cport}"
    done
    export SE_ROLLOUT_SERVER_URLS="${_rollout_urls_all}"
    echo "[env] Step 2 SE_ROLLOUT_SERVER_URLS=${SE_ROLLOUT_SERVER_URLS}"

    echo ""
    echo "========== Step 2: 离线 rollout =========="

    OFFLINE_WORK_DIR="${storage_path}"
    OFFLINE_MERGE_DIR="${OFFLINE_WORK_DIR}/merged"
    mkdir -p "${OFFLINE_WORK_DIR}" "${OFFLINE_MERGE_DIR}"

    echo "  data_file: ${data_file}"
    SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER="${SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER:-${SE_OFFLINE_ROLLOUT_STEPS:-2}}"
    export SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER
    OFFLINE_DRIVER_STEPS=$(( SYNTH_PPO_STEPS * SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER ))
    OFFLINE_DRIVER_BATCH="${SE_OFFLINE_ROLLOUT_BATCH_SIZE:-${SYNTH_BATCH_SIZE:-16}}"
    _OFFLINE_NEED=$(( OFFLINE_DRIVER_STEPS * OFFLINE_DRIVER_BATCH ))
    echo "  PPO 训练步数 T=${SYNTH_PPO_STEPS}"
    echo "  offline driver: --steps ${OFFLINE_DRIVER_STEPS} (= T×mult)  --batch-size ${OFFLINE_DRIVER_BATCH}  (need target=${_OFFLINE_NEED} rows; mult=${SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER})  rollout_n=${SE_OFFLINE_ROLLOUT_N}"
    SE_OFFLINE_RESET_STATE="${SE_OFFLINE_RESET_STATE:-0}"

    offline_cmd=(
        "${PYTHON}" -m "${SE_CODE_MODULE}.solver_offline_driver" run
        --data-files "${data_file}"
        --steps "${OFFLINE_DRIVER_STEPS}"
        --batch-size "${OFFLINE_DRIVER_BATCH}"
        --work-dir "${OFFLINE_WORK_DIR}"
        --merge-output-dir "${OFFLINE_MERGE_DIR}"
        --merge-prefix "train_data"
        --skill-type "${SE_OFFLINE_SKILL_TYPE}"
        --rollout-n "${SE_OFFLINE_ROLLOUT_N}"
        --num-random-questions "${SE_OFFLINE_NUM_RANDOM_Q}"
        --model-path "${solver_model_path}"
    )
    if [ "${SE_OFFLINE_RESET_STATE}" = "1" ]; then
        offline_cmd+=(--reset-state)
        echo "  [offline] SE_OFFLINE_RESET_STATE=1：从 cursor=0 重置 state"
    fi
    if [ -n "$embedding_cache_path" ]; then
        offline_cmd+=(--embedding-cache-path "${embedding_cache_path}")
    fi
    "${offline_cmd[@]}" || { echo "Error: 离线 rollout 失败"; exit 1; }

    echo "离线 rollout 完成"

    TRAIN_DATA_FILE="${OFFLINE_MERGE_DIR}/train_data.parquet"
    [ -f "${TRAIN_DATA_FILE}" ] || TRAIN_DATA_FILE="${OFFLINE_MERGE_DIR}/train_data.jsonl"
    [ -f "${TRAIN_DATA_FILE}" ] || { echo "Error: 未找到训练数据"; exit 1; }
    echo "训练数据: ${TRAIN_DATA_FILE}"

    # ===========================================================================
    # Step 2.5: 释放前半 GPU（与 Synthesizer 训练用卡一致）
    # ===========================================================================
    echo ""
    echo "========== Step 2.5: 结束前半卡 vLLM 与 rollout_http_client（释放 Synthesizer GPU）=========="

    for ((i=0; i<SE_N_GPUS; i++)); do
        if [ "$i" -lt "$HALF" ]; then
            vp=$((VLLM_HTTP_BASE_PORT + i))
            cp=$((ROLLOUT_HTTP_BASE_PORT + i))
            echo "  释放索引 i=${i}（物理 GPU ${ALL_GPUS[$i]}）: 结束 vLLM 端口 ${vp}、client 端口 ${cp}"
            lsof -ti:"${vp}" | xargs -r kill -15 || true
            lsof -ti:"${cp}" | xargs -r kill -15 || true
        fi
    done
    sleep 3

    _rollout_urls_tail=""
    for ((i=HALF; i<SE_N_GPUS; i++)); do
        cport=$((ROLLOUT_HTTP_BASE_PORT + i))
        _rollout_urls_tail="${_rollout_urls_tail:+${_rollout_urls_tail} }http://${SE_ROLLOUT_HOST}:${cport}"
    done
    export SE_ROLLOUT_SERVER_URLS="${_rollout_urls_tail}"
    echo "[env] Step 3 SE_ROLLOUT_SERVER_URLS=${SE_ROLLOUT_SERVER_URLS}"
fi

# ===========================================================================
# Step 3: Synthesizer RL 训练
# ===========================================================================
echo ""
echo "========== Step 3: Synthesizer RL 训练 =========="

batch_size="${SYNTH_BATCH_SIZE}"
rollout_query_num="${SYNTH_ROLLOUT_QUERY_NUM}"
query_top_p="${SYNTH_QUERY_TOP_P}"
query_top_k="${SYNTH_QUERY_TOP_K}"
kl_loss_coef="${SYNTH_KL_LOSS_COEF}"
query_temperature="${SYNTH_QUERY_TEMPERATURE}"
tp="${SYNTH_TP}"
max_prompt_length="${SYNTH_MAX_PROMPT_LENGTH}"
max_response_length="${SYNTH_MAX_RESPONSE_LENGTH}"
gpu_mem_util="${SYNTH_GPU_MEM_UTIL}"
random_q_coef="${SYNTH_RANDOM_Q_COEF}"
use_skill_type="${SYNTH_USE_SKILL_TYPE}"

ppo_mini_batch_size=$((batch_size / 4))
micro_batch_size_per_gpu=$((ppo_mini_batch_size / 4))
if [ "${micro_batch_size_per_gpu}" -lt 1 ]; then
    micro_batch_size_per_gpu=1
fi

echo "  训练 GPU: ${SYNTH_GPU_IDS} (${N_SYNTH_GPUS} 张)"
echo "  batch=${batch_size}   kl=${kl_loss_coef}  temp=${query_temperature}"

CUDA_VISIBLE_DEVICES=${SYNTH_GPU_IDS} "${PYTHON}" -m "${SE_CODE_MODULE}.main_synthesizer" \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA_FILE}" \
    data.shuffle=False \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${synthesizer_model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${query_temperature} \
    actor_rollout_ref.rollout.top_p=${query_top_p} \
    actor_rollout_ref.rollout.top_k="${query_top_k}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=${rollout_query_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=synthesizer \
    reward_model.reward_kwargs.storage_path=${storage_path} \
    +reward_model.reward_kwargs.use_skill_type=${use_skill_type} \
    +reward_model.reward_kwargs.random_q_coef=${random_q_coef} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.project_name="${SE_PROJECT_NAME}" \
    trainer.experiment_name="Synthesizer-${exp_name}" \
    trainer.n_gpus_per_node=${N_SYNTH_GPUS} \
    trainer.total_training_steps=${SYNTH_PPO_STEPS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 &

TRAINING_PID=$!
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "Synthesizer 训练失败，退出码: $TRAINING_EXIT_CODE"
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "=============================================="
echo "  ${exp_name} Synthesizer（vLLM HTTP rollout）训练完成"
echo "  model path: ${CKPTS_DIR}"
echo "=============================================="
