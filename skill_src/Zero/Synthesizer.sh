#!/usr/bin/env bash
# =============================================================================
# Synthesizer.sh - Synthesizer 完整训练执行器
# =============================================================================
# 用法（由 main.sh 调用，参数和超参数均通过位置参数 + 环境变量传入）:
#   ./Synthesizer.sh <exp_name> <synthesizer_model_path> <solver_model_path> \
#                    <training_steps> <data_file> [<embedding_cache_path>]
#   第 6 个参数可选：embedding cache 目录（含 *.meta.json + *.npz）；省略或与 SE_EMBEDDING_CACHE_PATH 均未设置时为空（随机采样）。
#
# 流程:
#   Step 1. 全机 SE_GPU_IDS 上启动 rollout (start_rollout_servers.sh) + 健康检查（offline 用满所有卡）
#   Step 2. solver_offline_driver 离线 rollout → merged parquet
#   2.5 offline 结束后 pkill python，释放 vLLM 与 GPU
#   Step 3. 仅后半卡再启 reward rollout + Step 4 RL：后半卡 vLLM 与 main_synthesizer 同时起，随后再等待各
#   /health（与 Ray/模型加载重叠，省时间；须 trainer.val_before_train=False 以免首步早于 HTTP reward）
#   退出时自动清理 Step 3 的 rollout server
#
#   SE_SYNTH_SKIP_OFFLINE=1|true|yes（由 main_o.sh 在 SE_RESUME 且已有 merged/train_data 时仅对本次子进程设置）:
#   跳过 Step1–2，要求 ${SYNTHESIZER_PATH_DIR}/<version>/merged/train_data.parquet 或 train_data.jsonl 已存在，再进入 Step3–4。
#
# 超参数环境变量（由 main.sh export）:
#   Offline rollout: 第4参 = PPO 基座步数 T；driver --steps = T×mult，--batch-size = SYNTH_BATCH；
#                    need=(T×mult)×batch，用于多采以抵消 all-fail；RL 仍为 T 步。SE_OFFLINE_ROLLOUT_N / …
#   游标: 默认不 --reset-state；work_dir 下 train_cursor_state.json 与 data_cursor.txt 在每次 offline 后更新。
#         需从 0 重跑全库时设 SE_OFFLINE_RESET_STATE=1
#   RL training:     SYNTH_BATCH_SIZE / SYNTH_ROLLOUT_QUERY_NUM / SYNTH_QUERY_TOP_P /
#                    SYNTH_QUERY_TOP_K / SYNTH_KL_LOSS_COEF / SYNTH_QUERY_TEMPERATURE /
#                    SYNTH_TP / SYNTH_MAX_PROMPT_LENGTH / SYNTH_MAX_RESPONSE_LENGTH /
#                    SYNTH_LR / SYNTH_GPU_MEM_UTIL / SYNTH_RANDOM_Q_COEF / SYNTH_USE_SKILL_TYPE
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/process_cleanup_lib.sh"

# ========================== 解析位置参数 ==========================
exp_version="$1"
synthesizer_model_path="$2"
solver_model_path="$3"
synthesizer_training_steps="$4"
data_file="$5"
# 可选第 7 参；仅当 $7 未设置时才回退 SE_EMBEDDING_CACHE_PATH；显式传空字符串则保持为空（不传 --embedding-cache-path）
embedding_cache_path="${7-${SE_EMBEDDING_CACHE_PATH:-}}"
echo "[version]: ${exp_version}"
# ========================== 验证参数 ==========================
for var_name in exp_version synthesizer_model_path solver_model_path synthesizer_training_steps data_file; do
    if [ -z "${!var_name}" ]; then
        echo "Error: ${var_name} 不能为空"; exit 1
    fi
done
if ! [[ "$synthesizer_training_steps" =~ ^[0-9]+$ ]]; then
    echo "Error: synthesizer_training_steps 必须是数字，当前值: $synthesizer_training_steps"; exit 1
fi
# 第4参 = PPO 基座步数 T；offline 时另算 driver --steps = T×mult
SYNTH_PPO_STEPS="${synthesizer_training_steps}"
[ -d "$synthesizer_model_path" ] || { echo "Error: synthesizer_model_path 不存在: $synthesizer_model_path"; exit 1; }
[ -d "$solver_model_path" ]      || { echo "Error: solver_model_path 不存在: $solver_model_path"; exit 1; }
[ -f "$data_file" ]              || { echo "Error: data_file 不存在: $data_file"; exit 1; }
storage_path=${SYNTHESIZER_PATH_DIR}/${exp_version}
CKPTS_DIR=${storage_path}/ckpts/
tensorboard_path=${TENSORBOARD_PATH}/Synthesizer-${EXP_NAME}-${exp_version}
mkdir -p ${CKPTS_DIR} ${tensorboard_path} 
export TENSORBOARD_DIR=${tensorboard_path}

echo "[路径配置] 工作目录: ${SE_WORKING_DIR}"
echo "[路径配置] 存储路径: ${storage_path}"

# ========================== GPU 分配 ==========================
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
ROLLOUT_BASE_PORT="${SE_ROLLOUT_BASE_PORT:-8760}"
# 仅用于启动 vLLM 时 curl /health 的最长轮询（秒），与 reward 里请求 /rollout 的超时无关。
# 后者由 SynthsizerRewardManager.rollout_request_timeout（默认约 2000s，可用
# SYNTH_ROLLOUT_REQUEST_TIMEOUT 覆盖）控制，见 skill_src/reward_manager.py。
MAX_WAIT=300

echo "[GPU配置] 训练/ reward 阶段 — Synthesizer GPUs: ${SYNTH_GPU_IDS} (${N_SYNTH_GPUS} 张)"
echo "[GPU配置] 训练/ reward 阶段 — Rollout GPUs:     ${ROLLOUT_GPU_IDS} (${N_ROLLOUT_SERVERS} 张)"
echo "[GPU配置] Offline 阶段将使用全机: ${SE_GPU_IDS} (共 ${SE_N_GPUS} 张)"

export SE_N_GPUS_SAVED="${SE_N_GPUS}"
export SE_GPU_IDS_SAVED="${SE_GPU_IDS}"

OFFLINE_WORK_DIR="${storage_path}"
OFFLINE_MERGE_DIR="${OFFLINE_WORK_DIR}/merged"
mkdir -p "${OFFLINE_WORK_DIR}" "${OFFLINE_MERGE_DIR}"

resolve_merged_train_data_file() {
    TRAIN_DATA_FILE="${OFFLINE_MERGE_DIR}/train_data.parquet"
    [ -f "${TRAIN_DATA_FILE}" ] || TRAIN_DATA_FILE="${OFFLINE_MERGE_DIR}/train_data.jsonl"
}

synth_skip_offline_is_true() {
    case "${SE_SYNTH_SKIP_OFFLINE:-}" in
        1|true|TRUE|True|yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

RUN_OFFLINE=1
if synth_skip_offline_is_true; then
    resolve_merged_train_data_file
    if [ ! -f "${TRAIN_DATA_FILE}" ]; then
        echo "Error: SE_SYNTH_SKIP_OFFLINE=1 但未找到 merged 训练数据，需要 ${OFFLINE_MERGE_DIR}/train_data.parquet 或 train_data.jsonl" >&2
        exit 1
    fi
    echo "[SE_SYNTH_SKIP_OFFLINE] 跳过 Step1–2（offline），使用已有训练数据: ${TRAIN_DATA_FILE}"
    RUN_OFFLINE=0
fi

# ===========================================================================
# Step 1: 全卡启动 rollout（仅用于 offline）+ 健康检查
# ===========================================================================
if [ "${RUN_OFFLINE}" -eq 1 ]; then
echo ""
echo "========== Step 1: 全卡 rollout（offline）=========="

export SE_N_GPUS="${SE_N_GPUS_SAVED}"
export SE_GPU_IDS="${SE_GPU_IDS_SAVED}"
export SE_ROLLOUT_N_SERVERS="${SE_N_GPUS_SAVED}"
export ROLLOUT_SERVER_MODEL="${solver_model_path}"
export SE_ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT}"
unset SE_ROLLOUT_SERVER_URLS
unset SE_ROLLOUT_PORTS

bash "${SCRIPT_DIR}/start_rollout_servers.sh" --model "${solver_model_path}" &
OFFLINE_ROLLOUT_PID=$!
echo "[offline] start_rollout_servers 父 shell PID: ${OFFLINE_ROLLOUT_PID}"

for ((i=0; i<SE_N_GPUS_SAVED; i++)); do
    port=$((ROLLOUT_BASE_PORT + i))
    waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            echo "  [offline] server ${i} (port ${port}) 就绪 (${waited}s)"
            break
        fi
        sleep 5; waited=$((waited + 5))
    done
    if [ $waited -ge $MAX_WAIT ]; then
        echo "Error: [offline] rollout server ${i} (port ${port}) 启动超时"; exit 1
    fi
done

# ===========================================================================
# Step 2: 离线 rollout
# ===========================================================================
echo ""
echo "========== Step 2: 离线 rollout =========="

SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER="${SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER:-${SE_OFFLINE_ROLLOUT_STEPS:-2}}"
export SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER

# offline need = (T×mult) × batch
OFFLINE_DRIVER_STEPS=$(( SYNTH_PPO_STEPS * SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER ))
OFFLINE_DRIVER_BATCH="${SE_OFFLINE_ROLLOUT_BATCH_SIZE:-${SYNTH_BATCH_SIZE:-16}}"
_OFFLINE_NEED=$(( OFFLINE_DRIVER_STEPS * OFFLINE_DRIVER_BATCH ))

echo "  data_file: ${data_file}"
echo "  PPO 训练步数 T=${SYNTH_PPO_STEPS}  (trainer.total_training_steps 与此一致)"
echo "  offline driver: --steps ${OFFLINE_DRIVER_STEPS} (= T×mult)  --batch-size ${OFFLINE_DRIVER_BATCH}  (need target=${_OFFLINE_NEED} rows; mult=${SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER})"
echo "  rollout_n=${SE_OFFLINE_ROLLOUT_N}"
# 默认不 reset，使 train_cursor_state.json / data_cursor.txt 跨次运行向前消费；仅 SE_OFFLINE_RESET_STATE=1 时从 cursor=0 重算
SE_OFFLINE_RESET_STATE="${SE_OFFLINE_RESET_STATE:-0}"

cd "${SE_WORKING_DIR}"
# driver 用 SE_ROLLOUT_BASE_PORT + SE_ROLLOUT_N_SERVERS 解析各 server URL（须与 Step 1 全卡数一致）
export SE_ROLLOUT_N_SERVERS="${SE_N_GPUS_SAVED}"
offline_cmd=(
    python3 -m "${SE_CODE_MODULE}.solver_offline_driver" run
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
resolve_merged_train_data_file
[ -f "${TRAIN_DATA_FILE}" ] || { echo "Error: 未找到训练数据（merged 下无 train_data.parquet / train_data.jsonl）"; exit 1; }

# ===========================================================================
# offline 与全卡 vLLM 全部结束，避免占显存
# ===========================================================================
echo ""
echo "========== 结束 offline: pkill python，释放全卡 rollout =========="
pkill python 2> /dev/null || true
if kill -0 "${OFFLINE_ROLLOUT_PID}" 2> /dev/null; then
    echo "  结束 start_rollout_servers 父进程 (PID ${OFFLINE_ROLLOUT_PID})"
    kill "${OFFLINE_ROLLOUT_PID}" 2> /dev/null || true
    wait "${OFFLINE_ROLLOUT_PID}" 2> /dev/null || true
fi
sleep 3
OFFLINE_ROLLOUT_PID=""

fi

# ===========================================================================
# Step 3: 仅后半卡再启 rollout（供 reward）— 与 Step 4 RL 时间重叠
# ===========================================================================
echo ""
echo "========== Step 3: 半卡启动 rollout（reward 评估用）+ Step 4 RL 并行起 =========="

export SE_N_GPUS="${N_ROLLOUT_SERVERS}"
export SE_GPU_IDS="${ROLLOUT_GPU_IDS}"
export SE_ROLLOUT_N_SERVERS="${N_ROLLOUT_SERVERS}"
export ROLLOUT_SERVER_MODEL="${solver_model_path}"
export SE_ROLLOUT_BASE_PORT="${ROLLOUT_BASE_PORT}"
unset SE_ROLLOUT_SERVER_URLS
unset SE_ROLLOUT_PORTS

ROLLOUT_SERVER_PID=""
cleanup_rollout() {
    if [ -n "${ROLLOUT_SERVER_PID}" ]; then
        echo "清理 reward rollout (PID: ${ROLLOUT_SERVER_PID})..."
        kill "${ROLLOUT_SERVER_PID}" 2> /dev/null || true
        wait "${ROLLOUT_SERVER_PID}" 2> /dev/null || true
    fi
    # 仅 kill 父 shell 时，子进程里的 vLLM/Ray 常仍占显存；与 main_o 里 retriever（常同用一张卡）冲突
    echo "清理: pkill python，释放 reward rollout / Ray / main_synthesizer 侧 vLLM..."
    pkill python 2> /dev/null || true
    sleep 3
    pkill python 2> /dev/null || true
    sleep 3
}
trap cleanup_rollout EXIT

[ -f "${TRAIN_DATA_FILE}" ] || { echo "Error: 未找到训练数据"; exit 1; }
echo "训练数据: ${TRAIN_DATA_FILE}"

# 从环境变量读取超参数（由 main.sh export）
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
micro_batch_size_per_gpu=$((ppo_mini_batch_size / 8))
if [ "${micro_batch_size_per_gpu}" -lt 1 ]; then
    micro_batch_size_per_gpu=1
fi

echo ""
echo "  后半卡: 起 reward rollout vLLM；前半卡: 与下方 /health 等待重叠启动 main_synthesizer"
echo "  Synthesizer GPUs: ${SYNTH_GPU_IDS} (${N_SYNTH_GPUS} 张)  batch=${batch_size}  kl=${kl_loss_coef}  temp=${query_temperature}"

bash "${SCRIPT_DIR}/start_rollout_servers.sh" --model "${solver_model_path}" &
ROLLOUT_SERVER_PID=$!
echo "  reward rollout 父 shell PID: ${ROLLOUT_SERVER_PID}"

CUDA_VISIBLE_DEVICES=${SYNTH_GPU_IDS} python3 -m ${SE_CODE_MODULE}.main_synthesizer \
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
echo "  main_synthesizer PID: ${TRAINING_PID}（与 reward /health 阶段并行，节省 wall time）"

echo ""
echo "========== 等待各 reward server /health（此期间 RL 已在初始化）=========="
TRAIN_REAPED=0
for ((i=0; i<N_ROLLOUT_SERVERS; i++)); do
    port=$((ROLLOUT_BASE_PORT + i))
    waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if [ "${TRAIN_REAPED}" -eq 0 ] && ! kill -0 "${TRAINING_PID}" 2> /dev/null; then
            wait "${TRAINING_PID}" 2> /dev/null || true
            TRAINING_EXIT_CODE=$?
            TRAIN_REAPED=1
            if [ "$TRAINING_EXIT_CODE" -ne 0 ]; then
                echo "Error: main_synthesizer 在 reward 全就绪前失败, exit=${TRAINING_EXIT_CODE}" >&2
                exit "$TRAINING_EXIT_CODE"
            fi
            echo "  [info] main_synthesizer 已正常结束（快于部分 /health），跳过后续 /health 与末次 wait"
            break 2
        fi
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            echo "  [reward] server ${i} (port ${port}) 就绪 (${waited}s)"
            break
        fi
        sleep 5; waited=$((waited + 5))
    done
    if [ $waited -ge $MAX_WAIT ]; then
        echo "Error: [reward] rollout server ${i} (port ${port}) 启动超时"; exit 1
    fi
done

if [ "${TRAIN_REAPED}" -eq 0 ]; then
    echo "  全部 reward /health 已通过；等待 Synthesizer RL 跑完剩余步..."
    wait $TRAINING_PID
    TRAINING_EXIT_CODE=$?
else
    TRAINING_EXIT_CODE=0
fi

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "Synthesizer 训练失败，退出码: $TRAINING_EXIT_CODE"
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "=============================================="
echo "  ${exp_name} Synthesizer 训练完成"
echo "  model path: ${CKPTS_DIR}"
echo "=============================================="
