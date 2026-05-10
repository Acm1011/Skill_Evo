#!/usr/bin/env bash
# 训练入口（无 Synthesizer 训练版）：Synthesizer(reward-only) → memory(after_sync) → Solver → memory(after_solver)，按 V1..Vn 展开（n = skill_evo_num_rounds）。
# SE_RESUME=1|true|yes：若已有对应当前步数的 Synthesizer/Solver checkpoint 与 memory 产物，则跳过该子步骤。
#  Synthesizer：若已有 global_step_* 最终 ckpt 则整段不跑；若无 ckpt 但 Synthesizer/<Vn>/merged/train_data.parquet（或
#  train_data.jsonl）已存在，则仅跳过 offline（由 SE_SYNTH_SKIP_OFFLINE=1 传入 Synthesizer.sh），仍跑 reward+RL。缺 memory
#  仍跑 after_sync/after_solver。续训步数须与已存在 ckpt 步数一致时 skip 最可靠。Synthesizer 内 verl 为 resume_mode=auto。

set -xeuo pipefail
# 获取脚本所在目录
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SE_RAY_TEMP_ROOT="${SE_RAY_TEMP_ROOT:-/home/ycy/sdi/tmp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# 路径与模型、数据
# =============================================================================
dir="${SE_BASE_DIR:-/home/ycy/sdi/}"
SE_MODEL_DIR="${SE_MODEL_DIR:-${dir}/models}"
SE_DATA_DIR="${SE_DATA_DIR:-${dir}/data}"
export SE_MODEL_DIR SE_DATA_DIR
model_dir="${SE_MODEL_DIR}"
data_dir="${SE_DATA_DIR}"
export data_dir

base_model_name="${SE_BASE_MODEL_NAME:-Qwen3-4B-Instruct-2507}"
base_model_path=${SE_MODEL_DIR}/${base_model_name}
data_name="${SE_DATA_NAME:-DeepMath-103K}"
data_file=${SE_DATA_DIR}/${data_name}.jsonl

project_name="${SE_PROJECT_NAME:-Skill_Evo}"
export project_name
exp_name=data_${data_name}_model_${base_model_name}_v3
export exp_name
variant="${exp_name}"
SE_SKILL_SAVED_ROOT="${SE_SKILL_SAVED_ROOT:-/home/ycy/sdi/skill_saved}"
# 产物：skill_saved/<项目名,如 Skill_Evo>/<原实验名 data_*_model_* >/Synthesizer|Solver|Memory|...

WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
saved_results_dir="${SE_SAVED_RESULTS_DIR:-${SE_SKILL_SAVED_ROOT}/${project_name}/${exp_name}}"
synthesizer_path_dir="${SE_SYNTHESIZER_DIR:-${saved_results_dir}/Synthesizer}"
solver_path_dir="${SE_SOLVER_DIR:-${saved_results_dir}/Solver}"
memory_path_dir="${SE_MEMORY_DIR:-${saved_results_dir}/Memory}"
log_path_dir="${SE_LOG_DIR:-${saved_results_dir}/logs}"
tensorboard_dir="${SE_TENSORBOARD_DIR:-${saved_results_dir}/tensorboard_log}"
TENSORBOARD_PATH="${TENSORBOARD_PATH:-${tensorboard_dir}}"
export TENSORBOARD_PATH tensorboard_dir

mkdir -p ${saved_results_dir} ${synthesizer_path_dir} ${solver_path_dir} ${memory_path_dir} ${log_path_dir} ${tensorboard_dir}
export SE_SAVED_RESULTS_DIR="${saved_results_dir}"
export SYNTHESIZER_PATH_DIR=${synthesizer_path_dir}
export SOLVER_PATH_DIR=${solver_path_dir}
# 与 main.sh / evaluation 脚本中的 SE_* 一致，便于子进程解析路径
export SE_Synthsizer_DIR="${synthesizer_path_dir}"
export SE_SOLVER_DIR="${solver_path_dir}"
export MEMORY_PATH_DIR=${memory_path_dir}
export LOG_PATH_DIR=${log_path_dir}
export EXP_NAME=${exp_name}
export WORKING_DIR=${WORKING_DIR}
export data_file
export SE_DATA_FILE="${data_file}"

# =============================================================================
# 通用：代码模块
# =============================================================================
export SE_CODE_MODULE="${SE_CODE_MODULE:-skill_src}"

# =============================================================================
# GPU 拓扑（Synthesizer.sh 需要 SE_N_GPUS / SE_GPU_IDS；请与 retriever 用卡错开避免争用）
# =============================================================================
SE_N_GPUS="${SE_N_GPUS:-4}"
SE_GPU_IDS="${SE_GPU_IDS:-0,1,2,3}"
export SE_N_GPUS SE_GPU_IDS

# =============================================================================
# Offline Rollout 参数
# 说明：SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER 只放大 offline driver 的 --steps（基座 T×mult），
#   need≈steps×batch。server 侧 vLLM SamplingParams.n 与 --rollout-n（SE_OFFLINE_ROLLOUT_N）一致，不再单独倍率。
#   verl total_training_steps 见下方 synthesizer_training_steps（基座 T+2）。
#   仍可用旧名 SE_OFFLINE_ROLLOUT_STEPS 作为 BATCH_MULTIPLIER 的缺省来源。
# =============================================================================
export SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER="${SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER:-${SE_OFFLINE_ROLLOUT_STEPS:-1}}"
export SE_OFFLINE_ROLLOUT_BATCH_SIZE="${SE_OFFLINE_ROLLOUT_BATCH_SIZE:-128}"
# 线下游标仅当 SE_OFFLINE_RESET_STATE=1 时从 0 重置（见 Synthesizer.sh / solver_offline_driver）
export SE_OFFLINE_RESET_STATE="${SE_OFFLINE_RESET_STATE:-0}"
export SE_OFFLINE_ROLLOUT_N="${SE_OFFLINE_ROLLOUT_N:-4}"
export SE_OFFLINE_NUM_RANDOM_Q="${SE_OFFLINE_NUM_RANDOM_Q:-4}"
export SE_OFFLINE_SKILL_TYPE="${SE_OFFLINE_SKILL_TYPE:-skill_generation_v1}"

# =============================================================================
# Synthesizer RL 训练超参数
# 基座 PPO 步数 T = SE_SYNTHESIZER_TRAINING_STEPS（= SE_SYNTHESIZER_TRAINING_STEPS_BASE）；
# 实际传给 verl 的 total_training_steps = synthesizer_training_steps = T+2（见下），比「只跑 T 步」多 2 步，总
#  reward/rollout 时间约 ×(T+2)/T。若与别机「20 分钟跑完」对齐，可改为一律用 T 或统一 base+2。
# =============================================================================
SE_SYNTHESIZER_TRAINING_STEPS_BASE="${SE_SYNTHESIZER_TRAINING_STEPS:-20}"
export SE_SYNTHESIZER_TRAINING_STEPS_BASE
SE_OFFLINE_ROLLOUT_DRIVER_STEPS=$(( SE_SYNTHESIZER_TRAINING_STEPS_BASE * SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER ))
export SE_OFFLINE_ROLLOUT_DRIVER_STEPS
synthesizer_training_steps="$((SE_SYNTHESIZER_TRAINING_STEPS_BASE+1))"
export synthesizer_training_steps
export SE_SYNTHESIZER_STEPS="${synthesizer_training_steps}"
export SE_SYNTHESIZER_TRAINING_STEPS="${SE_SYNTHESIZER_TRAINING_STEPS_BASE}"

export SYNTH_BATCH_SIZE="${SYNTH_BATCH_SIZE:-128}"
export SYNTH_ROLLOUT_QUERY_NUM="${SYNTH_ROLLOUT_QUERY_NUM:-4}"
export SYNTH_QUERY_TOP_P="${SYNTH_QUERY_TOP_P:-0.99}"
export SYNTH_QUERY_TOP_K="${SYNTH_QUERY_TOP_K:--1}"
export SYNTH_KL_LOSS_COEF="${SYNTH_KL_LOSS_COEF:-0.01}"
export SYNTH_QUERY_TEMPERATURE="${SYNTH_QUERY_TEMPERATURE:-0.7}"
export SYNTH_TP="${SYNTH_TP:-1}"
export SYNTH_MAX_PROMPT_LENGTH="${SYNTH_MAX_PROMPT_LENGTH:-8192}"
export SYNTH_MAX_RESPONSE_LENGTH="${SYNTH_MAX_RESPONSE_LENGTH:-512}"
export SYNTH_GPU_MEM_UTIL="${SYNTH_GPU_MEM_UTIL:-0.60}"
export SYNTH_RANDOM_Q_COEF="${SYNTH_RANDOM_Q_COEF:-0.5}"
export SYNTH_USE_SKILL_TYPE="${SYNTH_USE_SKILL_TYPE:-skill_use_v1}"
export SYNTH_SOLVER_ROLLOUT_MAX_WORKERS="${SYNTH_SOLVER_ROLLOUT_MAX_WORKERS:-16}"
# 离线 rollout：设为 >0（如 32/64）时按连续样本切多笔 HTTP POST 并行；0=每台 GPU 只发一单大包（并行≈GPU 数）
export SE_OFFLINE_ROLLOUT_HTTP_CHUNK_SIZE="${SE_OFFLINE_ROLLOUT_HTTP_CHUNK_SIZE:-0}"

# =============================================================================
# Solver 训练超参数（供 solver.sh 子进程；tensorboard 路径与 exp 名）
# =============================================================================
solver_retrain_steps="${SE_SOLVER_RETRAIN_STEPS:-40}"
export solver_retrain_steps
# solver.sh 内 trainer.total_training_steps = 传参 +5（DAPO/verl 约定）；resume 与 prev ckpt 路径须与此一致
solver_training_total_steps="$((solver_retrain_steps + 5))"
export solver_training_total_steps
prev_synth_ckpt_step="${SE_SYNTHESIZER_PREV_CKPT_STEP:-${SE_SYNTHESIZER_TRAINING_STEPS_BASE}}"
prev_solver_ckpt_step="${SE_SOLVER_PREV_CKPT_STEP:-${solver_retrain_steps}}"
export prev_synth_ckpt_step prev_solver_ckpt_step
solver_batch_size="${SE_SOLVER_BATCH_SIZE:-128}"
rollout_n="${SE_SOLVER_ROLLOUT_N:-4}"
export SE_SOLVER_BATCH_SIZE="${solver_batch_size}"
export solver_batch_size rollout_n

# memory ingest：同一题多 traj 时只入 reward 最高一条；且 utility(reward) >= 该值才入池（默认 0）
SE_MEMORY_MIN_UTILITY="${SE_MEMORY_MIN_UTILITY:-0}"
export SE_MEMORY_MIN_UTILITY

# =============================================================================
# Syn↔Solver 交替进化：每轮含 Synthesizer → after_sync → Solver → after_solver，共 n 轮得 V1..Vn
# =============================================================================
skill_evo_num_rounds="${SE_SKILL_EVO_NUM_ROUNDS:-10}"
export skill_evo_num_rounds

# =============================================================================
# 训练后基准评测（eval_single_math_data.sh）与 retriever（memory 同步前需 HTTP）
# =============================================================================
solver_eval_step=40
solver_eval_temperature=0.6

RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8766}"
SE_RETRIEVER_EMBEDDING_MODEL="${SE_RETRIEVER_EMBEDDING_MODEL:-/home/xzs/data/model/Qwen3-Embedding-0.6B}"
RETRIEVER_CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES:-1}"
RETRIEVER_TENSOR_PARALLEL_SIZE="${RETRIEVER_TENSOR_PARALLEL_SIZE:-1}"
# vLLM 要求 free >= utilization*总显存；0.2*79GiB≈15.8GiB 在残留进程时易踩线，0.15 更稳（可 env 覆盖）
RETRIEVER_GPU_MEMORY_UTILIZATION="${RETRIEVER_GPU_MEMORY_UTILIZATION:-0.15}"
RETRIEVER_INSTRUCT_TASK="${RETRIEVER_INSTRUCT_TASK:-Given a question, retrieve relevant skills that help answer it}"
RETRIEVER_IDLE_TIMEOUT="${RETRIEVER_IDLE_TIMEOUT:-300}"
export RETRIEVER_HOST RETRIEVER_PORT
export SE_RETRIEVER_EMBEDDING_MODEL
export RETRIEVER_CUDA_VISIBLE_DEVICES
export RETRIEVER_TENSOR_PARALLEL_SIZE
export RETRIEVER_GPU_MEMORY_UTILIZATION
export RETRIEVER_INSTRUCT_TASK
export RETRIEVER_IDLE_TIMEOUT
export SE_RETRIEVER_URL="http://${RETRIEVER_HOST}:${RETRIEVER_PORT}"
# SkillManager 调用 retriever_server /rank、/health 的 HTTP 读超时（秒）；prepare_solver_skills 逐条检索时建议 ≥300
export SE_RETRIEVER_TIMEOUT="${SE_RETRIEVER_TIMEOUT:-300}"

# start_retriever_server.sh 也读 RETRIEVER_* / SE_RETRIEVER_*
RETRIEVER_MAX_WAIT_S="${RETRIEVER_MAX_WAIT_S:-300}"

# =============================================================================
# Resume：从已有 ckpt / memory / merged 训练数据 续跑，跳过已完成阶段
#   SE_RESUME=1|true|yes 时启用；Synthesizer ckpt global_step_* = synthesizer_training_steps（T+2）；
#   Solver ckpt global_step_* = solver_training_total_steps（= SE_SOLVER_RETRAIN_STEPS+5，与 solver.sh 一致）
#   无 Synthesizer 最终 ckpt 但已有 merged/train_data.(parquet|jsonl) 时：跳过 offline、仅跑 RL（见 se_synth_merged_train_ready）
#   memory: memory_after_syn_vN.jsonl / memory_after_sol_vN.jsonl（N 为版本号 1,2,…）
# =============================================================================
SE_RESUME="${SE_RESUME:-true}"
export SE_RESUME

se_resume_is_true() {
    case "${SE_RESUME}" in
        1|true|TRUE|True|yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

# 该轮 Synthesizer reward 已产出（用于 after_sync 初始化 utility）
se_synth_reward_ready() {
    local ev="$1"
    local d="${synthesizer_path_dir}/${ev}/reward_info"
    ls "${d}"/exp_data_step_*.jsonl >/dev/null 2>&1
}

# 若已存在可用 reward 或 memory 产物则跳过本轮 Synthesizer(reward-only)
se_resume_skip_synth() {
    local ev="$1"
    se_resume_is_true || return 1
    local n="${ev#V}"
    se_synth_reward_ready "${ev}" || [ -f "${memory_path_dir}/memory_after_syn_v${n}.jsonl" ]
}

# 该轮 Synthesizer 的 offline 已产出合并训练数据（无最终 ckpt 时用于跳过 offline、仅跑 RL）
se_synth_merged_train_ready() {
    local ev="$1"
    [ -f "${synthesizer_path_dir}/${ev}/merged/train_data.parquet" ] \
        || [ -f "${synthesizer_path_dir}/${ev}/merged/train_data.jsonl" ]
}

# 若已存在则跳过后续 retriever+after_sync
se_resume_skip_mem_sync() {
    local ev="$1"
    se_resume_is_true || return 1
    local n="${ev#V}"
    [ -f "${memory_path_dir}/memory_after_syn_v${n}.jsonl" ]
}

# 若已存在则跳过后续 Solver 训练
se_resume_skip_solver() {
    local ev="$1"
    se_resume_is_true || return 1
    [ -d "${solver_path_dir}/${ev}/ckpts/global_step_${prev_solver_ckpt_step}/actor/huggingface" ]
}

# 若已存在则跳过后续 after_solver
se_resume_skip_mem_solver() {
    local ev="$1"
    se_resume_is_true || return 1
    local n="${ev#V}"
    [ -f "${memory_path_dir}/memory_after_sol_v${n}.jsonl" ]
}

resolve_prev_ckpt_hf_dir() {
    local model_root="$1"
    local ev="$2"
    local preferred_step="$3"
    local preferred="${model_root}/${ev}/ckpts/global_step_${preferred_step}/actor/huggingface"
    if [ -d "${preferred}" ]; then
        echo "${preferred}"
        return 0
    fi
    local ckpt_root="${model_root}/${ev}/ckpts"
    local best_step=""
    local d bn s
    for d in "${ckpt_root}"/global_step_*; do
        [ -d "${d}" ] || continue
        bn="${d##*/}"
        s="${bn#global_step_}"
        if [[ "${s}" =~ ^[0-9]+$ ]]; then
            if [ -z "${best_step}" ] || [ "${s}" -gt "${best_step}" ]; then
                best_step="${s}"
            fi
        fi
    done
    if [ -n "${best_step}" ] && [ -d "${ckpt_root}/global_step_${best_step}/actor/huggingface" ]; then
        echo "${ckpt_root}/global_step_${best_step}/actor/huggingface"
        return 0
    fi
    return 1
}

# 新版本 Vk：若本版本目录下尚无游标，从 V(k-1) 继承，使 Synthesizer/Solver 在原始语料上各自继续向前取数（与 data_cursor.txt / train_cursor_state 一致）
se_inherit_data_cursors_from_prev() {
    local ev="$1"
    local prev_iter="$2"
    if [ "${prev_iter}" -lt 1 ]; then
        return 0
    fi
    local pev="V${prev_iter}"
    if [ ! -f "${synthesizer_path_dir}/${ev}/train_cursor_state.json" ] && [ -f "${synthesizer_path_dir}/${pev}/train_cursor_state.json" ]; then
        mkdir -p "${synthesizer_path_dir}/${ev}"
        cp -a "${synthesizer_path_dir}/${pev}/train_cursor_state.json" "${synthesizer_path_dir}/${ev}/"
        if [ -f "${synthesizer_path_dir}/${pev}/data_cursor.txt" ]; then
            cp -a "${synthesizer_path_dir}/${pev}/data_cursor.txt" "${synthesizer_path_dir}/${ev}/"
        fi
        echo "[cursor] 已从 ${pev} 继承 Synthesizer 游标至 ${ev}（train_cursor_state.json + data_cursor.txt）"
    fi
    if [ ! -f "${solver_path_dir}/${ev}/data_cursor.txt" ] && [ -f "${solver_path_dir}/${pev}/data_cursor.txt" ]; then
        mkdir -p "${solver_path_dir}/${ev}"
        cp -a "${solver_path_dir}/${pev}/data_cursor.txt" "${solver_path_dir}/${ev}/"
        echo "[cursor] 已从 ${pev} 继承 Solver data_cursor.txt 至 ${ev}"
    fi
}

# memory_func_after_sync：等待检索服务 /health
retriever_memory_sync_after_synth() {
    local ev="$1"
    echo "retriever 启动前：再次 pkill python（Synthesizer 的 EXIT 已清过一轮；此处防残留）..."
    pkill python 2> /dev/null || true
    sleep 4
    pkill python 2> /dev/null || true
    sleep 8
    echo "启动 retriever 服务 (memory 同步用)..."
    bash "${SCRIPT_DIR}/start_retriever_server.sh" &
    local _wait=0
    while [ "${_wait}" -lt "${RETRIEVER_MAX_WAIT_S}" ]; do
        if curl -sf "http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health" > /dev/null 2>&1; then
            echo "  retriever 已就绪 (http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health)"
            break
        fi
        sleep 2
        _wait=$((_wait + 2))
    done
    if [ "${_wait}" -ge "${RETRIEVER_MAX_WAIT_S}" ]; then
        echo "Error: retriever 在 ${RETRIEVER_MAX_WAIT_S}s 内未就绪" >&2
        pkill python 2> /dev/null || true
        return 1
    fi
    echo "更新 skill memory（after Synthesizer, ${ev}）并准备 Solver 数据..."
    bash "${SCRIPT_DIR}/memory_func_after_sync.sh" "${ev}" || {
        pkill python 2> /dev/null || true
        return 1
    }
    echo "关闭 retriever 相关 python 进程 (pkill python)..."
    pkill python 2> /dev/null || true
    return 0
}

function now() {
    date '+%Y-%m-%d-%H-%M'
}

# 将本脚本解析后的关键变量 + 当前全部已 export 的环境写入实验目录，便于复现实验
main_o_write_experiment_config() {
    local cfg="${saved_results_dir}/experiment_config.txt"
    local tmp="${cfg}.$$.tmp"
    {
        printf '%s\n' \
            '# skill_evo main_o — experiment_config.txt' \
            "# iso_time=$(date -Iseconds 2>/dev/null || date)" \
            "# host=$(hostname 2>/dev/null || echo unknown) user=$(id -un 2>/dev/null || echo unknown)" \
            "# cwd_at_launch=$(pwd)" \
            "# main_o=${SCRIPT_DIR}/main_o.sh" \
            ''
        printf '%s\n' '# ========== paths & identity =========='
        local __k
        for __k in dir SE_BASE_DIR SE_MODEL_DIR SE_DATA_DIR model_dir data_dir \
            base_model_name base_model_path data_name data_file \
            project_name exp_name variant WORKING_DIR \
            saved_results_dir SE_SAVED_RESULTS_DIR SE_SKILL_SAVED_ROOT \
            synthesizer_path_dir SYNTHESIZER_PATH_DIR SE_Synthsizer_DIR \
            solver_path_dir SOLVER_PATH_DIR SE_SOLVER_DIR \
            memory_path_dir MEMORY_PATH_DIR \
            log_path_dir LOG_PATH_DIR tensorboard_dir TENSORBOARD_PATH \
            SE_DATA_FILE SE_CODE_MODULE; do
            [[ -v $__k ]] && printf '%s=%q\n' "$__k" "${!__k}"
        done
        printf '\n%s\n' '# ========== GPUs & offline rollout =========='
        for __k in SE_N_GPUS SE_GPU_IDS \
            SE_OFFLINE_ROLLOUT_BATCH_MULTIPLIER SE_OFFLINE_ROLLOUT_BATCH_SIZE \
            SE_OFFLINE_RESET_STATE SE_OFFLINE_ROLLOUT_N SE_OFFLINE_NUM_RANDOM_Q SE_OFFLINE_SKILL_TYPE; do
            [[ -v $__k ]] && printf '%s=%q\n' "$__k" "${!__k}"
        done
        printf '\n%s\n' '# ========== Synthesizer =========='
        for __k in SE_SYNTHESIZER_TRAINING_STEPS_BASE SE_OFFLINE_ROLLOUT_DRIVER_STEPS \
            synthesizer_training_steps SE_SYNTHESIZER_STEPS SE_SYNTHESIZER_TRAINING_STEPS \
            SYNTH_BATCH_SIZE SYNTH_ROLLOUT_QUERY_NUM SYNTH_QUERY_TOP_P SYNTH_QUERY_TOP_K \
            SYNTH_KL_LOSS_COEF SYNTH_QUERY_TEMPERATURE SYNTH_TP \
            SYNTH_MAX_PROMPT_LENGTH SYNTH_MAX_RESPONSE_LENGTH SYNTH_GPU_MEM_UTIL \
            SYNTH_RANDOM_Q_COEF SYNTH_USE_SKILL_TYPE; do
            [[ -v $__k ]] && printf '%s=%q\n' "$__k" "${!__k}"
        done
        printf '\n%s\n' '# ========== Solver & evolution =========='
        for __k in solver_retrain_steps solver_training_total_steps SE_SOLVER_RETRAIN_STEPS solver_batch_size SE_SOLVER_BATCH_SIZE \
            rollout_n SE_SOLVER_ROLLOUT_N SE_MEMORY_MIN_UTILITY skill_evo_num_rounds SE_SKILL_EVO_NUM_ROUNDS \
            solver_eval_step solver_eval_temperature; do
            [[ -v $__k ]] && printf '%s=%q\n' "$__k" "${!__k}"
        done
        printf '\n%s\n' '# ========== retriever & resume =========='
        for __k in RETRIEVER_HOST RETRIEVER_PORT SE_RETRIEVER_URL SE_RETRIEVER_EMBEDDING_MODEL \
            RETRIEVER_CUDA_VISIBLE_DEVICES RETRIEVER_TENSOR_PARALLEL_SIZE \
            RETRIEVER_GPU_MEMORY_UTILIZATION RETRIEVER_INSTRUCT_TASK RETRIEVER_IDLE_TIMEOUT \
            RETRIEVER_MAX_WAIT_S SE_RESUME VLLM_WORKER_MULTIPROC_METHOD SE_RAY_TEMP_ROOT; do
            [[ -v $__k ]] && printf '%s=%q\n' "$__k" "${!__k}"
        done
        printf '\n%s\n' '# ========== shell options (set +o) =========='
        set +o
        printf '\n%s\n' '# ========== export -p (sorted, full snapshot) =========='
        export -p | LC_ALL=C sort
    } > "${tmp}" && mv -f "${tmp}" "${cfg}"
}

main_o_write_experiment_config
exec > >(tee -a "${log_path_dir}/main_${variant}-$(now).log") 2>&1

cd ${WORKING_DIR}
echo "实验参数已写入: ${saved_results_dir}/experiment_config.txt"
echo "========== 训练调度: 共 ${skill_evo_num_rounds} 轮 Syn↔Solver 进化 (V1..V${skill_evo_num_rounds}) =========="
if se_resume_is_true; then
    echo "[resume] SE_RESUME=true：已有 reward/Memory/Solver ckpt 的环节会跳过；Synthesizer 使用 reward-only 版本"
else
    echo "（非 resume 全流程跑；Synthesizer 使用 reward-only 版本）"
fi

for iter in $(seq 1 "${skill_evo_num_rounds}"); do
    exp_version="V${iter}"
    echo ""
    echo "########## 版本 ${exp_version} / V${skill_evo_num_rounds} ##########"

    se_inherit_data_cursors_from_prev "${exp_version}" $((iter - 1))

    if [ "${iter}" -eq 1 ]; then
        prev_synthesizer_model_path="${base_model_path}"
        prev_solver_model_path="${base_model_path}"
    else
        prev_synthesizer_model_path="${base_model_path}"
        prev_solver_model_path="$(resolve_prev_ckpt_hf_dir "${solver_path_dir}" "V$((iter - 1))" "${prev_solver_ckpt_step}")" || {
            echo "Error: 未找到上一轮 Solver checkpoint: ${solver_path_dir}/V$((iter - 1))/ckpts/global_step_*/actor/huggingface" >&2
            exit 1
        }
    fi

    if se_resume_skip_synth "${exp_version}"; then
        echo "[resume] 跳过 Synthesizer(reward-only) ${exp_version}（已有 reward_info 或 memory_after_syn）"
    elif se_resume_is_true && se_synth_merged_train_ready "${exp_version}"; then
        echo "[resume] Synthesizer(reward-only) ${exp_version}：已有 merged 训练数据，跳过 offline、仅跑 reward-only（SE_SYNTH_SKIP_OFFLINE=1）"
        SE_SYNTH_SKIP_OFFLINE=1 bash "${SCRIPT_DIR}/Synthesizer_reward_only.sh" \
            "${exp_version}" "${prev_synthesizer_model_path}" \
            "${prev_solver_model_path}" "${synthesizer_training_steps}" "${data_file}" || {
            echo "Error: Synthesizer(reward-only) ${exp_version} 执行失败" >&2
            exit 1
        }
    else
        echo "执行 Synthesizer(reward-only) (${exp_name} ${exp_version})..."
        bash "${SCRIPT_DIR}/Synthesizer_reward_only.sh" \
            "${exp_version}" "${prev_synthesizer_model_path}" \
            "${prev_solver_model_path}" "${synthesizer_training_steps}" "${data_file}" || {
            echo "Error: Synthesizer(reward-only) ${exp_version} 执行失败" >&2
            exit 1
        }
    fi

    if se_resume_skip_mem_sync "${exp_version}"; then
        echo "[resume] 跳过 retriever + memory after_sync ${exp_version}（已有 memory_after_syn_v${iter}.jsonl）"
    else
        retriever_memory_sync_after_synth "${exp_version}" || {
            echo "Error: memory_func_after_sync ${exp_version} 失败" >&2
            exit 1
        }
    fi

    if se_resume_skip_solver "${exp_version}"; then
        echo "[resume] 跳过 Solver ${exp_version}（已有 .../ckpts/global_step_${prev_solver_ckpt_step}/actor/huggingface）"
    else
        echo "训练 Solver (${exp_name} ${exp_version})..."
        bash "${SCRIPT_DIR}/solver.sh" "${exp_version}" "${prev_solver_model_path}" "${solver_retrain_steps}" || {
            echo "Error: Solver ${exp_version} 训练失败" >&2
            exit 1
        }
    fi

    if se_resume_skip_mem_solver "${exp_version}"; then
        echo "[resume] 跳过 memory after_solver ${exp_version}（已有 memory_after_sol_v${iter}.jsonl）"
    else
        echo "按 Solver reward 更新 skill utility（${exp_version}）..."
        bash "${SCRIPT_DIR}/memory_func_after_solver.sh" "${exp_version}" || {
            echo "Error: memory_func_after_solver ${exp_version} 失败" >&2
            exit 1
        }
    fi

    echo "版本 ${exp_version} 本阶段完成"
done

prepare_test_data_after_training() {
    local latest_n=-1
    local latest_mem=""
    local p bn n
    for p in "${memory_path_dir}"/memory_after_sol_v*.jsonl; do
        [ -f "${p}" ] || continue
        bn="$(basename "${p}")"
        if [[ "${bn}" =~ ^memory_after_sol_v([0-9]+)\.jsonl$ ]]; then
            n="${BASH_REMATCH[1]}"
            if [ "${n}" -gt "${latest_n}" ]; then
                latest_n="${n}"
                latest_mem="${p}"
            fi
        fi
    done
    if [ -z "${latest_mem}" ]; then
        echo "Error: 未找到 ${memory_path_dir}/memory_after_sol_v*.jsonl，无法准备测试数据" >&2
        return 1
    fi

    local test_input_1="${SE_PREPARE_TEST_INPUT_1:-/home/ycy/sdi/data/temp_data.jsonl}"
    local test_input_2="${SE_PREPARE_TEST_INPUT_2:-/home/ycy/sdi/data/greedy_data.jsonl}"
    local prepare_top_k="${SE_PREPARE_TEST_TOP_K:-3}"
    local default_prepare_out_dir="${SE_SKILL_SAVED_ROOT}/${project_name}/${exp_name}"
    local out_dir="${SE_PREPARE_TEST_OUTPUT_DIR:-${default_prepare_out_dir}}"
    local script="${WORKING_DIR}/skill_src/prepare_test_data.py"

    [ -f "${script}" ] || {
        echo "Error: 未找到 prepare 脚本: ${script}" >&2
        return 1
    }
    [ -f "${test_input_1}" ] || {
        echo "Error: 输入文件不存在: ${test_input_1}" >&2
        return 1
    }
    [ -f "${test_input_2}" ] || {
        echo "Error: 输入文件不存在: ${test_input_2}" >&2
        return 1
    }

    echo "训练后准备测试数据：memory=${latest_mem}, output_dir=${out_dir} (default=${default_prepare_out_dir})"
    echo "retriever 启动前：清理残留 python 进程..."
    pkill python 2> /dev/null || true
    sleep 4
    pkill python 2> /dev/null || true
    sleep 8

    echo "启动 retriever 服务 (prepare_test_data 用)..."
    bash "${SCRIPT_DIR}/start_retriever_server.sh" &
    local _wait=0
    while [ "${_wait}" -lt "${RETRIEVER_MAX_WAIT_S}" ]; do
        if curl -sf "http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health" > /dev/null 2>&1; then
            echo "  retriever 已就绪 (http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health)"
            break
        fi
        sleep 2
        _wait=$((_wait + 2))
    done
    if [ "${_wait}" -ge "${RETRIEVER_MAX_WAIT_S}" ]; then
        echo "Error: retriever 在 ${RETRIEVER_MAX_WAIT_S}s 内未就绪" >&2
        pkill python 2> /dev/null || true
        return 1
    fi

    python3 "${script}" \
        --memory-jsonl "${latest_mem}" \
        --inputs "${test_input_1}" "${test_input_2}" \
        --output-dir "${out_dir}" \
        --top-k "${prepare_top_k}" \
        --write-jsonl \
        --write-parquet || {
        echo "Error: prepare_test_data.py 执行失败" >&2
        pkill python 2> /dev/null || true
        return 1
    }

    echo "关闭 retriever 相关 python 进程 (pkill python)..."
    pkill python 2> /dev/null || true
    return 0
}

prepare_test_data_after_training || {
    echo "Error: 训练后自动准备测试数据失败" >&2
    exit 1
}

echo "所有训练完成！"
# echo "开始评估..."
# dataset="${SE_EVAL_DATASET:-AIME24}"
# eval_script="${SE_EVAL_SCRIPT:-${WORKING_DIR}/evaluation/eval_single_math_data.sh}"
# bash "$eval_script" "${variant}"  "${solver_eval_step}" "${skill_evo_num_rounds}" "${base_model_name}" "${dataset}" || {
#     echo "Error: 评估失败"
#     exit 1
# }
