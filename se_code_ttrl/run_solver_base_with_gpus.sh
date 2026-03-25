#!/usr/bin/env bash
# =============================================================================
# run_solver_base_with_gpus.sh - GPU-aware launcher for Base Model TTRL training
# =============================================================================
#
# 用法:
#   ./run_solver_base_with_gpus.sh <n_gpus> [--eval] [--eval_steps "step1,step2,..."]
#   ./run_solver_base_with_gpus.sh 4           # 使用 4 张 GPU 进行训练
#   ./run_solver_base_with_gpus.sh 4 --eval    # 训练后自动评测所有 checkpoint
#   ./run_solver_base_with_gpus.sh 4 --eval --eval_steps "20,40,60"  # 评测指定 steps
#
# 功能:
#   1. 检测当前服务器 GPU 状态（显存占用、利用率）
#   2. 等待直到空闲 GPU 数量满足 n_gpus
#   3. 选择空闲 GPU 并设置环境变量
#   4. 执行 solver_base 训练
#   5. (可选) 训练完成后自动评测
#
# 环境变量 (可覆盖默认值):
#   SB_BASB_DIR           - 基础目录 (默认: /home/ycy/data1)
#   SB_MODEL_DIR          - 模型目录 (默认: ${SB_BASB_DIR}/models)
#   SB_MODEL_NAME         - 基础模型名称 (默认: Qwen3-4B-Base)
#   SB_DATA_DIR           - 数据目录 (默认: ${SB_BASB_DIR}/data)
#   SB_SAVED_RESULTS_DIR  - 结果保存目录 (默认: /home/ycy/data3/ttrl_saved)
#   SB_EXP_NAME           - 实验名称 (默认: ttrl_${SB_MODEL_NAME}_bsz128)
#   SB_TRAIN_FILE         - 训练数据文件
#   SB_TEST_FILE          - 测试数据文件
#   SB_TRAIN_BATCH_SIZE   - 训练 batch size (默认: 128)
#   SB_N_RESP_PER_PROMPT  - 每个 prompt 的响应数 (默认: 8)
#   SB_TOTAL_EPOCHS       - 训练轮数 (默认: 4)
#   SB_SAVE_FREQ          - 保存频率 (默认: 20)
#   SB_TEST_FREQ          - 测试频率 (默认: 10)
#
# =============================================================================

set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# 参数解析
# =============================================================================

N_GPUS=""
DO_EVAL=false
EVAL_STEPS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval)
            DO_EVAL=true
            shift
            ;;
        --eval_steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        *)
            if [[ -z "$N_GPUS" ]]; then
                N_GPUS="$1"
            fi
            shift
            ;;
    esac
done

# =============================================================================
# 配置参数
# =============================================================================

# GPU 空闲判定阈值
GPU_MEMORY_THRESHOLD_MB="${GPU_MEMORY_THRESHOLD_MB:-4000}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-5}"

# 轮询配置
POLL_INTERVAL="${POLL_INTERVAL:-1800}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-48}"

# =============================================================================
# 路径配置 - 通过环境变量统一管理
# =============================================================================

SB_BASB_DIR="${SB_BASB_DIR:-/home/ycy/data1}"
SB_PROJECT_NAME="${SB_PROJECT_NAME:-Self-evolving-Agent}"
SB_CODE_MODULE="${SB_CODE_MODULE:-se_code_ttrl}"

# 派生路径
SB_WORKING_DIR="${SB_BASB_DIR}/${SB_PROJECT_NAME}"
SB_MODEL_DIR="${SB_MODEL_DIR:-${SB_BASB_DIR}/models}"
SB_MODEL_NAME="${SB_MODEL_NAME:-Qwen3-4B}"
SB_DATA_DIR="${SB_DATA_DIR:-${SB_BASB_DIR}/data}"
SB_SAVED_RESULTS_DIR="${SB_SAVED_RESULTS_DIR:-/home/ycy/data3/ttrl_saved}"
SB_TENSORBOARD_DIR="${SB_SAVED_RESULTS_DIR}/tensorboard_log"

# 训练相关配置
SB_EXP_NAME="${SB_EXP_NAME:-ttrl_${SB_MODEL_NAME}_bsz128}"
SB_SOLVER_PATH_DIR="${SB_SAVED_RESULTS_DIR}/Solver_ttrl_Base"
SB_STORAGE_PATH="${SB_SOLVER_PATH_DIR}/${SB_EXP_NAME}"
SB_CKPTS_DIR="${SB_STORAGE_PATH}/ckpts"
SB_TENSORBOARD_PATH="${SB_TENSORBOARD_DIR}/Solver_ttrl_Base-${SB_EXP_NAME}"

# 数据文件
SB_TRAIN_FILE="${SB_TRAIN_FILE:-${SB_DATA_DIR}/ttrl/ttrl_math_mix.parquet}"
SB_TEST_FILE="${SB_TEST_FILE:-${SB_DATA_DIR}/ttrl/test_set.parquet}"

# 训练超参数
SB_TRAIN_BATCH_SIZE="${SB_TRAIN_BATCH_SIZE:-128}"
SB_N_RESP_PER_PROMPT="${SB_N_RESP_PER_PROMPT:-8}"
SB_TOTAL_EPOCHS="${SB_TOTAL_EPOCHS:-4}"
SB_SAVE_FREQ="${SB_SAVE_FREQ:-20}"
SB_TEST_FREQ="${SB_TEST_FREQ:-10}"
SB_LR="${SB_LR:-1e-6}"
SB_KL_LOSS_COEF="${SB_KL_LOSS_COEF:-0.001}"

# 创建必要的目录
mkdir -p "${SB_CKPTS_DIR}" "${SB_TENSORBOARD_PATH}" "${SB_WORKING_DIR}/logs"

# =============================================================================
# 辅助函数
# =============================================================================

function now() {
    date '+%Y-%m-%d-%H-%M'
}

print_banner() {
    echo "=============================================="
    echo "  Solver Base TTRL Training Launcher"
    echo "=============================================="
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

print_usage() {
    echo "用法: $0 <n_gpus> [--eval] [--eval_steps \"step1,step2,...\"]"
    echo ""
    echo "参数:"
    echo "  n_gpus        需要的 GPU 数量"
    echo "  --eval        训练后自动评测"
    echo "  --eval_steps  指定评测的 checkpoint steps (逗号分隔)"
    echo ""
    echo "示例:"
    echo "  $0 4                                    # 使用 4 张 GPU 训练"
    echo "  $0 4 --eval                             # 训练后评测所有 checkpoint"
    echo "  $0 4 --eval --eval_steps \"20,40,60\"    # 训练后评测指定 steps"
    echo ""
    echo "环境变量:"
    echo "  SB_MODEL_NAME              基础模型名称 (默认: Qwen3-4B-Base)"
    echo "  SB_EXP_NAME                实验名称"
    echo "  SB_TRAIN_BATCH_SIZE        训练 batch size (默认: 128)"
    echo "  SB_TOTAL_EPOCHS            训练轮数 (默认: 4)"
    echo "  SB_SAVE_FREQ               保存频率 (默认: 20)"
    echo "  GPU_MEMORY_THRESHOLD_MB    显存空闲阈值 (默认: 4000 MB)"
}

log_info() {
    echo "[INFO] $(date '+%H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date '+%H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%H:%M:%S') - $1" >&2
}

# =============================================================================
# GPU 检测函数
# =============================================================================

get_gpu_info() {
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

get_idle_gpus() {
    local mem_threshold=$1
    local util_threshold=$2
    local idle_gpus=""
    
    while IFS=',' read -r gpu_id mem_used mem_total gpu_util; do
        if [ -z "$gpu_id" ]; then
            continue
        fi
        
        if [ "$mem_used" -lt "$mem_threshold" ] && [ "$gpu_util" -lt "$util_threshold" ]; then
            if [ -z "$idle_gpus" ]; then
                idle_gpus="$gpu_id"
            else
                idle_gpus="$idle_gpus $gpu_id"
            fi
        fi
    done < <(get_gpu_info)
    
    echo "$idle_gpus"
}

count_idle_gpus() {
    local idle_gpus
    idle_gpus=$(get_idle_gpus "$GPU_MEMORY_THRESHOLD_MB" "$GPU_UTIL_THRESHOLD")
    if [ -z "$idle_gpus" ]; then
        echo "0"
    else
        echo "$idle_gpus" | wc -w | tr -d ' '
    fi
}

select_gpus() {
    local n_needed=$1
    local idle_gpus
    idle_gpus=$(get_idle_gpus "$GPU_MEMORY_THRESHOLD_MB" "$GPU_UTIL_THRESHOLD")
    
    echo "$idle_gpus" | tr ' ' '\n' | head -n "$n_needed" | tr '\n' ' ' | sed 's/ $//'
}

print_gpu_status() {
    echo "当前 GPU 状态:"
    echo "----------------------------------------"
    printf "%-5s %-12s %-12s %-10s %-8s\n" "GPU" "显存使用" "显存总量" "利用率" "状态"
    echo "----------------------------------------"
    
    while IFS=',' read -r gpu_id mem_used mem_total gpu_util; do
        if [ -z "$gpu_id" ]; then
            continue
        fi
        
        local status="占用"
        if [ "$mem_used" -lt "$GPU_MEMORY_THRESHOLD_MB" ] && [ "$gpu_util" -lt "$GPU_UTIL_THRESHOLD" ]; then
            status="空闲"
        fi
        
        printf "%-5s %-12s %-12s %-10s %-8s\n" \
            "$gpu_id" "${mem_used}MB" "${mem_total}MB" "${gpu_util}%" "$status"
    done < <(get_gpu_info)
    
    echo "----------------------------------------"
    echo "空闲判定阈值: 显存 < ${GPU_MEMORY_THRESHOLD_MB}MB, 利用率 < ${GPU_UTIL_THRESHOLD}%"
}

# =============================================================================
# 等待 GPU 函数
# =============================================================================

wait_for_gpus() {
    local n_needed=$1
    local max_iterations=$((MAX_WAIT_HOURS * 3600 / POLL_INTERVAL))
    local iteration=0
    
    log_info "等待 $n_needed 张空闲 GPU..."
    log_info "检查间隔: ${POLL_INTERVAL}秒, 最大等待: ${MAX_WAIT_HOURS}小时"
    echo ""
    
    while true; do
        local n_idle
        n_idle=$(count_idle_gpus)
        
        log_info "当前空闲 GPU: $n_idle / 需要: $n_needed"
        
        if [ "$n_idle" -ge "$n_needed" ]; then
            log_info "GPU 资源满足要求!"
            return 0
        fi
        
        iteration=$((iteration + 1))
        if [ "$iteration" -ge "$max_iterations" ]; then
            log_error "等待超时 (${MAX_WAIT_HOURS}小时)，无法获取足够的 GPU"
            return 1
        fi
        
        if [ $((iteration % 10)) -eq 1 ]; then
            echo ""
            print_gpu_status
            echo ""
        fi
        
        log_info "等待 ${POLL_INTERVAL} 秒后重试..."
        sleep "$POLL_INTERVAL"
    done
}

# =============================================================================
# 环境变量设置函数
# =============================================================================

setup_gpu_env() {
    local n_gpus=$1
    local selected_gpus=$2
    
    local gpu_array=($selected_gpus)
    local n_selected=${#gpu_array[@]}
    
    if [ "$n_selected" -lt "$n_gpus" ]; then
        log_error "选中的 GPU 数量不足: $n_selected < $n_gpus"
        return 1
    fi
    
    # 构建 GPU ID 列表
    local gpu_ids=""
    for ((i=0; i<n_gpus; i++)); do
        if [ -z "$gpu_ids" ]; then
            gpu_ids="${gpu_array[$i]}"
        else
            gpu_ids="${gpu_ids},${gpu_array[$i]}"
        fi
    done
    
    # 设置环境变量
    export SB_N_GPUS="$n_gpus"
    export SB_GPU_IDS="$gpu_ids"
    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    export TENSORBOARD_DIR="${SB_TENSORBOARD_PATH}"
    
    # 导出路径环境变量
    export SB_BASB_DIR
    export SB_PROJECT_NAME
    export SB_CODE_MODULE
    export SB_WORKING_DIR
    export SB_MODEL_DIR
    export SB_MODEL_NAME
    export SB_DATA_DIR
    export SB_SAVED_RESULTS_DIR
    export SB_TENSORBOARD_DIR
    export SB_EXP_NAME
    export SB_SOLVER_PATH_DIR
    export SB_STORAGE_PATH
    export SB_CKPTS_DIR
    export SB_TENSORBOARD_PATH
    export SB_TRAIN_FILE
    export SB_TEST_FILE
    export SB_TRAIN_BATCH_SIZE
    export SB_N_RESP_PER_PROMPT
    export SB_TOTAL_EPOCHS
    export SB_SAVE_FREQ
    export SB_TEST_FREQ
    export SB_LR
    export SB_KL_LOSS_COEF
    
    # 打印配置
    echo ""
    echo "GPU 分配配置:"
    echo "=============================================="
    echo "总 GPU 数量:       $SB_N_GPUS"
    echo "选中 GPU IDs:      $SB_GPU_IDS"
    echo "=============================================="
    echo ""
    echo "训练配置:"
    echo "=============================================="
    echo "实验名称:          $SB_EXP_NAME"
    echo "基础模型:          $SB_MODEL_NAME"
    echo "模型路径:          ${SB_MODEL_DIR}/${SB_MODEL_NAME}"
    echo "训练数据:          $SB_TRAIN_FILE"
    echo "测试数据:          $SB_TEST_FILE"
    echo "Batch Size:        $SB_TRAIN_BATCH_SIZE"
    echo "Resp per Prompt:   $SB_N_RESP_PER_PROMPT"
    echo "训练轮数:          $SB_TOTAL_EPOCHS"
    echo "保存频率:          $SB_SAVE_FREQ"
    echo "测试频率:          $SB_TEST_FREQ"
    echo "学习率:            $SB_LR"
    echo "KL Loss 系数:      $SB_KL_LOSS_COEF"
    echo "=============================================="
    echo ""
    echo "路径配置:"
    echo "=============================================="
    echo "工作目录:          $SB_WORKING_DIR"
    echo "Checkpoint 目录:   $SB_CKPTS_DIR"
    echo "TensorBoard 目录:  $SB_TENSORBOARD_PATH"
    echo "=============================================="
    echo ""
}

# =============================================================================
# 训练函数
# =============================================================================

run_training() {
    log_info "启动 Solver Base 训练..."
    
    local solver_model_path="${SB_MODEL_DIR}/${SB_MODEL_NAME}"
    local train_prompt_mini_bsz=$((SB_TRAIN_BATCH_SIZE / 2))
    local max_prompt_length=$((1024 * 1))
    local max_response_length=$((1024 * 3))
    local actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
    
    cd "${SB_WORKING_DIR}"
    
    python3 -m se_code_ttrl.main_solver_dapo \
        data.train_files="${SB_TRAIN_FILE}" \
        data.val_files="${SB_TEST_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.val_batch_size=512 \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.return_raw_chat=True \
        data.train_batch_size=${SB_TRAIN_BATCH_SIZE} \
        actor_rollout_ref.rollout.n=${SB_N_RESP_PER_PROMPT} \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=${SB_KL_LOSS_COEF} \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.2 \
        algorithm.filter_groups.enable=False \
        algorithm.filter_groups.max_num_gen_batches=5 \
        algorithm.filter_groups.metric=acc \
        algorithm.filter_groups.filter_lower=0.125 \
        algorithm.filter_groups.filter_high=0.875 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.model.path="${solver_model_path}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=${SB_LR} \
        actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=${actor_ppo_max_token_len} \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        reward_model.reward_manager=solver \
        reward_model.reward_kwargs.storage_path=${SB_STORAGE_PATH} \
        reward_model.reward_kwargs.filter_lower=0.125 \
        reward_model.reward_kwargs.filter_high=0.875 \
        trainer.logger='["console","tensorboard"]' \
        trainer.project_name="Self-evolving-Agent-ttrl" \
        trainer.experiment_name="ttrl_Base-${SB_EXP_NAME}" \
        trainer.n_gpus_per_node=${SB_N_GPUS} \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        trainer.test_freq=${SB_TEST_FREQ} \
        trainer.save_freq=${SB_SAVE_FREQ} \
        trainer.total_epochs=${SB_TOTAL_EPOCHS} \
        trainer.default_local_dir="${SB_CKPTS_DIR}" \
        trainer.resume_mode=auto
    
    local exit_code=$?
    return $exit_code
}

# =============================================================================
# 评测函数
# =============================================================================

get_checkpoint_steps() {
    # 获取所有可用的 checkpoint steps
    local steps=()
    if [ -d "${SB_CKPTS_DIR}" ]; then
        for dir in "${SB_CKPTS_DIR}"/global_step_*; do
            if [ -d "$dir" ]; then
                local step=$(basename "$dir" | sed 's/global_step_//')
                steps+=("$step")
            fi
        done
    fi
    # 排序
    echo "${steps[@]}" | tr ' ' '\n' | sort -n | tr '\n' ' '
}

run_evaluation() {
    log_info "开始评测..."
    
    local steps_to_eval=""
    
    if [ -n "$EVAL_STEPS" ]; then
        # 使用用户指定的 steps
        steps_to_eval=$(echo "$EVAL_STEPS" | tr ',' ' ')
    else
        # 获取所有 checkpoint steps
        steps_to_eval=$(get_checkpoint_steps)
    fi
    
    if [ -z "$steps_to_eval" ]; then
        log_warn "没有找到可评测的 checkpoint"
        return 0
    fi
    
    log_info "将评测以下 steps: $steps_to_eval"
    
    # 调用评测脚本
    local eval_script="${SB_WORKING_DIR}/evaluation/eval_single_model_steps.sh"
    
    if [ -f "$eval_script" ]; then
        bash "$eval_script" \
            --exp_name "${SB_EXP_NAME}" \
            --ckpts_dir "${SB_CKPTS_DIR}" \
            --steps "$steps_to_eval" \
            --base_model_name "${SB_MODEL_NAME}" \
            --temperature 0.6
    else
        log_error "评测脚本不存在: $eval_script"
        return 1
    fi
}

# =============================================================================
# 主函数
# =============================================================================

main() {
    print_banner
    
    # 参数检查
    if [ -z "$N_GPUS" ]; then
        print_usage
        exit 1
    fi
    
    # 验证参数
    if ! [[ "$N_GPUS" =~ ^[0-9]+$ ]]; then
        log_error "n_gpus 必须是正整数: $N_GPUS"
        print_usage
        exit 1
    fi
    
    if [ "$N_GPUS" -lt 1 ]; then
        log_error "n_gpus 必须至少为 1"
        exit 1
    fi
    
    log_info "请求 GPU 数量: $N_GPUS"
    log_info "训练后评测: $DO_EVAL"
    if [ -n "$EVAL_STEPS" ]; then
        log_info "指定评测 steps: $EVAL_STEPS"
    fi
    echo ""
    
    # 设置日志
    local log_file="${SB_WORKING_DIR}/logs/train_solver_base_${SB_MODEL_NAME}-$(now).log"
    exec > >(tee -a "$log_file") 2>&1
    log_info "日志文件: $log_file"
    
    # 显示当前状态
    print_gpu_status
    echo ""
    
    # 等待足够的 GPU
    if ! wait_for_gpus "$N_GPUS"; then
        log_error "无法获取足够的 GPU，退出"
        exit 1
    fi
    
    # 选择 GPU
    local selected_gpus
    selected_gpus=$(select_gpus "$N_GPUS")
    log_info "选中 GPU: $selected_gpus"
    
    # 设置环境变量
    setup_gpu_env "$N_GPUS" "$selected_gpus"
    
    # 执行训练
    echo ""
    echo "=============================================="
    echo "  开始训练"
    echo "=============================================="
    echo ""
    
    run_training
    local train_exit_code=$?
    
    if [ $train_exit_code -ne 0 ]; then
        log_error "训练失败，退出码: $train_exit_code"
        exit $train_exit_code
    fi
    
    log_info "训练成功完成!"
    
    # 执行评测（如果启用）
    if [ "$DO_EVAL" = true ]; then
        echo ""
        echo "=============================================="
        echo "  开始评测"
        echo "=============================================="
        echo ""
        
        sleep 10  # 等待资源释放
        
        run_evaluation
        local eval_exit_code=$?
        
        if [ $eval_exit_code -ne 0 ]; then
            log_error "评测失败，退出码: $eval_exit_code"
            exit $eval_exit_code
        fi
        
        log_info "评测成功完成!"
    fi
    
    log_info "全部任务完成!"
    exit 0
}

# =============================================================================
# 入口
# =============================================================================

main "$@"
