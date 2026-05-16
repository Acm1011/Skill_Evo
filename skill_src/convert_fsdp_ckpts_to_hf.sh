#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

usage() {
    cat <<'EOF'
批量将一个或多个目录中的 FSDP actor checkpoints 转换为 HuggingFace 权重。
默认只转换，不删除原始 shard；只有显式传 --delete-shards 才会在确认转换成功后清理。

用法:
  bash skill_src/convert_fsdp_ckpts_to_hf.sh [--delete-shards] <path> [<path> ...]

示例:
  bash skill_src/convert_fsdp_ckpts_to_hf.sh \
    /home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b/skillrl_grpo_qwen3_4b
  bash skill_src/convert_fsdp_ckpts_to_hf.sh \
    /home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/verl_grpo_qwen3_4b
  bash skill_src/convert_fsdp_ckpts_to_hf.sh \
    /home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b \
    /home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/verl_gspo_qwen3_4b

目录结构预期:
  <ckpts_root>/global_step_100/actor/
    - model_world_size_*_rank_*.pt
    - optim_world_size_*_rank_*.pt
    - extra_state_world_size_*_rank_*.pt
    - huggingface/

脚本行为:
  1. 若输入目录本身包含 global_step_*，直接按一个 ckpt root 处理
  2. 若输入目录不直接包含 global_step_*，自动向下发现真实 ckpt root
  3. 将每个 global_step_*/actor 下的 FSDP shard 合并到 actor/huggingface
  4. 仅当传入 --delete-shards 且发现有效 HF 权重文件时，才删除 actor 下原始 shard ckpt
EOF
}

DELETE_SHARDS="false"
INPUT_PATHS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delete-shards)
            DELETE_SHARDS="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            INPUT_PATHS+=( "$1" )
            shift
            ;;
    esac
done

if [[ ${#INPUT_PATHS[@]} -eq 0 ]]; then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MERGER_SCRIPT="${REPO_ROOT}/skill_src/model_merger.py"

if [[ ! -f "${MERGER_SCRIPT}" ]]; then
    echo "Error: merger 脚本不存在: ${MERGER_SCRIPT}"
    exit 1
fi

is_hf_export_complete() {
    local hf_dir="$1"
    [[ -d "${hf_dir}" ]] || return 1
    [[ -f "${hf_dir}/config.json" ]] || return 1
    has_hf_weight_files "${hf_dir}" || return 1
    has_hf_tokenizer_assets "${hf_dir}" || return 1
    return 0
}

has_hf_weight_files() {
    local hf_dir="$1"
    [[ -d "${hf_dir}" ]] || return 1
    [[ -f "${hf_dir}/model.safetensors" ]] && return 0
    [[ -f "${hf_dir}/pytorch_model.bin" ]] && return 0

    local safetensor_shards=( "${hf_dir}"/model-*.safetensors )
    [[ ${#safetensor_shards[@]} -gt 0 ]] && [[ -f "${hf_dir}/model.safetensors.index.json" ]] && return 0

    local bin_shards=( "${hf_dir}"/pytorch_model-*.bin )
    [[ ${#bin_shards[@]} -gt 0 ]] && [[ -f "${hf_dir}/pytorch_model.bin.index.json" ]] && return 0

    return 1
}

has_hf_tokenizer_assets() {
    local hf_dir="$1"
    [[ -d "${hf_dir}" ]] || return 1

    local tokenizer_files=(
        "${hf_dir}/tokenizer.json"
        "${hf_dir}/tokenizer_config.json"
        "${hf_dir}/vocab.json"
        "${hf_dir}/merges.txt"
        "${hf_dir}/spiece.model"
        "${hf_dir}/sentencepiece.bpe.model"
    )

    local asset
    for asset in "${tokenizer_files[@]}"; do
        [[ -f "${asset}" ]] && return 0
    done

    return 1
}

describe_hf_export_state() {
    local hf_dir="$1"
    local parts=()

    if [[ ! -d "${hf_dir}" ]]; then
        echo "目录不存在"
        return 0
    fi

    if [[ -f "${hf_dir}/config.json" ]]; then
        parts+=( "config=ok" )
    else
        parts+=( "config=missing" )
    fi

    if has_hf_weight_files "${hf_dir}"; then
        parts+=( "weights=ok" )
    else
        parts+=( "weights=missing" )
    fi

    if has_hf_tokenizer_assets "${hf_dir}"; then
        parts+=( "tokenizer=ok" )
    else
        parts+=( "tokenizer=missing" )
    fi

    printf '%s' "${parts[*]}"
}

has_fsdp_shards() {
    local actor_dir="$1"
    local shard_files=(
        "${actor_dir}"/model_world_size_*_rank_*.pt
        "${actor_dir}"/optim_world_size_*_rank_*.pt
        "${actor_dir}"/extra_state_world_size_*_rank_*.pt
    )
    [[ ${#shard_files[@]} -gt 0 ]]
}

delete_fsdp_shards() {
    local actor_dir="$1"
    local shard_files=(
        "${actor_dir}"/model_world_size_*_rank_*.pt
        "${actor_dir}"/optim_world_size_*_rank_*.pt
        "${actor_dir}"/extra_state_world_size_*_rank_*.pt
    )

    if [[ ${#shard_files[@]} -eq 0 ]]; then
        echo "    没有发现可删除的 shard 文件"
        return 0
    fi

    rm -f "${shard_files[@]}"
    echo "    已删除 ${#shard_files[@]} 个原始 shard 文件"
}

is_ckpts_root() {
    local ckpts_root="$1"
    local step_dirs=( "${ckpts_root}"/global_step_* )
    [[ ${#step_dirs[@]} -gt 0 ]]
}

discover_ckpts_roots() {
    local input_path="$1"

    if [[ ! -d "${input_path}" ]]; then
        echo "Error: 输入目录不存在: ${input_path}" >&2
        return 1
    fi

    if is_ckpts_root "${input_path}"; then
        printf '%s\n' "${input_path}"
        return 0
    fi

    find "${input_path}" -mindepth 1 -maxdepth 4 -type d | while read -r candidate; do
        if is_ckpts_root "${candidate}"; then
            printf '%s\n' "${candidate}"
        fi
    done
}

declare -A CKPTS_ROOT_MAP=()
for input_path in "${INPUT_PATHS[@]}"; do
    while IFS= read -r discovered_root; do
        [[ -n "${discovered_root}" ]] || continue
        CKPTS_ROOT_MAP["${discovered_root}"]=1
    done < <(discover_ckpts_roots "${input_path}")
done

CKPTS_ROOTS=( "${!CKPTS_ROOT_MAP[@]}" )
IFS=$'\n' CKPTS_ROOTS=( $(printf '%s\n' "${CKPTS_ROOTS[@]}" | sort) )
unset IFS

if [[ ${#CKPTS_ROOTS[@]} -eq 0 ]]; then
    echo "Error: 未发现任何包含 global_step_* 的 checkpoint 根目录"
    exit 1
fi

echo "=============================================="
echo "Batch FSDP -> HuggingFace conversion"
echo "  input_paths: ${#INPUT_PATHS[@]}"
for input_path in "${INPUT_PATHS[@]}"; do
    echo "    - ${input_path}"
done
echo "  ckpts_roots: ${#CKPTS_ROOTS[@]}"
for ckpts_root in "${CKPTS_ROOTS[@]}"; do
    echo "    - ${ckpts_root}"
done
echo "  repo_root:  ${REPO_ROOT}"
echo "  delete:     ${DELETE_SHARDS}"
echo "=============================================="

converted_count=0
skipped_count=0
failed_count=0
deleted_count=0

for ckpts_root in "${CKPTS_ROOTS[@]}"; do
    step_dirs=( "${ckpts_root}"/global_step_* )
    if [[ ${#step_dirs[@]} -eq 0 ]]; then
        echo
        echo "==> 跳过 ckpt root: ${ckpts_root}"
        echo "    原因: 未找到任何 global_step_* 目录"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    echo
    echo "=============================================="
    echo "处理 ckpt root: ${ckpts_root}"
    echo "=============================================="

    for step_dir in "${step_dirs[@]}"; do
        step_name="$(basename "${step_dir}")"
        actor_dir="${step_dir}/actor"
        hf_dir="${actor_dir}/huggingface"

        echo
        echo "==> 处理 ${step_name}"

        if [[ ! -d "${actor_dir}" ]]; then
            echo "    跳过: actor 目录不存在: ${actor_dir}"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        mkdir -p "${hf_dir}"

        if ! has_fsdp_shards "${actor_dir}"; then
            if is_hf_export_complete "${hf_dir}"; then
                echo "    跳过: 已存在有效 HF 权重，且 shard 已清理"
                skipped_count=$((skipped_count + 1))
                continue
            fi

            if has_hf_weight_files "${hf_dir}" || [[ -f "${hf_dir}/config.json" ]]; then
                echo "    失败: 发现半成品 HF 导出，但 shard 已不存在，无法自动修复"
                echo "    当前状态: $(describe_hf_export_state "${hf_dir}")"
                failed_count=$((failed_count + 1))
                continue
            fi

            echo "    跳过: 未发现 FSDP shard 文件"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        if is_hf_export_complete "${hf_dir}"; then
            if [[ "${DELETE_SHARDS}" == "true" ]]; then
                echo "    已存在有效 HF 权重，开始删除原始 shard"
                delete_fsdp_shards "${actor_dir}"
                deleted_count=$((deleted_count + 1))
            else
                echo "    已存在有效 HF 权重，保留原始 shard"
            fi
            skipped_count=$((skipped_count + 1))
            continue
        fi

        if has_hf_weight_files "${hf_dir}" || [[ -f "${hf_dir}/config.json" ]] || has_hf_tokenizer_assets "${hf_dir}"; then
            echo "    检测到半成品 HF 导出，重新执行 merge 修复"
            echo "    当前状态: $(describe_hf_export_state "${hf_dir}")"
        fi

        echo "    开始合并到 ${hf_dir}"
        if python "${MERGER_SCRIPT}" merge \
            --backend fsdp \
            --local_dir "${actor_dir}" \
            --target_dir "${hf_dir}"; then
            if is_hf_export_complete "${hf_dir}"; then
                converted_count=$((converted_count + 1))
                if [[ "${DELETE_SHARDS}" == "true" ]]; then
                    echo "    合并成功，开始删除原始 shard"
                    delete_fsdp_shards "${actor_dir}"
                    deleted_count=$((deleted_count + 1))
                else
                    echo "    合并成功，保留原始 shard"
                fi
            else
                echo "    失败: merger 返回成功，但未检测到有效 HF 权重"
                failed_count=$((failed_count + 1))
            fi
        else
            echo "    失败: 合并命令执行出错"
            failed_count=$((failed_count + 1))
        fi
    done
done

echo
echo "=============================================="
echo "完成"
echo "  转换成功: ${converted_count}"
echo "  跳过:     ${skipped_count}"
echo "  失败:     ${failed_count}"
echo "  已清理:   ${deleted_count}"
echo "=============================================="

if [[ ${failed_count} -gt 0 ]]; then
    exit 1
fi
