#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

usage() {
    cat <<'EOF'
批量将给定目录下的 FSDP actor checkpoints 转换为 HuggingFace 权重。
默认只转换，不删除原始 shard；只有显式传 --delete-shards 才会在确认转换成功后清理。

用法:
  bash skill_src/convert_fsdp_ckpts_to_hf.sh [--delete-shards] <ckpts_root>

示例:
  bash skill_src/convert_fsdp_ckpts_to_hf.sh \
    /home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b/skillrl_grpo_qwen3_4b

目录结构预期:
  <ckpts_root>/global_step_100/actor/
    - model_world_size_*_rank_*.pt
    - optim_world_size_*_rank_*.pt
    - extra_state_world_size_*_rank_*.pt
    - huggingface/

脚本行为:
  1. 遍历所有 global_step_* 目录
  2. 将 actor 下的 FSDP shard 合并到 actor/huggingface
  3. 仅当传入 --delete-shards 且发现有效 HF 权重文件时，才删除 actor 下原始 shard ckpt
EOF
}

DELETE_SHARDS="false"

if [[ $# -eq 2 && "$1" == "--delete-shards" ]]; then
    DELETE_SHARDS="true"
    shift
fi

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

CKPTS_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MERGER_SCRIPT="${REPO_ROOT}/SkillRL/scripts/model_merger.py"

if [[ ! -d "${CKPTS_ROOT}" ]]; then
    echo "Error: checkpoint 根目录不存在: ${CKPTS_ROOT}"
    exit 1
fi

if [[ ! -f "${MERGER_SCRIPT}" ]]; then
    echo "Error: merger 脚本不存在: ${MERGER_SCRIPT}"
    exit 1
fi

is_hf_export_complete() {
    local hf_dir="$1"
    [[ -d "${hf_dir}" ]] || return 1
    [[ -f "${hf_dir}/config.json" ]] || return 1
    [[ -f "${hf_dir}/model.safetensors" ]] && return 0
    [[ -f "${hf_dir}/pytorch_model.bin" ]] && return 0

    local safetensor_shards=( "${hf_dir}"/model-*.safetensors )
    [[ ${#safetensor_shards[@]} -gt 0 ]] && [[ -f "${hf_dir}/model.safetensors.index.json" ]] && return 0

    local bin_shards=( "${hf_dir}"/pytorch_model-*.bin )
    [[ ${#bin_shards[@]} -gt 0 ]] && [[ -f "${hf_dir}/pytorch_model.bin.index.json" ]] && return 0

    return 1
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

echo "=============================================="
echo "Batch FSDP -> HuggingFace conversion"
echo "  ckpts_root: ${CKPTS_ROOT}"
echo "  repo_root:  ${REPO_ROOT}"
echo "  delete:     ${DELETE_SHARDS}"
echo "=============================================="

step_dirs=( "${CKPTS_ROOT}"/global_step_* )
if [[ ${#step_dirs[@]} -eq 0 ]]; then
    echo "Error: 未找到任何 global_step_* 目录: ${CKPTS_ROOT}"
    exit 1
fi

converted_count=0
skipped_count=0
failed_count=0
deleted_count=0

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
