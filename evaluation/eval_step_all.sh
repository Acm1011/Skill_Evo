#!/bin/bash
# 批量评测多个实验的多个 steps

cd "$(dirname "$0")"

exp_list=(
    ttrl_Qwen2.5-Math-7B_AIME25_bsz8_epoch80
    ttrl_Qwen2.5-Math-7B_AMC23_bsz8_epoch30
    ttrl_Qwen2.5-Math-7B_MATH500_bsz32_epoch10
    ttrl_Qwen2.5-Math-7B_Minerva_bsz32_epoch10
    ttrl_Qwen2.5-Math-7B_OlympiadBench_bsz32_epoch10
)
dataset_list=(
    AIME25
    AMC23
    MATH500
    Minerva
    OlympiadBench
)

for i in "${!exp_list[@]}"; do
    exp_name="${exp_list[$i]}"
    dataset="${dataset_list[$i]}"
    
    echo "==> [$((i+1))/${#exp_list[@]}] ${exp_name} on ${dataset}"
    
    ./eval_single_math_data_steps.sh \
        --exp_name "${exp_name}" \
        --dataset "${dataset}"
done

echo "==> All done!"
