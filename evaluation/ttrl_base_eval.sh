#!/bin/bash
set -euo pipefail

exp_list=(
    # ttrl_Qwen2.5-Math-1.5B_AIME24_bsz8_epoch80
    # ttrl_Qwen2.5-Math-1.5B_AIME25_bsz8_epoch80
    # ttrl_Qwen2.5-Math-1.5B_AMC23_bsz8_epoch30
    # ttrl_Qwen2.5-Math-1.5B_MATH500_bsz32_epoch10
    # ttrl_Qwen2.5-Math-1.5B_Minerva_bsz32_epoch10
    # ttrl_Qwen2.5-Math-1.5B_OlympiadBench_bsz32_epoch10
    ttrl_Qwen3-4B_AIME24_bsz8_epoch80
    ttrl_Qwen3-4B_AIME25_bsz8_epoch80
    ttrl_Qwen3-4B_AMC23_bsz8_epoch30
    ttrl_Qwen3-4B_MATH500_bsz32_epoch10
    ttrl_Qwen3-4B_Minerva_bsz32_epoch10
    ttrl_Qwen3-4B_OlympiadBench_bsz32_epoch10
)

cd "$(dirname "$0")"

for exp_name in "${exp_list[@]}"; do
    echo "=============================================="
    echo "开始评测: ${exp_name}"
    echo "=============================================="
    
    bash eval_single_model_steps.sh --exp_name "${exp_name}" || {
        echo "Error: Failed to evaluate ${exp_name}"
        exit 1
    }
    bash eval_single_model_steps.sh --exp_name "${exp_name}" || {
        echo "Error: Failed to evaluate ${exp_name}"
        exit 1
    }
    
    # 清理评测相关的 Python 进程（更精确的匹配）
    #pkill -f "eval_.*_step.py" 2>/dev/null || true
    pkill -f python 2>/dev/null || true
    sleep 5
    
    echo "完成: ${exp_name}"
    echo ""
done

echo "=============================================="
echo "所有实验评测完成！"
echo "=============================================="