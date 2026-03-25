set -euo pipefail
function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash solver_base.sh > /home/ycy/data1/Self-evolving-Agent/logs/ttrl_base_mix_data_Qwen3-4B-Base-bsz128-$(now).log 2>&1 &
#nohup bash solver_base.sh > logs/filter_data_Qwen3-4B-Base-$(now).log 2>&1 &