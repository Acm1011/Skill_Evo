set -euo pipefail
function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash solver_base.sh > logs/RR-Zero-V1-Step15-DeepScaleR-40K-$(now).log 2>&1 &
nohup bash solver_base.sh > logs/filter_data_Qwen3-4B-Base-$(now).log 2>&1 &