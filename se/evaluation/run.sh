set -euo pipefail
function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
nohup bash run_sigle_model.sh  > ../eval_logs/eval-RR-Zero-V1-Step15-DeepScaleR-$(now).log 2>&1 &
