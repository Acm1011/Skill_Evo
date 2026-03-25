set -euo pipefail
function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
nohup bash eval_run.sh Qwen3-4B-Base > eval_logs/eval_Qwen3-4B-Base-$(now).log 2>&1 &
