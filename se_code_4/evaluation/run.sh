set -euo pipefail
variant=$1
temperature=$2
eval_step=$3
num_iter=$4

function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
#nohup bash run_sigle_model.sh  > ../eval_logs/filter_data_Qwen3-4B-Base-Step15-DeepScaleR-$(now).log 2>&1 &
nohup bash eval_all_run.sh $variant $temperature $eval_step $num_iter > ../eval_logs/eval_${variant}_step${eval_step}_iter${num_iter}_temperature${temperature}-$(now).log 2>&1 &
