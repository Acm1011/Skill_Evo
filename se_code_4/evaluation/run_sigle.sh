set -euo pipefail

function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
#nohup bash run_sigle_model.sh  > ../eval_logs/filter_data_Qwen3-4B-Base-Step15-DeepScaleR-$(now).log 2>&1 &
nohup bash run_all.sh > ../eval_logs/eval_qr_Rule_gqTopic_AoPS-$(now).log 2>&1 &
