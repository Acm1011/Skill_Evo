set -euo pipefail
function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash solver_base.sh > logs/RR-Zero-V1-Step15-DeepScaleR-40K-$(now).log 2>&1 &
nohup bash main.sh > ../logs/qr_Rule_gq_weakness_q4b-$(now).log 2>&1 &
#nohup bash main_gan.sh > ../logs/prompt2_gan_se-Zero-$(now).log 2>&1 &
#nohup bash main_rule.sh > ../logs/prompt2_rule_C20_se-Zero-$(now).log 2>&1 &
#nohup bash main_entropy.sh > ../logs/se-Zero-Entropy-$(now).log 2>&1 &
#nohup bash challenger_base.sh > ../logs/challenger_test-$(now).log 2>&1 &
#nohup bash challenger_entropy_base.sh > ../logs/challenger_entropy_base_test-$(now).log 2>&1 &