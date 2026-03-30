#!/usr/bin/env bash

set -xeuo pipefail
# 获取脚本所在目录
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 从环境变量获取路径配置，如果未设置则使用默认值


base_model_name=Qwen2.5-3B-Instruct
base_model_path=${SE_MODEL_DIR}/${base_model_name}
data_name=DeepMath-103K
data_file=${SE_DATA_DIR}/${data_name}.jsonl


variant=data_${data_name}_model_${base_model_name}

exp_name=${variant}-V1


solver_retrain_steps=15
solver_eval_step=15
solver_eval_temperature=0.6
solver_eval_num_iter=15
mkdir -p ${saved_results_dir} ${challenger_path_dir} ${solver_path_dir} ${tensorboard_dir}  ${WORKING_DIR}/logs
function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
exec > >(tee -a "${WORKING_DIR}/logs/train_${variant}-$(now).log") 2>&1

cd ${WORKING_DIR}
echo "开始第一轮训练..."
echo "训练 Challenger..."
bash ${SCRIPT_DIR}/challenger.sh $exp_name $base_model_path $base_model_path $challenger_training_steps $question_reward $group_question_repetion_penalty $gen_question_func $train_file|| {
    echo "Error: 第一轮 Challenger 训练失败"
    exit 1
}

echo "训练 Solver..."
bash ${SCRIPT_DIR}/solver.sh $exp_name ${challenger_path_dir}/${exp_name}/ckpts/global_step_${challenger_training_steps}/actor/huggingface $base_model_path $solver_training_steps $gen_question_func $hybrid_data  $train_file $solver_batch_size $real_data_ratio $rollout_n || {
    echo " Error: 第一轮 Solver 训练失败"
    exit 1
}


for iter in $(seq 2 ${solver_eval_num_iter}); do
    prev=$((iter-1))
    prev_exp_name=${variant}-V${prev}
    exp_name=${variant}-V${iter}
    prev_challenger_model_path=${challenger_path_dir}/${prev_exp_name}/ckpts/global_step_${challenger_training_steps}/actor/huggingface
    cur_challenger_model_path=${challenger_path_dir}/${exp_name}/ckpts/global_step_${challenger_training_steps}/actor/huggingface
    prev_solver_model_path=${solver_path_dir}/${prev_exp_name}/ckpts/global_step_${solver_retrain_steps}/actor/huggingface
    cur_solver_model_path=${solver_path_dir}/${exp_name}/ckpts/global_step_${solver_retrain_steps}/actor/huggingface

    echo "开始第 ${iter} 轮训练..."
    echo "训练 Challenger (${exp_name})..."
    bash ${SCRIPT_DIR}/challenger.sh \
        ${exp_name} ${prev_challenger_model_path} \
        ${prev_solver_model_path} $challenger_training_steps $question_reward $group_question_repetion_penalty $gen_question_func $train_file || {
        echo "Error: 第 ${iter} 轮 Challenger 训练失败"
        exit 1
    }

    echo "训练 Solver (${exp_name})..."
    bash ${SCRIPT_DIR}/solver.sh \
        ${exp_name} ${cur_challenger_model_path} \
        ${prev_solver_model_path} $solver_training_steps $gen_question_func $hybrid_data $train_file $solver_batch_size $real_data_ratio $rollout_n|| {
        echo "Error: 第 ${iter} 轮 Solver 训练失败"
        exit 1
    }
    
    echo "第 ${iter} 轮训练完成"
done

echo "所有训练完成！"
echo "开始评估..."
eval_script="${SE_EVAL_SCRIPT:-${WORKING_DIR}/evaluation/eval_single_math_data.sh}"
bash "$eval_script" "${variant}"  "${solver_eval_step}" "${solver_eval_num_iter}" "${base_model_name}" "${dataset}" || {
    echo "Error: 评估失败"
    exit 1
}
bash "$eval_script" "${variant}"  "${solver_eval_step}" "${solver_eval_num_iter}" "${base_model_name}" "${dataset}"|| {
    echo "Error: 评估失败"
    exit 1
}

# eval_script="${SE_EVAL_SCRIPT:-${WORKING_DIR}/evaluation/eval_all_run.sh}"
# bash "$eval_script" "${variant}"  "${temperature}" "${solver_eval_step}" "${solver_eval_num_iter}" "${base_model_name}" || {
#     echo "Error: 评估失败"
#     exit 1
# }
# bash "$eval_script" "${variant}" "${temperature}" "${solver_eval_step}" "${solver_eval_num_iter}" "${base_model_name}"|| {
#     echo "Error: 评估失败"
#     exit 1
# }
# echo "评估完成！"