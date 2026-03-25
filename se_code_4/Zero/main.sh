#!/usr/bin/env bash
set -xeuo pipefail
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

project_name=Self-evolving-Agent
dir=/home/ycy/data1
model_dir=${dir}/models
data_dir=${dir}/data
saved_results_dir=${dir}/saved_results
WORKING_DIR=${dir}/${project_name}
challenger_path_dir=${saved_results_dir}/Challenger
solver_path_dir=${saved_results_dir}/Solver
tensorboard_dir=${saved_results_dir}/tensorboard_log
base_model_name=Qwen3-4B-Base
base_model_path=${model_dir}/${base_model_name}
question_reward=R_Zero
group_question_repetion_penalty=True
gen_question_func=weakness
challenger_training_steps=20
solver_training_steps=20
variant=qr_${question_reward}_gq${gen_question_func}_${base_model_name}
exp_name=${variant}-V1

solver_retrain_steps=15
solver_eval_step=15
solver_eval_temperature=0.6
solver_eval_num_iter=5
mkdir -p ${saved_results_dir} ${challenger_path_dir} ${solver_path_dir} ${tensorboard_dir} 


cd ${WORKING_DIR}
echo "开始第一轮训练..."
echo "训练 Challenger..."
bash ${SCRIPT_DIR}/challenger.sh $exp_name $base_model_path $base_model_path $challenger_training_steps $question_reward $group_question_repetion_penalty $gen_question_func || {
    echo "Error: 第一轮 Challenger 训练失败"
    exit 1
}

echo "训练 Solver..."
bash ${SCRIPT_DIR}/solver.sh $exp_name ${challenger_path_dir}/${exp_name}/ckpts/global_step_${challenger_training_steps}/actor/huggingface $base_model_path $solver_training_steps $gen_question_func || {
    echo "Error: 第一轮 Solver 训练失败"
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
        ${prev_solver_model_path} $challenger_training_steps $question_reward $group_question_repetion_penalty $gen_question_func || {
        echo "Error: 第 ${iter} 轮 Challenger 训练失败"
        exit 1
    }

    echo "训练 Solver (${exp_name})..."
    bash ${SCRIPT_DIR}/solver.sh \
        ${exp_name} ${cur_challenger_model_path} \
        ${prev_solver_model_path} $solver_training_steps $gen_question_func || {
        echo "Error: 第 ${iter} 轮 Solver 训练失败"
        exit 1
    }
    
    echo "第 ${iter} 轮训练完成"
done

echo "所有训练完成！"
echo "开始评估..."
bash "${WORKING_DIR}/se_code/evaluation/eval_all_run.sh" "${variant}" "${solver_eval_temperature}" "${solver_eval_step}" "${solver_eval_num_iter}" || {
    echo "Error: 评估失败"
    exit 1
}
echo "评估完成！"