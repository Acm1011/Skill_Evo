#!/usr/bin/env bash
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

project_name='Self-evolving-Agent'
WORKING_DIR=/root/users/ycy/Self-evolving-Agent
saved_results_dir=/root/users/ycy/saved_results
challenger_path_dir=${saved_results_dir}/Challenger
solver_path_dir=${saved_results_dir}/Solver
tensorboard_dir=${saved_results_dir}/tensorboard_log
base_model_path_dir=/root/users/ycy/models/shares
base_model_name=Qwen3-4B-Base
base_model_path=${base_model_path_dir}/${base_model_name}
variant=RRR-Zero_${base_model_name}
exp_name=${variant}-V1
challenger_training_steps=5
solver_training_steps=20
solver_retrain_steps=15

mkdir -p ${saved_results_dir} ${challenger_path_dir} ${solver_path_dir} ${tensorboard_dir}
echo "开始第一轮训练..."
echo "训练 Challenger..."
bash ${SCRIPT_DIR}/challenger.sh $exp_name $base_model_path $base_model_path $challenger_training_steps || {
    echo "Error: 第一轮 Challenger 训练失败"
    exit 1
}

echo "训练 Solver..."
bash ${SCRIPT_DIR}/solver.sh $exp_name ${challenger_path_dir}/${exp_name}/ckpts//global_step_${challenger_training_steps}/actor/huggingface $base_model_path $solver_training_steps || {
    echo "Error: 第一轮 Solver 训练失败"
    exit 1
}


for iter in {2..6}; do
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
        ${prev_solver_model_path} $challenger_training_steps || {
        echo "Error: 第 ${iter} 轮 Challenger 训练失败"
        exit 1
    }

    echo "训练 Solver (${exp_name})..."
    bash ${SCRIPT_DIR}/solver.sh \
        ${exp_name} ${cur_challenger_model_path} \
        ${prev_solver_model_path} $solver_training_steps || {
        echo "Error: 第 ${iter} 轮 Solver 训练失败"
        exit 1
    }
    
    echo "第 ${iter} 轮训练完成"
done

echo "所有训练完成！"
