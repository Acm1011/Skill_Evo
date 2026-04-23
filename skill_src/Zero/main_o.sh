#!/usr/bin/env bash

set -xeuo pipefail
# 获取脚本所在目录
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 从环境变量获取路径配置，如果未设置则使用默认值


base_model_name=Qwen3-4B-Instruct-2507
base_model_path=${SE_MODEL_DIR}/${base_model_name}
data_name=DeepMath-103K
data_file=${SE_DATA_DIR}/${data_name}.jsonl

project_name="${SE_PROJECT_NAME:-Skill_Evo}"
exp_name=data_${data_name}_model_${base_model_name}
initial_version=V1

dir="${SE_BASE_DIR:-/home/ycy/sdi/}"
model_dir="${SE_MODEL_DIR:-${dir}/models}"
data_dir="${SE_DATA_DIR:-${dir}/data}"
WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
saved_results_dir="${SE_SAVED_RESULTS_DIR:-${SE_SKILL_SAVED_ROOT}/${exp_name}}"
synthesizer_path_dir="${SE_SYNTHESIZER_DIR:-${saved_results_dir}/Synthesizer}"
solver_path_dir="${SE_SOLVER_DIR:-${saved_results_dir}/Solver}"
memory_path_dir="${SE_MEMORY_DIR:-${saved_results_dir}/Memory}"
log_path_dir="${SE_LOG_DIR:-${saved_results_dir}/logs}"
mkdir -p ${saved_results_dir} ${synthesizer_path_dir} ${solver_path_dir} ${memory_path_dir} ${log_path_dir}
export SYNTHESIZER_PATH_DIR=${synthesizer_path_dir}
export SOLVER_PATH_DIR=${solver_path_dir}
export MEMORY_PATH_DIR=${memory_path_dir}
export LOG_PATH_DIR=${log_path_dir}
export EXP_NAME=${exp_name}
export WORKING_DIR=${WORKING_DIR}

solver_retrain_steps=40
synthesizer_training_steps=20
solver_eval_step=15
solver_eval_temperature=0.6
solver_eval_num_iter=15

function now() {
	    date '+%Y-%m-%d-%H-%M'
    }
exec > >(tee -a "${log_path_dir}/main_${variant}-$(now).log") 2>&1

cd ${WORKING_DIR}
echo "开始第一轮训练..."
echo "训练 Challenger..."
bash ${SCRIPT_DIR}/Synthesizer.sh ${initial_version} ${base_model_path} ${base_model_path} ${synthesizer_training_steps} ${data_file} || {
    echo "Error: 第一轮 Challenger 训练失败"
    exit 1
}

echo "训练 Solver..."
bash ${SCRIPT_DIR}/solver.sh ${initial_version} ${base_model_path} ${solver_training_steps} || {
    echo " Error: 第一轮 Solver 训练失败"
    exit 1
}


for iter in $(seq 2 ${solver_eval_num_iter}); do
    prev=$((iter-1))
    prev_exp_version=V${prev}
    exp_version=V${iter}
    
    prev_synthesizer_model_path=${synthesizer_path_dir}/${prev_exp_version}/ckpts/global_step_${synthesizer_training_steps}/actor/huggingface
    cur_synthesizer_model_path=${challenger_path_dir}/${exp_version}/ckpts/global_step_${synthesizer_training_steps}/actor/huggingface
    prev_solver_model_path=${solver_path_dir}/${prev_exp_name}/ckpts/global_step_${solver_retrain_steps}/actor/huggingface
    cur_solver_model_path=${solver_path_dir}/${exp_version}/ckpts/global_step_${solver_retrain_steps}/actor/huggingface

    echo "开始第 ${iter} 轮训练..."
    echo "训练 Challenger (${exp_name})..."
    bash ${SCRIPT_DIR}/Synthesizer.sh \
        ${exp_version} ${prev_synthesizer_model_path} \
        ${prev_solver_model_path} ${synthesizer_training_steps} ${data_file} || {
        echo "Error: 第 ${iter} 轮 Challenger 训练失败"
        exit 1
    }

    echo "训练 Solver (${exp_name})..."
    bash ${SCRIPT_DIR}/solver.sh \
        ${exp_version} ${prev_solver_model_path} ${solver_training_steps} || {
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