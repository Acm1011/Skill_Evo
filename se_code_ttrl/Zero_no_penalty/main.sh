#!/usr/bin/env bash
set -xeuo pipefail
# 获取脚本所在目录
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 从环境变量获取路径配置，如果未设置则使用默认值
project_name="${SE_PROJECT_NAME:-Self-evolving-Agent}"
dir="${SE_BASE_DIR:-/home/ycy/data1}"
model_dir="${SE_MODEL_DIR:-${dir}/models}"
data_dir="${SE_DATA_DIR:-${dir}/data}"
saved_results_dir="${SE_SAVED_RESULTS_DIR:-${dir}/saved_results}"
WORKING_DIR="${SE_WORKING_DIR:-${dir}/${project_name}}"
challenger_path_dir="${SE_CHALLENGER_DIR:-${saved_results_dir}/Challenger}"
solver_path_dir="${SE_SOLVER_DIR:-${saved_results_dir}/Solver}"
tensorboard_dir="${SE_TENSORBOARD_DIR:-${saved_results_dir}/tensorboard_log}"
base_model_name="${SE_BASE_MODEL_NAME:-Qwen2.5-Math-1.5B}"
base_model_path="${SE_BASE_MODEL_PATH:-${model_dir}/${base_model_name}}"
# test_file="${TTRL_TEST_FILE:-${data_dir}/ttrl/test_set.parquet}"
# export TTRL_TRAIN_FILE=${train_file}
# export TTRL_TEST_FILE=${test_file}
echo "[路径配置] 工作目录: ${WORKING_DIR}"
echo "[路径配置] 模型目录: ${model_dir}"
echo "[路径配置] 数据目录: ${data_dir}"
echo "[路径配置] 结果目录: ${saved_results_dir}"
echo "[路径配置] 基础模型: ${base_model_path}"
# R_Zero_ref_q
question_reward=R_Zero

dataset=Minerva
real_data_ratio=5.0
# dataset -> batch_size 映射表
declare -A dataset_batch_size_map=(
    ["AIME24"]=16
    ["AIME25"]=16
    ["AMC23"]=16
    ["MATH500"]=64
    ["Minerva"]=64
    ["OlympiadBench"]=64
)
declare -A dataset_rollout_n_map=(
    ["AIME24"]=16
    ["AIME25"]=16
    ["AMC23"]=16
    ["MATH500"]=8
    ["Minerva"]=8
    ["OlympiadBench"]=8
)
train_file=${data_dir}/ttrl/${dataset}.parquet
# 根据 dataset 获取对应的 batch_size，默认值为 8
solver_batch_size=${dataset_batch_size_map[$dataset]:-8}
rollout_n=${dataset_rollout_n_map[$dataset]:-8}
echo "[配置] dataset: ${dataset}, solver_batch_size: ${solver_batch_size}, train_file: ${train_file}"

group_question_repetion_penalty=False
gen_question_func=ttrl_icl
hybrid_data=True
if [ "$SE_N_GPUS" -eq 4 ]; then
    challenger_training_steps=5
else
    challenger_training_steps=5
fi
solver_training_steps=15

variant=no_d_data_${dataset}_${question_reward}_gq${gen_question_func}_${base_model_name}

#variant=ttrl_qr_${question_reward}_gq${gen_question_func}_${base_model_name}
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
bash ${SCRIPT_DIR}/solver.sh $exp_name ${challenger_path_dir}/${exp_name}/ckpts/global_step_${challenger_training_steps}/actor/huggingface $base_model_path $solver_training_steps $gen_question_func $hybrid_data  $train_file $solver_batch_size $real_data_ratio $rollout_n|| {
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