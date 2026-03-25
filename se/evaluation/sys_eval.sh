export VLLM_DISABLE_COMPILE_CACHE=1
export HF_ENDPOINT=https://hf-mirror.com
eval_path_dir=/root/users/ycy/saved_results/sys_eval_results
base_path_dir=/root/users/ycy/models/shares
solver_path_dir=/root/users/ycy/saved_results/Solver

model_name=Qwen3-4B-Base

if [ $model_name == "Qwen3-4B-Base" ]; then
  model_path=${base_path_dir}/Qwen3-4B-Base
elif [ $model_name == "Qwen3-8B-Base" ]; then
  model_path=${base_path_dir}/Qwen3-8B-Base
elif [ $model_name == "Qwen3-14B-Base" ]; then
  model_path=${base_path_dir}/Qwen3-14B-Base
elif [[ $model_name == *"V"[1-9] ]]; then
  model_path=${solver_path_dir}/${model_name}/ckpts/global_step_20/actor/huggingface
else
  echo "错误: 不支持的模型名称: $model_name"
  echo "支持的模型: Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-14B-Base, 或包含V1-V9的模型名"
  exit 1
fi

mkdir -p ${eval_path_dir}
CUDA_VISIBLE_DEVICES=0 python eval_all_math.py --model "${model_path}" --dataset "math500_sys" --save_path "${eval_path_dir}" 