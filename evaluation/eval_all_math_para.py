import vllm
import argparse
from transformers import AutoTokenizer, AutoConfig
import json
from math_verify import parse, verify
import os
import pandas as pd
import numpy as np
import torch
import gc
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

#STORAGE_PATH = os.getenv("STORAGE_PATH")
#/users/ycy/saved_results'


def get_max_position_embeddings(model_path):
    """从模型配置中获取 max_position_embeddings"""
    try:
        config = AutoConfig.from_pretrained(model_path)
        return getattr(config, 'max_position_embeddings', None)
    except Exception as e:
        print(f"Warning: 无法读取模型配置: {e}")
        return None


def is_qwen3_post_trained(model_path):
    """判断是否是后训练的 Qwen3 模型（非 Base 模型）
    
    通过 model_type 判断是否是 Qwen3，通过 _name_or_path 判断是否是 Base 模型
    Qwen3 Base 模型的命名格式是 Qwen3-*B-Base，如 Qwen/Qwen3-4B-Base
    """
    try:
        config = AutoConfig.from_pretrained(model_path)
        model_type = getattr(config, 'model_type', '')
        name_or_path = getattr(config, '_name_or_path', model_path)
        
        # 检查是否是 Qwen3 模型
        is_qwen3 = 'qwen3' in model_type.lower()
        
        # 通过 _name_or_path 判断是否是 Base 模型
        # Qwen3 Base 模型的命名格式是 Qwen3-*B-Base，如 Qwen/Qwen3-4B-Base
        # 匹配模式：qwen3-xxxb-base 或 qwen3-xxx-base（忽略大小写）
        is_base = bool(re.search(r'qwen3-[^/-]+-base\b', name_or_path.lower()))
        
        return is_qwen3 and not is_base
    except Exception as e:
        print(f"Warning: 无法读取模型配置: {e}")
        return False


def clear_model_memory():
    """清理模型显存"""
    # 清空GPU缓存
    torch.cuda.empty_cache()
    
    # 强制垃圾回收
    gc.collect()
    
    print("模型显存已清理")



def main(args):
    clear_model_memory()
    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    save_path_dir = os.path.join(args.save_path_dir, args.model_name)
    os.makedirs(save_path_dir, exist_ok=True)
    print(f'eval model: {args.model_name} performance on the math dataset: {args.dataset}, eval results will be saved in {save_path_dir}, using GPUs: {gpu_ids}, temperature: {args.temperature}, n_samples: {args.n_samples}')
    
    
    dataset_path = os.path.join(args.data_path_dir, f'{args.dataset}.parquet')
    if not os.path.exists(dataset_path):
        raise ValueError(f"dataset:{args.dataset} not found")
    dataset = pd.read_parquet(dataset_path)
    print(f'load {len(dataset)} examples from {dataset_path} successfully')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 获取模型配置信息
    max_pos_emb = get_max_position_embeddings(args.model_path)
    
    # 检查是否需要禁用 thinking（后训练的 Qwen3 模型，Base 模型不需要）
    qwen3_post_trained = is_qwen3_post_trained(args.model_path)
    if qwen3_post_trained:
        print(f"检测到后训练的 Qwen3 模型，将使用 enable_thinking=False")
    
    # 根据 max_position_embeddings 动态调整 max_tokens
    if max_pos_emb is not None and max_pos_emb <= 4096:
        max_tokens = min(args.max_gen_len, 3072)
        print(f"模型 max_position_embeddings={max_pos_emb}，调整 max_tokens={max_tokens}")
    elif max_pos_emb is not None and max_pos_emb <= 8192:
        max_tokens = min(args.max_gen_len, 6144)
        print(f"模型 max_position_embeddings={max_pos_emb}，调整 max_tokens={max_tokens}")
    else:
        max_tokens = args.max_gen_len
        print(f"模型 max_position_embeddings={max_pos_emb}，使用默认 max_tokens={max_tokens}")
    
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=0.9
    )
    
    sample_params = vllm.SamplingParams(
        max_tokens=max_tokens,
        temperature=args.temperature,
        stop_token_ids=[tokenizer.eos_token_id])
    n_samples = args.n_samples

    model_name = args.model_name
    print(f'model[{model_name}] with dataset[{args.dataset}] set sampling parameters: max_tokens={max_tokens}, temperature={args.temperature}, n_samples={n_samples}')
    print(f'model[{model_name}] with dataset[{args.dataset}] start to generate responses for {len(dataset)} examples')
    chat_lst = dataset['prompt'].tolist()
    response_lst = [[] for _ in range(n_samples)]
    prompts = []
    batch_size=args.batch_size
    num_batch = -(- len(dataset) // batch_size)
    #chats=[[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": question}] for question in questions]
    #for batch_idx in tqdm(range(num_batch), desc=f'model[{model_name}] with dataset[{args.dataset}]', total=num_batch):
    for batch_idx in range(num_batch):
        batch_chats = chat_lst[batch_idx * batch_size: (batch_idx+1)*batch_size]
        if tokenizer.chat_template:
            inputs = []
            for chat in batch_chats:
                if qwen3_post_trained:
                    formatted_chat = tokenizer.apply_chat_template(
                        chat,
                        add_generation_prompt=True,
                        tokenize=False,
                        #add_special_tokens=True,
                        enable_thinking=False
                    )
                else:
                    formatted_chat = tokenizer.apply_chat_template(
                        chat,
                        add_generation_prompt=True,
                        tokenize=False,
                        #add_special_tokens=True
                    )
                inputs.append(formatted_chat)
            prompts.extend(inputs)
        else:
            raise NotImplementedError(f'not support ')
        
        for n_sample in range(n_samples):
            print(f'model[{model_name}] with dataset[{args.dataset}] generating response {n_sample+1} of {n_samples} for batch {batch_idx+1} of {num_batch}',flush=True)
            outputs = model.generate(inputs, sampling_params=sample_params,use_tqdm=False)
            outputs = [out.outputs[0].text for out in outputs]

            n_outputs = []
            for i in range(len(outputs)):
                response_item = outputs[i]
                n_outputs.append(response_item)

            response_lst[n_sample].extend(n_outputs)
    
    response_lst = np.array(response_lst, dtype=object)
    response_lst = np.transpose(response_lst, axes=(1, 0)).tolist()
    prompts = np.array(prompts, dtype=object)
    assert len(response_lst) == len(dataset) == len(prompts), f'length of response_lst and dataset and prompts are not equal, please check the dataset'
    dataset['responses'] = response_lst
    dataset['formatted_prompt'] = prompts
    print(f'model[{model_name}] with dataset[{args.dataset}] generated {len(response_lst[0])} responses for {len(dataset)} examples successfully')
    
    data_path=os.path.join(save_path_dir, f'{args.dataset}_responses.parquet')
    tmp_path = data_path + '.tmp'
    dataset.to_parquet(tmp_path)
    os.rename(tmp_path, data_path)
    print(f'responses for {args.dataset} have been saved in {save_path_dir}')
    
    # Clean up model memory
    del model
    clear_model_memory()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/ycy/data3/ttrl_saved/Solver_ttrl/data_MATH500_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-7B-V20/ckpts/global_step_10/actor/huggingface")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-Math-7B-math500-v1")
    parser.add_argument("--dataset", type=str, default='MATH500')
    parser.add_argument("--save_path_dir", type=str, default="/home/ycy/data1/Self-evolving-Agent/evaluation")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_gen_len", type=int, default=4096)   
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--data_path_dir", type=str, default='/home/ycy/data1/data/nop')

    args = parser.parse_args()
    main(args)