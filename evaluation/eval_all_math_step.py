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
import random
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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
    torch.cuda.empty_cache()
    gc.collect()
    print("模型显存已清理")


def extract_step_from_path(model_path):
    """从模型路径中提取 step 数字"""
    # 匹配 global_step_123 格式
    match = re.search(r'global_step_(\d+)', model_path)
    if match:
        return int(match.group(1))
    return None


def extract_step_from_name(model_name):
    """从模型名称中提取 step 数字"""
    # 匹配 xxx-step123 格式
    match = re.search(r'-step(\d+)$', model_name)
    if match:
        return int(match.group(1))
    return None


def main(args):
    clear_model_memory()
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    
    # 自动提取 step 信息（如果未手动指定）
    step = args.step
    if step is None:
        step = extract_step_from_path(args.model_path)
    if step is None:
        step = extract_step_from_name(args.model_name)
    
    # 构建保存目录
    if step is not None:
        # 按 step 组织: {save_path_dir}/step_{step}/
        save_path_dir = os.path.join(args.save_path_dir, f"step_{step}")
        display_name = f"{args.model_name} (step {step})"
    else:
        # 兼容旧模式: {save_path_dir}/{model_name}/
        save_path_dir = os.path.join(args.save_path_dir, args.model_name)
        display_name = args.model_name
    
    os.makedirs(save_path_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"评测配置:")
    print(f"  模型名称: {display_name}")
    print(f"  模型路径: {args.model_path}")
    print(f"  数据集: {args.dataset}")
    print(f"  保存目录: {save_path_dir}")
    print(f"  GPU: {gpu_ids}")
    print(f"  温度: {args.temperature}")
    print(f"  采样数: {args.n_samples}")
    if step is not None:
        print(f"  Step: {step}")
    print(f"=" * 60)
    
    # 检查是否已完成
    result_file = os.path.join(save_path_dir, f'{args.dataset}_responses.parquet')
    if os.path.exists(result_file) and not args.overwrite:
        print(f"结果文件已存在，跳过: {result_file}")
        print(f"如需覆盖，请使用 --overwrite 参数")
        return
    
    dataset_path = os.path.join(args.data_path_dir, f'{args.dataset}.parquet')
    if not os.path.exists(dataset_path):
        raise ValueError(f"dataset:{args.dataset} not found at {dataset_path}")
    
    dataset = pd.read_parquet(dataset_path)
    print(f'加载 {len(dataset)} 条数据从 {dataset_path}')
    
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
        stop_token_ids=[tokenizer.eos_token_id]
    )
    n_samples = args.n_samples

    print(f'采样参数: max_tokens={max_tokens}, temperature={args.temperature}, n_samples={n_samples}')
    print(f'开始为 {len(dataset)} 条数据生成响应...')
    
    chat_lst = dataset['prompt'].tolist()
    response_lst = [[] for _ in range(n_samples)]
    prompts = []
    batch_size = args.batch_size
    num_batch = -(- len(dataset) // batch_size)
    
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
                        enable_thinking=False
                    )
                else:
                    formatted_chat = tokenizer.apply_chat_template(
                        chat,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                inputs.append(formatted_chat)
                if random.randint(0,1000) == 0:
                    print(f'{formatted_chat=}')
            prompts.extend(inputs)
        else:
            raise NotImplementedError('模型不支持 chat_template')
        
        for n_sample in range(n_samples):
            progress = f"[Step {step}] " if step else ""
            print(f'{progress}数据集[{args.dataset}] 生成响应 {n_sample+1}/{n_samples}, 批次 {batch_idx+1}/{num_batch}', flush=True)
            outputs = model.generate(inputs, sampling_params=sample_params, use_tqdm=False)
            outputs = [out.outputs[0].text for out in outputs]

            n_outputs = []
            for i in range(len(outputs)):
                response_item = outputs[i]
                n_outputs.append(response_item)

            response_lst[n_sample].extend(n_outputs)
    
    response_lst = np.array(response_lst, dtype=object)
    response_lst = np.transpose(response_lst, axes=(1, 0)).tolist()
    prompts = np.array(prompts, dtype=object)
    
    assert len(response_lst) == len(dataset) == len(prompts), \
        f'长度不匹配: response_lst={len(response_lst)}, dataset={len(dataset)}, prompts={len(prompts)}'
    
    dataset['responses'] = response_lst
    dataset['formatted_prompt'] = prompts
    
    # 添加元数据
    if step is not None:
        dataset['step'] = step
    # dataset['model_name'] = args.model_name
    # dataset['model_path'] = args.model_path
    
    print(f'为 {len(dataset)} 条数据生成了 {len(response_lst[0])} 个响应')
    
    # 原子写入
    data_path = os.path.join(save_path_dir, f'{args.dataset}_responses.parquet')
    tmp_path = data_path + '.tmp'
    dataset.to_parquet(tmp_path)
    os.rename(tmp_path, data_path)
    print(f'响应已保存至: {data_path}')
    
    # 保存评测元信息
    meta_info = {
        'model_name': args.model_name,
        'model_path': args.model_path,
        'dataset': args.dataset,
        'step': step,
        'n_samples': args.n_samples,
        'temperature': args.temperature,
        'max_gen_len': max_tokens,
        'max_position_embeddings': max_pos_emb,
        'qwen3_post_trained': qwen3_post_trained,
        'num_examples': len(dataset),
    }
    meta_path = os.path.join(save_path_dir, f'{args.dataset}_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta_info, f, indent=2)
    print(f'元信息已保存至: {meta_path}')
    
    # 清理模型显存
    del model
    clear_model_memory()
    
    print(f"=" * 60)
    print(f"评测完成: {display_name} on {args.dataset}")
    print(f"=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测单个 checkpoint step 的数学能力")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--model_name", type=str, required=True,
                        help="模型名称 (用于显示和保存)")
    parser.add_argument("--dataset", type=str, default='aime24',
                        help="评测数据集名称")
    parser.add_argument("--save_path_dir", type=str, required=True,
                        help="结果保存根目录")
    parser.add_argument("--data_path_dir", type=str, default='/home/ycy/data1/data',
                        help="数据集所在目录")
    parser.add_argument("--step", type=int, default=None,
                        help="训练 step (可选，自动从路径/名称提取)")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="批处理大小")
    parser.add_argument("--max_gen_len", type=int, default=4096,
                        help="最大生成长度")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="每个问题采样次数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的结果文件")

    args = parser.parse_args()
    main(args)
