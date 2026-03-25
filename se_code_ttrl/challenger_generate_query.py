import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import vllm
import torch
from transformers import AutoTokenizer, AutoConfig
import argparse
from typing import List
from vllm.outputs import RequestOutput
import os, sys
import random
import json
import regex as re
from se_code_ttrl.Challenger_dataset import get_prompts
from se_code_ttrl.reward_manager import custom_extract_boxed_content


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
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 检查是否需要禁用 thinking（后训练的 Qwen3 模型，Base 模型不需要）
    qwen3_post_trained = is_qwen3_post_trained(args.model)
    if qwen3_post_trained:
        print(f"检测到后训练的 Qwen3 模型，将使用 enable_thinking=False")
    
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.8,
        seed=int(args.suffix),
    )
    
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    
    # 为每个样本重复所有topic的prompts
    dataframe=get_prompts(num_querys=args.num_samples, get_prompts_func=args.get_prompts_func, icl_files=args.train_file)
    prompt = [
            tokenizer.apply_chat_template(
                example['prompt'], 
                tokenize=False,
                add_generation_prompt=True, 
                add_special_tokens=True,
                **({"enable_thinking": False} if qwen3_post_trained else {})
            ) 
            for example in dataframe
        ]
    if random.randint(0,64)==0:
        print(f'{prompt[0]=}')
    completions: List[RequestOutput] = model.generate(prompt, sampling_params=sample_params,use_tqdm=False)
    results=[]
    
    for idx, completion in enumerate(completions):
        response = completion.outputs[0].text
        reference_question = dataframe[idx].get('reference_question','')
        test_item = dataframe[idx].get('test_item',None)
        data_source = dataframe[idx].get('data_source','')
        try:
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            #answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if args.get_prompts_func == "R_Zero" or args.get_prompts_func == "ttrl_icl" or args.get_prompts_func == "weakness_icl":
                answers = custom_extract_boxed_content(response)
            elif args.get_prompts_func == "weakness":
                answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
            else:
                answers="None"
                print("Warning: get_prompts_func is not supported extracted answers, default to None")

            if questions and answers:
                question = questions[-1].strip()                
                answer = answers.strip()
                results.append({
                    "idx": idx, 
                    "data_source": data_source,
                    'prompt': prompt[idx],
                    'reference_question': reference_question,
                    'response':response,
                    "question": question,                   
                    'answer': answer,
                    "score": 0,
                    'is_synthetic': True
                })
            else:
                results.append({
                    "idx": idx, 
                    "data_source": data_source,
                    'prompt': prompt[idx],
                    'reference_question': reference_question,
                    'response':response,
                    "question": '', 
                    'answer': '',
                    "score": -1
                })
            if test_item is not None:
                # test_item 已经在 Challenger_dataset.py 中被清理为可 JSON 序列化的格式
                # 这里直接使用即可
                results.append({
                    "example": test_item,
                    "is_synthetic": False,
                    "score": 0,
                })
        except:
            results.append({
                "idx": idx, 
                'prompt': prompt[idx],
                'reference_question': reference_question,
                'response':response,
                "data_source": data_source,
                "question": '', 
                'answer': '',
                "score": -1
            })
    random.shuffle(results)
    os.makedirs(args.storage_path, exist_ok=True)
    with open(f"{args.storage_path}/{args.suffix}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/users/ycy/models/shares/Qwen3-4B-Base")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--suffix", type=str, default="1", help="Suffix to add to the output file")
    parser.add_argument("--storage_path", type=str, default="/root/users/ycy/Self-evolving-Agent/se_code", help="")
    parser.add_argument("--get_prompts_func", type=str, default="R_Zero", help="Function to get prompts")
    parser.add_argument("--train_file", type=str, default="", help="Train file")
    #parser.add_argument("--save_name", type=str, default="challenger_generated_question", help="")
    args = parser.parse_args()
    print(f"[train_file]: {args.train_file}")


    main(args) 