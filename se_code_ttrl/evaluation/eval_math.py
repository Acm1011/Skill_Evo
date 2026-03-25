import vllm
import argparse
from transformers import AutoTokenizer
import json
from math_verify import parse, verify
import os
import pandas as pd
import numpy as np
import re
import random
import requests
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#STORAGE_PATH = os.getenv("STORAGE_PATH")
#/users/ycy/saved_results'
ANSWER_PATTERN_MULTICHOICE = r"(?:\$\$\s*)?\\boxed\{[^}]*?([A-Z])[^}]*\}(?:\s*\$\$)?|(?:\*{0,2}\s*)?(?:Final|Correct)\s*Answer:\s*([A-Z])\."
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
ANSWER_PATTERN_BOXED = r"(?i)\\boxed\s*{([^\n]+)}"

api_urls = ["http://199.68.217.242:5896/v1/chat/completions"]
api_keys=['sk-E77SOhSu4J1zTCRQVypzVtdoYRP5nX9I3cWlbtwppJcrLcXU']



def process_example(answer, response):
    try:
        example = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a math answer checker."},
                {"role": "user", "content": f"Hi, there is a answer: {answer}\n\n, and the ground truth answer is: {response}\n\n, please check whether the answer is correct or not, and return the **only** Yes or No."}
            ],
            "temperature": 0.1
        }
        api_index = random.randint(0, len(api_urls)-1)
        api_url = api_urls[api_index]
        api_key = api_keys[api_index]
        gpt_response = requests.post(api_url, headers={"Authorization": f'Bearer {api_key}',"Content-Type": "application/json"}, json=example, timeout=20)
        return gpt_response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return "No"
def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()
    return None

def compute_score(response: str, gts: list):
    pred = extract_boxed_content(response[-300:])
    if not isinstance(gts, list):
        gts = [gts]
    
    # 如果预测为空，直接返回
    if pred is None:
        return False, False, False, 'None'
    
    rule_correct = False
    is_check = False
    check_correct = False
    
    # 检查是否任何一个gt匹配成功
    for gt in gts:    
        # 规则匹配
        current_rule_correct = verify(parse(str(gt)), parse(str(pred)))
        if current_rule_correct:
            rule_correct = True
            check_correct = True
            break  # 找到匹配就退出
        
        # 如果规则匹配失败，尝试LLM检查
        check_rsp = process_example(str(pred), str(gt))
        if 'yes' in check_rsp.lower():
            check_correct = True
            is_check = True
            break  # 找到匹配就退出
    
    return rule_correct, check_correct, is_check, str(pred)




def main(args):
    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    print(f'eval model: {args.model} performance on the math dataset: {args.dataset}, eval results will be saved in {args.save_path}, using GPUs: {gpu_ids}')
    assert args.dataset in ['aime24','aime25','amc23','minerva','olympiadbench','math500'], f"dataset:{args.dataset} should be one of aime24, aime25, amc23, minerva, olympiadbench, math500"
    dataset_path_dir = '/root/users/ycy/data'
    dataset_path = os.path.join(dataset_path_dir, f'{args.dataset}.parquet')
    if not os.path.exists(dataset_path):
        raise ValueError(f"dataset:{args.dataset} not found")
    dataset = pd.read_parquet(dataset_path)
    print(f'load {len(dataset)} examples from {dataset_path} successfully')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.95
    )
    if 'aime' in args.dataset or 'amc' in args.dataset: 
        sample_params = vllm.SamplingParams(
            max_tokens=args.max_gen_len,
            temperature=1.0,
            top_p=0.95,
            top_k=50,
            stop_token_ids=[tokenizer.eos_token_id],)
        n_samples = 32
    else:
        sample_params = vllm.SamplingParams(
            max_tokens=args.max_gen_len,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            stop_token_ids=[tokenizer.eos_token_id],)
        n_samples = 1
    
    
    chat_lst = dataset['prompt'].tolist()
    response_lst = [[] for _ in range(n_samples)]
    os.makedirs(args.save_path, exist_ok=True)

    batch_size=args.batch_size
    num_batch = -(- len(dataset) // batch_size)
    #chats=[[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": question}] for question in questions]
    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process {args.dataset} dataset with model {args.model}.")
        batch_chats = chat_lst[batch_idx * batch_size: (batch_idx+1)*batch_size]
        if tokenizer.chat_template:
            inputs = []
            for chat in batch_chats:
                formatted_chat = tokenizer.apply_chat_template(
                    chat,
                add_generation_prompt=True,
                tokenize=False,
                add_special_tokens=True
            )
                inputs.append(formatted_chat)
        else:
            raise NotImplementedError(f'not support ')
        
        for n_sample in range(n_samples):
            outputs = model.generate(inputs, sampling_params=sample_params,use_tqdm=False)
            outputs = [out.outputs[0].text for out in outputs]

            n_outputs = []
            for i in range(len(outputs)):
                response_item = outputs[i]
                n_outputs.append(response_item)

            response_lst[n_sample].extend(n_outputs)
    response_lst = np.array(response_lst, dtype=object)
    response_lst = np.transpose(response_lst, axes=(1, 0)).tolist()
    dataset['responses'] = response_lst
    final_results=[]
    for data_source, data in dataset.groupby('data_source'):
        results = []
        for i, (_, data_item) in enumerate(data.iterrows()):
            responses = data_item['responses']
            gts = data_item['reward_model']['ground_truth']
            # 确保gts是Python原生类型
            if isinstance(gts, np.ndarray):
                gts = gts.tolist()

            rule_scores = []
            checked_scores = []
            preds = []
            is_checks = []
            rsp_lst = []
            for idx, rsp in enumerate(responses):
                rule_correct, check_correct, is_check, pred = compute_score(rsp, gts)
                is_checks.append(is_check)
                rule_scores.append(float(rule_correct))
                checked_scores.append(float(check_correct))
                preds.append(pred)
                rsp_lst.append({
                    'rsp_idx':idx,
                    'response_str': rsp,
                    'pred':pred,
                    'gt':gts,
                    'is_rule_correct': rule_correct,
                    'is_check_correct': check_correct,
                    'is_checked':is_check
                })
            results.append({
                'idx':i,
                'data_source':data_source,
                'problem': data_item['extra_info']['problem'],
                'prompt':data_item['prompt'],
                'rule_scores':rule_scores,
                "checked_scores":checked_scores,
                'ground_truth':gts,
                'preds':preds,
                'is_gpt_checks':is_checks,
                'rsp_info':rsp_lst,
            })
        
        # 确保所有数据都是JSON可序列化的
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        with open(f"{args.save_path}/{data_source}_eval_results.jsonl", "w") as f:
            for result in results:
                serializable_result = convert_to_json_serializable(result)
                f.write(json.dumps(serializable_result, ensure_ascii=False) + '\n')
        rule_score = sum([example['rule_scores'][0] for example in results]) / len(results)
        mean_rule_score = sum([
            sum([example['rule_scores'][n_sample] for example in results]) / len(results)
            for n_sample in range(n_samples)
        ]) / n_samples

        checked_scores = sum([example['checked_scores'][0] for example in results]) / len(results)
        mean_checked_score = sum([
            sum([example['checked_scores'][n_sample] for example in results]) / len(results)
                for n_sample in range(n_samples)
        ]) / n_samples

        final_results.append({
            'data_source':data_source,
            'model':args.model.split("/")[-1],
            'rule@first': f'{rule_score*100:.2f}',
            f'rule_mean@{n_samples}':f'{mean_rule_score*100:.2f}',
            'checked@first': f'{checked_scores*100:.2f}',
            f'checked_mean@{n_samples}': f'{mean_checked_score*100:.2f}',
        })
            

    with open(f"{args.save_path}/{args.dataset}_Overall_results.jsonl", "w") as f:
        for line in final_results:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default='aime24')
    parser.add_argument("--save_path", type=str, default="/root/users/ycy/saved_results/evaluation")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_gen_len", type=int, default=4096)    

    args = parser.parse_args()
    main(args)