import json
import requests
from tqdm import tqdm
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_list", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--save_path_dir", type=str, default="/root/users/ycy/saved_results/evaluation")
args = parser.parse_args()

api_urls=['https://fast.ominiai.cn/v1/chat/completions']
api_keys=['sk-m7dpHEcWQkQlgoRuOm9S0mYeurwC9BTMpmLnXDrpPOmmkn98','sk-Lplron1sCtErNjSaUDxhgwFuwCyjGKfwJpa9cwFNXcJsMNyB']


def process_example(pred, gt):
    try:
        example = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a math answer checker."},
                {"role": "user", "content": f"Hi, there is a answer: {pred}\n\n, and the ground truth answer is: {gt}\n\n, please check whether the answer is correct or not, and return the **only** Yes or No."}
            ],
            "temperature": 0.1
        }
        api_index = random.randint(0, len(api_urls)-1)
        key_index = random.randint(0, len(api_keys)-1)
        api_url = api_urls[api_index]
        api_key = api_keys[key_index]
        gpt_response = requests.post(api_url, headers={"Authorization": f'Bearer {api_key}',"Content-Type": "application/json"}, json=example, timeout=20)
        return gpt_response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return "No"
new_results = []
old_results = []
base_path_dir='/root/users/ycy/models/shares'
solver_path_dir='/root/users/ycy/saved_results/Solver'
for model_name in [args.model_list]:
    save_path=os.path.join(args.save_path_dir, model_name)
    for dataset in [
    "math",
    "gsm8k", 
    "amc",
    "minerva",
    "olympiad",
    "aime2024",
    "aime2025",
    ]:
        
        if not os.path.exists(f'{save_path}/{dataset}_rule_based_eval_results.json'):
            print(f"Warning: {dataset}_rule_based_eval_results.json not found")
            continue
        with open(f'{save_path}/{dataset}_rule_based_eval_results.json', 'r') as f:
            results = json.load(f)
        old_results.append({
            'model': model_name,
            'dataset': dataset,
            'score': results[-1]['average_score']
        })
        for i in tqdm(range(len(results)-1)):
            # 确保所有结果都有 checked_score 键
            results[i]['checked_score']=results[i]['score']
            results[i]['is_checked']=False
            
            
            if results[i]['score'] < 0.5:
                gpt_check = process_example(results[i]['response'],results[i]['answer'])
                if "yes" in gpt_check.lower():
                    results[i]['checked_score']=1
                    results[i]['is_checked']=True
        with open(f'{save_path}/{dataset}_rsp_checked_eval_results.jsonl', 'w') as f:
            for result in results[:-1]:
                f.write(json.dumps(result,ensure_ascii=False) + '\n')
        
        new_results.append({
            'model': model_name,
            'dataset': dataset,
            'score': round(sum([result['checked_score'] for result in results[:-1]])/len(results[:-1])*100, 2)
        })
    print(f'final results for {model_name} have been saved in {args.save_path}/final_results.jsonl')
    print(f'final results: {new_results}')
    print(f'old results: {old_results}')
    with open(f'{save_path}/rule_based_final_results.jsonl', 'w') as f:
        for result in old_results:
            f.write(json.dumps(result,ensure_ascii=False) + '\n')
    with open(f'{save_path}/rsp_checked_final_results.jsonl', 'w') as f:
        for result in new_results:
            f.write(json.dumps(result,ensure_ascii=False) + '\n')
        





