import os
import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import argparse
import numpy as np
import random
import torch
import gc
import requests
from mathruler.grader import grade_answer
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

ANSWER_PATTERN_MULTICHOICE = r"(?:\$\$\s*)?\\boxed\{[^}]*?([A-Z])[^}]*\}(?:\s*\$\$)?|(?:\*{0,2}\s*)?(?:Final|Correct)\s*Answer:\s*([A-Z])\."
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
ANSWER_PATTERN_BOXED = r"(?i)\\boxed\s*{([^\n]+)}"

# api_urls = ["http://199.68.217.242:5896/v1/chat/completions"]
# api_keys=['sk-E77SOhSu4J1zTCRQVypzVtdoYRP5nX9I3cWlbtwppJcrLcXU']
api_urls=['https://fast.ominiai.cn/v1/chat/completions']
api_keys=['sk-Lplron1sCtErNjSaUDxhgwFuwCyjGKfwJpa9cwFNXcJsMNyB']

def clear_model_memory():

    # 清空GPU缓存
    torch.cuda.empty_cache()
    
    # 强制垃圾回收
    gc.collect()
    
    print("模型显存已清理")

@retry(
    stop=stop_after_attempt(5),  # 最多尝试5次
    wait=wait_fixed(3),  # 每次重试间隔3秒
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout, KeyError, ConnectionError)),  # 只在特定异常时重试
    reraise=True  # 最后一次重试失败后抛出异常
)
def process_example(preds, gt):
    """
    使用GPT-4o-mini检查答案是否正确
    
    重试策略：
    - 网络超时/连接错误：自动重试最多5次
    - API响应格式错误：自动重试
    - 其他错误：不重试，直接返回错误信息
    """
    example = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a math answer checker."},
            {"role": "user", "content": f"Hi, there is a answer: {gt}\n\n, and the ground truth answer is: {preds}\n\n, please check whether the answer is correct or not, and return the **only** Yes or No."}
        ],
        "temperature": 0.1
    }
    api_index = random.randint(0, len(api_urls)-1)
    key_index = random.randint(0, len(api_keys)-1)
    api_url = api_urls[api_index]
    api_key = api_keys[key_index]
    
    # 发送请求，让 retry 装饰器处理异常
    gpt_response = requests.post(
        api_url, 
        headers={"Authorization": f'Bearer {api_key}', "Content-Type": "application/json"}, 
        json=example, 
        timeout=30  # 增加超时时间到30秒
    )
    gpt_response.raise_for_status()  # 如果状态码不是2xx，抛出HTTPError
    
    # 解析响应
    response_data = gpt_response.json()
    content = response_data['choices'][0]['message']['content']
    
    return content, None
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
    rule_correct = False
    is_check = False
    check_correct = False
    check_rsp=None
    error=None
    check_type="not checked"
    # 如果预测为空，直接返回
    if pred is None:
        is_check = True
        for gt in gts:
            check_type="rsp_check"
            try:
                check_rsp,error = process_example(str(response), str(gt))
                if 'yes' in check_rsp.lower():
                    check_correct = True
                    break
            except Exception as e:
                # 重试5次后仍然失败，记录错误但继续处理
                print(f"Error after retries in process_example (rsp_check): {e}")
                check_rsp = "No"
                error = {'error': f"Failed after retries: {str(e)}"}
                break
        return rule_correct, check_correct, is_check, str(pred),check_rsp,error,check_type
        
    
    # 检查是否任何一个gt匹配成功
    for gt in gts:    
        # 规则匹配
        # 使用 parsing_timeout=None 和 timeout_seconds=None 以避免多线程环境中的 signal.alarm() 冲突
        current_rule_correct = grade_answer(str(pred), str(gt))
        if current_rule_correct:
            rule_correct = True
            check_correct = True
            break  # 找到匹配就退出
        
        # 如果规则匹配失败，尝试LLM检查
        check_type="pred_check"
        try:
            check_rsp,error = process_example(str(pred), str(gt))
            is_check = True
            if 'yes' in check_rsp.lower():
                check_correct = True
                break  # 找到匹配就退出
        except Exception as e:
            # 重试5次后仍然失败，记录错误但继续处理
            print(f"Error after retries in process_example (pred_check): {e}")
            check_rsp = "No"
            error = {'error': f"Failed after retries: {str(e)}"}
            is_check = True
            break
    
    return rule_correct, check_correct, is_check, str(pred),check_rsp,error,check_type

def process_data_item(args):
    #(i, data_item['extra_info']['idx'],data_item['data_source'], data_item['problem'], data_item['formatted_prompt'],
    #  data_item['responses'], data_item['reward_model']['ground_truth'])
    i,idx,data_source, problem, formatted_prompt,responses, gts = args
    # 确保gts是Python原生类型
    if isinstance(gts, np.ndarray):
        gts = gts.tolist()
    if not isinstance(gts, list):
        gts = [gts]
    rule_scores = []
    checked_scores = []
    preds = []
    is_checks = []
    rsp_lst = []
    check_rsp_lst = []
    error_lst = []
    check_type_lst = []
    for rsp_idx, rsp in enumerate(responses):
        rule_correct, check_correct, is_check, pred,check_rsp,error,check_type = compute_score(rsp, gts)
        is_checks.append(is_check)
        rule_scores.append(float(rule_correct))
        checked_scores.append(float(check_correct))
        preds.append(pred)
        check_rsp_lst.append(check_rsp)
        error_lst.append(error)
        check_type_lst.append(check_type)
        rsp_lst.append({
            'rsp_idx':rsp_idx,
            'response_str': rsp,
            'pred':pred,
            'gt':gts,
            'is_rule_correct': rule_correct,
            'is_check_correct': check_correct,
            'is_checked':is_check,
            'check_rsp':check_rsp,
            'error':error,
            'check_type':check_type
        })
    res={
        'i':i,
        'idx':idx,
        'data_source':data_source,
        'problem': problem,
        'formatted_prompt':formatted_prompt,
        'rule_scores':rule_scores,
        "checked_scores":checked_scores,
        'ground_truth':gts,
        'preds':preds,
        'is_gpt_checks':is_checks,
        'check_rsp_lst':check_rsp_lst,
        'rsp_info':rsp_lst,
        'error_lst':error_lst,
        'check_type_lst':check_type_lst,
    }
    return res

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



def post_eval(save_path_dir, dataset_name, model_name, n_samples, step):
    print(f'post_eval: save_path_dir: {save_path_dir}, dataset_name: {dataset_name}, model_name: {model_name}, n_samples: {n_samples}')
    final_results=[]
    save_path_dir = os.path.join(save_path_dir, model_name,f'step_{step}')
    
    data_path = os.path.join(save_path_dir, f'{dataset_name}_responses.parquet')
    if not os.path.exists(data_path):
        raise ValueError(f"responses for {dataset_name} not found in {data_path}")
    dataset = pd.read_parquet(data_path)
    print(f'responses for [{dataset_name}] generated by {model_name} with step [{step}] successfully, len(dataset): {len(dataset)}')
    assert 'formatted_prompt' in dataset.columns and 'reward_model' in dataset.columns and 'data_source' in dataset.columns and 'problem' in dataset.columns and 'responses' in dataset.columns, f'dataset columns are not correct, please check the dataset'
   
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(),100)) as executor:
        args = [
            (i, data_item['extra_info']['idx'],data_item['data_source'], data_item['problem'], data_item['formatted_prompt'], data_item['responses'], data_item['reward_model']['ground_truth'])
            for i, (_, data_item) in enumerate(dataset.iterrows())
        ]
        futures = [executor.submit(process_data_item, arg) for arg in args]
        results = pd.DataFrame([future.result() for future in futures])
        final_results = []
        for data_source, data in results.groupby('data_source'):
            data = data.to_dict(orient='records')
            data = [convert_to_json_serializable(item) for item in data]
            data = sorted(data, key=lambda x: x['idx'])
            output_path = os.path.join(save_path_dir,  f'{data_source}_eval_results.jsonl')
            with open(output_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f'model[{model_name}] with dataset[{dataset_name}] evaluate [{data_source}] dataset results saved in {output_path} successfully')
            
            # 计算每个data_source的统计结果（无需锁，因为future.result()是串行调用的）
            data_source_len = len(data)
            rule_score = sum([example['rule_scores'][0] for example in data]) / data_source_len if data_source_len > 0 else 0
            mean_rule_score = sum([
                sum([example['rule_scores'][n_sample] for example in data]) / data_source_len if data_source_len > 0 else 0
                for n_sample in range(n_samples)
            ]) / n_samples
            sample_mean = sum([sum(example['checked_scores']) for example in data])  / (data_source_len * n_samples) if data_source_len > 0 else 0
            checked_scores = sum([example['checked_scores'][0] for example in data]) / data_source_len if data_source_len > 0 else 0
            mean_checked_score = sum([
                sum([example['checked_scores'][n_sample] for example in data]) / data_source_len if data_source_len > 0 else 0
                for n_sample in range(n_samples)
            ]) / n_samples
            final_results.append({
                'data_source':data_source,
                'model':model_name,
                'rule@first': f'{rule_score*100:.2f}',
                f'rule_mean@{n_samples}':f'{mean_rule_score*100:.2f}',
                f'checked_sample_mean@{n_samples}': f'{sample_mean*100:.2f}',
                'checked@first': f'{checked_scores*100:.2f}',
                f'checked_mean@{n_samples}': f'{mean_checked_score*100:.2f}',
                "n_samples": n_samples,
                }
            )
        

    overall_results_path = os.path.join(save_path_dir,  f'{dataset_name}_Overall_results.jsonl')
    with open(overall_results_path, "w") as f:
        for line in final_results:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f'model[{model_name}] with dataset[{dataset_name}] Overall results saved in {overall_results_path} successfully')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path_dir", type=str, default="/root/users/ycy/saved_results/evaluation")
    parser.add_argument("--dataset", type=str, default='greedy_data')
    parser.add_argument("--model_name", type=str, default="Qwen3-4B-Base")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()
    post_eval(args.save_path_dir, args.dataset, args.model_name, args.n_samples, args.step)