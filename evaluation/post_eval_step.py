import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import argparse
import numpy as np
import random
import torch
import gc
import requests
from mathruler.grader import grade_answer
from tenacity import retry, stop_after_attempt, wait_fixed

ANSWER_PATTERN_MULTICHOICE = r"(?:\$\$\s*)?\\boxed\{[^}]*?([A-Z])[^}]*\}(?:\s*\$\$)?|(?:\*{0,2}\s*)?(?:Final|Correct)\s*Answer:\s*([A-Z])\."
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
ANSWER_PATTERN_BOXED = r"(?i)\\boxed\s*{([^\n]+)}"

api_urls=['https://fast.ominiai.cn/v1/chat/completions']
api_keys=['sk-Lplron1sCtErNjSaUDxhgwFuwCyjGKfwJpa9cwFNXcJsMNyB']
ENABLE_API_CHECK = False

def clear_model_memory():
    """清理 GPU 显存"""
    torch.cuda.empty_cache()
    gc.collect()
    print("模型显存已清理")


def extract_step_from_name(model_name):
    """从模型名称中提取 step 数字（格式：xxx-stepN）"""
    match = re.search(r'-step(\d+)$', model_name)
    if match:
        return int(match.group(1))
    return None


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def process_example(preds, gt):
    try:
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
        gpt_response = requests.post(api_url, headers={"Authorization": f'Bearer {api_key}',"Content-Type": "application/json"}, json=example, timeout=20)
        gpt_response.raise_for_status()
        return gpt_response.json()['choices'][0]['message']['content'], None
    except Exception as e:
        print(f"Error in process_example (attempt failed, will retry if attempts remain): {e}")
        raise e


def extract_boxed_content(text: str) -> str:
    """Extracts answers in \\boxed{}."""
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

            if depth == -1:
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
    check_rsp = None
    error = None
    check_type = "not checked"
    
    if pred is None:
        if not ENABLE_API_CHECK:
            return rule_correct, check_correct, is_check, str(pred), check_rsp, error, check_type
        is_check = True
        for gt in gts:
            check_type = "rsp_check"
            try:
                check_rsp, error = process_example(str(response), str(gt))
                if 'yes' in check_rsp.lower():
                    check_correct = True
                    break
            except Exception as e:
                print(f"All retries failed for response check: {e}")
                check_rsp = "No"
                error = {'error': f"All retries failed: {str(e)}"}
                break
        return rule_correct, check_correct, is_check, str(pred), check_rsp, error, check_type
        
    for gt in gts:    
        current_rule_correct = grade_answer(str(pred), str(gt))
        if current_rule_correct:
            rule_correct = True
            check_correct = True
            break

        if not ENABLE_API_CHECK:
            continue

        check_type = "pred_check"
        try:
            check_rsp, error = process_example(str(pred), str(gt))
            is_check = True
            if 'yes' in check_rsp.lower():
                check_correct = True
                break
        except Exception as e:
            print(f"All retries failed for pred check: {e}")
            check_rsp = "No"
            error = {'error': f"All retries failed: {str(e)}"}
            is_check = True
    
    return rule_correct, check_correct, is_check, str(pred), check_rsp, error, check_type


def process_data_item(args):
    i, idx, data_source, problem, formatted_prompt, responses, gts = args
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
        rule_correct, check_correct, is_check, pred, check_rsp, error, check_type = compute_score(rsp, gts)
        is_checks.append(is_check)
        rule_scores.append(float(rule_correct))
        checked_scores.append(float(check_correct))
        preds.append(pred)
        check_rsp_lst.append(check_rsp)
        error_lst.append(error)
        check_type_lst.append(check_type)
        rsp_lst.append({
            'rsp_idx': rsp_idx,
            'response_str': rsp,
            'pred': pred,
            'gt': gts,
            'is_rule_correct': rule_correct,
            'is_check_correct': check_correct,
            'is_checked': is_check,
            'check_rsp': check_rsp,
            'error': error,
            'check_type': check_type
        })
    
    res = {
        'i': i,
        'idx': idx,
        'data_source': data_source,
        'problem': problem,
        'formatted_prompt': formatted_prompt,
        'rule_scores': rule_scores,
        "checked_scores": checked_scores,
        'ground_truth': gts,
        'preds': preds,
        'is_gpt_checks': is_checks,
        'check_rsp_lst': check_rsp_lst,
        'rsp_info': rsp_lst,
        'error_lst': error_lst,
        'check_type_lst': check_type_lst,
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


def post_eval(save_path_dir, dataset_name, model_name, n_samples, temperature, step=None):
    """
    后处理评测结果
    
    Args:
        save_path_dir: 结果保存根目录
        dataset_name: 数据集名称
        model_name: 模型名称
        n_samples: 采样数
        temperature: 采样温度
        step: 训练 step (可选，自动从 model_name 提取)
    """
    # 自动提取 step
    if step is None:
        step = extract_step_from_name(model_name)
    
    # 根据是否有 step 确定目录结构
    if step is not None:
        # 新目录结构: {save_path_dir}/step_{step}/
        result_dir = os.path.join(save_path_dir, f"step_{step}")
        display_name = f"{model_name} (step {step})"
    else:
        # 兼容旧目录结构: {save_path_dir}/{model_name}/
        result_dir = os.path.join(save_path_dir, model_name)
        display_name = model_name
    
    print("=" * 60)
    print(f"Post Evaluation")
    print(f"  结果目录: {result_dir}")
    print(f"  数据集: {dataset_name}")
    print(f"  模型: {display_name}")
    print(f"  Step: {step if step else 'N/A'}")
    print(f"  采样数: {n_samples}")
    print(f"  温度: {temperature}")
    print(f"  外部API校验: {'开启' if ENABLE_API_CHECK else '关闭'}")
    print("=" * 60)
    
    save_path = os.path.join(result_dir, f'{dataset_name}_responses.parquet')
    if not os.path.exists(save_path):
        raise ValueError(f"responses for {dataset_name} not found in {save_path}")
    
    dataset = pd.read_parquet(save_path)
    print(f'加载响应文件: {save_path}, 共 {len(dataset)} 条数据')
    
    assert 'formatted_prompt' in dataset.columns and 'reward_model' in dataset.columns and \
           'data_source' in dataset.columns and 'problem' in dataset.columns and \
           'responses' in dataset.columns, \
           f'数据集列不正确，请检查数据集'
   
    final_results = []
   
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 100)) as executor:
        args = [
            (i, data_item['extra_info']['idx'], data_item['data_source'], data_item['problem'], 
             data_item['formatted_prompt'], data_item['responses'], data_item['reward_model']['ground_truth'])
            for i, (_, data_item) in enumerate(dataset.iterrows())
        ]
        futures = [executor.submit(process_data_item, arg) for arg in args]
        results = pd.DataFrame([future.result() for future in futures])
        
        for data_source, data in results.groupby('data_source'):
            data = data.to_dict(orient='records')
            data = [convert_to_json_serializable(item) for item in data]
            data = sorted(data, key=lambda x: x['idx'])
            
            output_path = os.path.join(result_dir, f'{data_source}_eval_results.jsonl')
            with open(output_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f'[{data_source}] 评测结果已保存: {output_path}')
            
            # 计算统计结果
            data_source_len = len(data)
            rule_score = sum([example['rule_scores'][0] for example in data]) / data_source_len if data_source_len > 0 else 0
            mean_rule_score = sum([
                sum([example['rule_scores'][n_sample] for example in data]) / data_source_len if data_source_len > 0 else 0
                for n_sample in range(n_samples)
            ]) / n_samples
            sample_mean = sum([sum(example['checked_scores']) for example in data]) / (data_source_len * n_samples) if data_source_len > 0 else 0
            checked_scores = sum([example['checked_scores'][0] for example in data]) / data_source_len if data_source_len > 0 else 0
            mean_checked_score = sum([
                sum([example['checked_scores'][n_sample] for example in data]) / data_source_len if data_source_len > 0 else 0
                for n_sample in range(n_samples)
            ]) / n_samples
            
            result_item = {
                'data_source': data_source,
                'model': model_name,
                'step': step,
                'rule@first': f'{rule_score*100:.2f}',
                f'rule_mean@{n_samples}': f'{mean_rule_score*100:.2f}',
                f'checked_sample_mean@{n_samples}': f'{sample_mean*100:.2f}',
                'checked@first': f'{checked_scores*100:.2f}',
                f'checked_mean@{n_samples}': f'{mean_checked_score*100:.2f}',
                "n_samples": n_samples,
                "temperature": temperature,
                }
            final_results.append(result_item)
        
            # 打印结果摘要
            print(f'  [{data_source}] rule@first: {rule_score*100:.2f}%, checked@first: {checked_scores*100:.2f}%')

    overall_results_path = os.path.join(result_dir, f'{dataset_name}_Overall_results.jsonl')
    with open(overall_results_path, "w") as f:
        for line in final_results:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f'总体结果已保存: {overall_results_path}')
    
    print("=" * 60)
    print(f"Post Evaluation 完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对 eval_all_math_step.py 生成的结果进行后处理评估")
    parser.add_argument("--save_path_dir", type=str, required=True,
                        help="结果保存根目录")
    parser.add_argument("--dataset", type=str, default='greedy_data',
                        help="数据集名称")
    parser.add_argument("--model_name", type=str, required=True,
                        help="模型名称 (用于显示，可包含 step 信息)")
    parser.add_argument("--step", type=int, default=None,
                        help="训练 step (可选，自动从 model_name 提取)")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="每个问题的采样数")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="采样温度")
    parser.add_argument("--enable_api_check", action="store_true",
                        help="启用外部 API 进行额外答案校验（默认关闭，仅规则判分）")
    
    args = parser.parse_args()
    ENABLE_API_CHECK = args.enable_api_check
    post_eval(args.save_path_dir, args.dataset, args.model_name, args.n_samples, args.temperature, args.step)
