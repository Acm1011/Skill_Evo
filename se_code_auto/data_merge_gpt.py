import os
import json
import time
import random
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_fixed



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


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def process_example(q):
    """处理单个问题，带重试机制
    
    Args:
        q: 问题文本
    
    重试机制：失败时会自动重试3次，每次间隔5秒
    """
    prompt="""Hi, there is a question: {q} 
Please read this question carefully and analyze its content.
1. Identify any logical inconsistencies or contradictions in the question.
2. If any are found, skip answering and return the response as "\\boxed{{\\text{{No Answer}}}}".
3. If the question is logically sound, reason step by step, and put your final answer within \\boxed{{}}.
"""
    example = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt.format(q=q)}
        ],
        "temperature": 0.1
    }
    api_urls=['https://fast.ominiai.cn/v1/chat/completions']
    api_keys=['sk-Lplron1sCtErNjSaUDxhgwFuwCyjGKfwJpa9cwFNXcJsMNyB']
    
    # 移除try-except，让异常抛出以触发重试机制
    api_index = random.randint(0, len(api_urls)-1)
    key_index = random.randint(0, len(api_keys)-1)
    api_url = api_urls[api_index]
    api_key = api_keys[key_index]
    
    gpt_response = requests.post(
        api_url, 
        headers={"Authorization": f'Bearer {api_key}',"Content-Type": "application/json"}, 
        json=example, 
        timeout=20
    )
    return gpt_response.json()['choices'][0]['message']['content'], None, q
    

def convert_to_serializable(obj):
    """将对象转换为JSON可序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def data_merge(args):
    data_path_dir = args.data_path_dir
    save_path_dir = args.save_path_dir
    data_list = os.listdir(data_path_dir)
    datas = []
    for data_file in data_list:
        if data_file.endswith(".json"):
            with open(os.path.join(data_path_dir, data_file), "r") as f:
                datas.extend(json.load(f))
    os.makedirs(save_path_dir, exist_ok=True)
    
    # 指令
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    
    # 首先构建初始数据（只包含score=0的）
    initial_data = []
    for idx, item in enumerate(datas):
        if item["score"] == 0:
            initial_data.append({
                'data_source': f'Challenger_{args.exp_name}',
                'topic': item['topic'],
                'level': item['level'],
                'prompt': [
                    {
                        'role': 'system',
                        'content': instruction
                    },
                    {
                        "role": "user",
                        'content': item["question"] 
                    }
                ],
                'reward_model': {
                    'style': 'rule', 
                },
                'ability': 'math',
                'extra_info': {
                    'idx': idx,
                    'question': item["question"],
                    'score': item["score"]
                }
            })
    
    print(f"成功读取 {len(initial_data)} 条数据 (score=0)")
    
    # 保存初始的 train_data_raw（未过滤）
    with open(f'{save_path_dir}/train_data_raw.jsonl', 'w', encoding='utf-8') as f:
        for line in initial_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    df = pd.DataFrame(initial_data)
    df.to_parquet(f'{save_path_dir}/train_data_raw.parquet')
    print(f"已保存初始数据到 {save_path_dir}/train_data_raw.parquet")
    
    # ============== 并发处理和过滤 ==============
    
    # 提取所有question
    questions = [item['extra_info']['question'] for item in initial_data]
    print(f"准备并发处理 {len(questions)} 个问题...")
    
    # 并发处理所有问题
    results = []
    concurrent_results_path = os.path.join(save_path_dir, 'concurrent_results.jsonl')
    
    # 检查是否已存在并发结果文件
    if os.path.exists(concurrent_results_path):
        print(f"发现已存在的并发结果文件，正在读取: {concurrent_results_path}")
        with open(concurrent_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                result_dict = json.loads(line.strip())
                # 将结果转换回元组格式 (gpt_response, error, q)
                results.append((result_dict['gpt_response'], result_dict['error'], result_dict['question']))
        print(f"成功读取 {len(results)} 条已保存的并发结果")
    else:
        print("开始并发处理...")
        max_workers = 1000  # 并发线程数
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(process_example, q): idx for idx, q in enumerate(questions)}
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    # 每处理1000个打印一次进度
                    if completed_count % 1000 == 0:
                        print(f"已完成: {completed_count}/{len(questions)}")
                        
                except Exception as e:
                    print(f"处理问题 {idx} 时发生异常: {e}")
                    results.append(("No", f"Exception: {e}", questions[idx]))
                    completed_count += 1
        
        print(f"并发处理完成，共处理 {len(results)} 条数据")
        
        # 保存并发结果为jsonl格式
        print(f"正在保存并发结果到: {concurrent_results_path}")
        with open(concurrent_results_path, 'w', encoding='utf-8') as f:
            for gpt_response, error, question in results:
                result_dict = {
                    'gpt_response': gpt_response,
                    'error': error,
                    'question': question
                }
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        print(f"并发结果已保存")
    
    # 构建question到result的映射（用于匹配，因为并发结果顺序被打乱）
    print("正在构建question到result的映射...")
    question_to_result = {}
    for gpt_response, error, q in results:
        if q:  # 跳过空question
            question_to_result[q] = (gpt_response, error)
    print(f"映射构建完成，共 {len(question_to_result)} 条结果")
    
    # 将结果添加到data中，并过滤无效数据
    filtered_data = []
    not_found_count = 0
    error_count = 0
    no_answer_count = 0
    empty_boxed_count = 0
    
    for i, item in enumerate(initial_data):
        question = item['extra_info']['question']
        
        # 通过question查找对应的result
        if question not in question_to_result:
            not_found_count += 1
            if not_found_count <= 5:  # 只打印前5个
                print(f"警告: 问题未找到对应结果: {question[:100]}...")
            continue
        
        gpt_response, error = question_to_result[question]
        
        # 如果发生错误，跳过这条数据
        if error is not None:
            error_count += 1
            continue
        
        # 提取boxed内容
        boxed_answers = extract_boxed_content(gpt_response)
        if not boxed_answers:
            empty_boxed_count += 1
            continue
        
        # 检查是否是No Answer
        if "no answer" in boxed_answers.lower():
            no_answer_count += 1
            continue
        
        # 将提取的答案存储到ground_truth字段中（列表形式）
        item['reward_model']['ground_truth'] = [str(boxed_answers)]
        item['extra_info']['gpt_response'] = str(gpt_response)
        filtered_data.append(item)
        
        # 打印进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(initial_data)} 条数据 (过滤: {not_found_count + error_count + no_answer_count + empty_boxed_count})")
    
    # 统计信息
    print(f"\n处理完成，共 {len(initial_data)} 条数据")
    print(f"过滤掉:")
    print(f"  - {not_found_count} 条未找到结果")
    print(f"  - {error_count} 条错误数据")
    print(f"  - {empty_boxed_count} 条无法提取boxed答案")
    print(f"  - {no_answer_count} 条No Answer数据")
    print(f"保留: {len(filtered_data)} 条有效数据")
    
    # 保存过滤后的数据为jsonl文件
    output_jsonl = os.path.join(save_path_dir, 'train_data.jsonl')
    print(f"\n正在保存过滤后的jsonl文件: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            # 将item转换为可序列化的格式
            serializable_item = convert_to_serializable(item)
            f.write(json.dumps(serializable_item, ensure_ascii=False) + '\n')
    print(f"已保存jsonl文件")
    
    # 保存过滤后的数据为parquet文件
    output_parquet = os.path.join(save_path_dir, 'train_data.parquet')
    print(f"正在保存过滤后的parquet文件: {output_parquet}")
    pd.DataFrame(filtered_data).to_parquet(output_parquet, index=False)
    print(f"已保存parquet文件")
    
    print("\n全部完成！")


            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_dir", type=str, default="", help="")
    parser.add_argument("--save_path_dir", type=str, default="", help="")
    parser.add_argument("--exp_name", type=str, default="", help="")
    args = parser.parse_args()
    data_merge(args)
    
    