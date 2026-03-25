import random
import requests
import pandas as pd
import os
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from mathruler.grader import extract_boxed_content
from tenacity import retry, stop_after_attempt, wait_fixed


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



model_path_dir='/root/users/ycy/saved_results/Solver'
model_name='entropy_prompt2_se-Zero_Qwen3-4B-Base-V2'
data_path = os.path.join(model_path_dir, model_name, 'train_data.parquet')
data = pd.read_parquet(data_path).to_dict(orient='records')

print(f"成功读取 {len(data)} 条数据")

# 备份原始数据为 train_data_raw（如果还没备份）
raw_data_path = os.path.join(model_path_dir, model_name, 'train_data_raw.parquet')
if not os.path.exists(raw_data_path):
    print(f"正在备份原始数据到: {raw_data_path}")
    pd.DataFrame(data).to_parquet(raw_data_path, index=False)
    # 同时保存jsonl格式
    raw_jsonl_path = os.path.join(model_path_dir, model_name, 'train_data_raw.jsonl')
    with open(raw_jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            serializable_item = convert_to_serializable(item)
            f.write(json.dumps(serializable_item, ensure_ascii=False) + '\n')
    print(f"原始数据已备份")
else:
    print(f"原始数据备份已存在: {raw_data_path}")

# 提取所有question
questions = [item['extra_info']['question'] for item in data]
print(f"准备并发处理 {len(questions)} 个问题...")

# 并发处理所有问题
results = []
final_concurrent_results_path = os.path.join(model_path_dir, model_name, 'concurrent_results.jsonl')

# 检查是否已存在并发结果文件
if os.path.exists(final_concurrent_results_path):
    print(f"发现已存在的并发结果文件，正在读取: {final_concurrent_results_path}")
    with open(final_concurrent_results_path, 'r', encoding='utf-8') as f:
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
                
                # 每处理100个打印一次进度
                if completed_count % 1000 == 0:
                    print(f"已完成: {completed_count}/{len(questions)}")
                    
            except Exception as e:
                print(f"处理问题 {idx} 时发生异常: {e}")
                results.append(("No", f"Exception: {e}", questions[idx]))
                completed_count += 1
    
    print(f"并发处理完成，共处理 {len(results)} 条数据")
    
    # 保存并发结果为jsonl格式
    print(f"正在保存并发结果到: {final_concurrent_results_path}")
    with open(final_concurrent_results_path, 'w', encoding='utf-8') as f:
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

for i, item in enumerate(data):
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
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(data)} 条数据 (过滤: {not_found_count + error_count + empty_boxed_count + no_answer_count})")
        continue
    
    # 提取boxed内容
    boxed_answers = extract_boxed_content(gpt_response)
    if not boxed_answers:
        empty_boxed_count += 1
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(data)} 条数据 (过滤: {not_found_count + error_count + empty_boxed_count + no_answer_count})")
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
        print(f"已处理 {i + 1}/{len(data)} 条数据 (过滤: {not_found_count + error_count + empty_boxed_count + no_answer_count})")

print(f"\n处理完成，共 {len(data)} 条数据")
print(f"过滤掉:")
print(f"  - {not_found_count} 条未找到结果")
print(f"  - {error_count} 条错误数据")
print(f"  - {empty_boxed_count} 条无法提取boxed答案")
print(f"  - {no_answer_count} 条No Answer数据")
print(f"保留: {len(filtered_data)} 条有效数据")

# 保存为jsonl文件
output_jsonl = os.path.join(model_path_dir, model_name, 'train_data.jsonl')
print(f"\n正在保存jsonl文件: {output_jsonl}")
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for item in filtered_data:
        # 将item转换为可序列化的格式
        serializable_item = convert_to_serializable(item)
        f.write(json.dumps(serializable_item, ensure_ascii=False) + '\n')
print(f"已保存jsonl文件")

# 保存为parquet文件
output_parquet = os.path.join(model_path_dir, model_name, 'train_data.parquet')
print(f"正在保存parquet文件: {output_parquet}")
pd.DataFrame(filtered_data).to_parquet(output_parquet, index=False)
print(f"已保存parquet文件")

print("\n全部完成！")
