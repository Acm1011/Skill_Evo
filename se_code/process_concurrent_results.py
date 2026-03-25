import pandas as pd
import os
import json
import numpy as np


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


def main():
    # 配置路径
    model_path_dir = '/root/users/ycy/saved_results/Solver'
    model_name = 'entropy_prompt2_se-Zero_Qwen3-4B-Base-V2'
    
    # 读取原始数据
    data_path = os.path.join(model_path_dir, model_name, 'train_data.parquet')
    print(f"正在读取原始数据: {data_path}")
    data = pd.read_parquet(data_path).to_dict(orient='records')
    print(f"成功读取 {len(data)} 条原始数据")
    
    # 读取并发结果
    concurrent_results_path = os.path.join(model_path_dir, model_name, 'concurrent_results.jsonl')
    print(f"正在读取并发结果: {concurrent_results_path}")
    
    question_to_result = {}
    with open(concurrent_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            question = result['question']
            if question:  # 跳过空question
                question_to_result[question] = {
                    'gpt_response': result['gpt_response'],
                    'error': result['error']
                }
    
    print(f"成功读取 {len(question_to_result)} 条并发结果")
    
    # 匹配并处理数据
    print("\n开始处理数据...")
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
        
        result = question_to_result[question]
        gpt_response = result['gpt_response']
        error = result['error']
        
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
            print(f"已处理 {i + 1}/{len(data)} 条数据 (过滤: {not_found_count + error_count + no_answer_count + empty_boxed_count})")
    
    # 统计信息
    print(f"\n处理完成，共 {len(data)} 条数据")
    print(f"过滤掉:")
    print(f"  - {not_found_count} 条未找到结果")
    print(f"  - {error_count} 条错误数据")
    print(f"  - {empty_boxed_count} 条无法提取boxed答案")
    print(f"  - {no_answer_count} 条No Answer数据")
    print(f"保留: {len(filtered_data)} 条有效数据")
    
    # 保存为jsonl文件
    output_jsonl = os.path.join(model_path_dir, model_name, 'train_data_filtered.jsonl')
    print(f"\n正在保存jsonl文件: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            # 将item转换为可序列化的格式
            serializable_item = convert_to_serializable(item)
            f.write(json.dumps(serializable_item, ensure_ascii=False) + '\n')
    print(f"已保存jsonl文件")
    
    # 保存为parquet文件
    output_parquet = os.path.join(model_path_dir, model_name, 'train_data_filtered.parquet')
    print(f"正在保存parquet文件: {output_parquet}")
    pd.DataFrame(filtered_data).to_parquet(output_parquet, index=False)
    print(f"已保存parquet文件")
    
    print("\n全部完成！")


if __name__ == '__main__':
    main()

