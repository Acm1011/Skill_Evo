#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将按 step 组织的评测结果上传到 TensorBoard
x 轴是训练 step（而非迭代次数）
base model 的结果作为 step 0
"""
from verl.utils.tracking import Tracking
import os
import re
import json
import pandas as pd
from collections import OrderedDict


def get_all_steps(save_path_dir, base_model_dir=None):
    """
    获取所有 step 目录
    
    Returns:
        steps: [(step_number, step_dir_path), ...]，按 step 排序
    """
    steps = []
    
    # 添加 base model 作为 step 0
    if base_model_dir and os.path.exists(base_model_dir):
        steps.append((0, base_model_dir))
    
    # 扫描 step_* 目录
    if os.path.exists(save_path_dir):
        for name in os.listdir(save_path_dir):
            path = os.path.join(save_path_dir, name)
            if os.path.isdir(path):
                match = re.match(r'^step_(\d+)$', name)
                if match:
                    step_num = int(match.group(1))
                    steps.append((step_num, path))
    
    # 按 step 排序
    steps.sort(key=lambda x: x[0])
    return steps


def read_math_eval(path, greedy=False):
    """读取数学数据集的评测结果"""
    res = {}
    all_data = []
    
    if not os.path.exists(path):
        return res, all_data
    
    try:
        eval_data = pd.read_json(path, lines=True).to_dict(orient='records')
        for data_item in eval_data:
            data_source = data_item.get('data_source', 'unknown')
            if greedy:
                data = data_item.get('checked_mean@1', 0)
                res[f'checked_mean@1/{data_source}'] = data
            else:
                data = data_item.get('checked_mean@32', 0)
                res[f'checked_mean@32/{data_source}'] = data
            all_data.append(float(data) if data else 0)
    except Exception as e:
        print(f"Error reading {path}: {e}")
    
    return res, all_data


def read_aggregated_eval(eval_path):
    """读取额外评测数据集的聚合评测结果"""
    res = {}
    general_avg_data = []
    aggregated_path = os.path.join(eval_path, 'aggregated_eval_results.json')
    
    if not os.path.exists(aggregated_path):
        return res, general_avg_data
    
    try:
        with open(aggregated_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            datasets = data.get('datasets')
            if datasets is None:
                datasets = data.get('additional_datasets', {})
            
            for dataset_name in ['bbeh', 'mmlupro', 'supergpqa', 'gpqa']:
                if dataset_name in datasets:
                    dataset_result = datasets[dataset_name]
                    accuracy = dataset_result.get('accuracy')
                    
                    if accuracy is not None:
                        # 确保是百分比格式
                        if accuracy <= 1.0:
                            accuracy = accuracy * 100
                        res[f'additional_eval/{dataset_name}_accuracy'] = round(accuracy, 2)
                        general_avg_data.append(accuracy)
    except Exception as e:
        print(f"Error reading aggregated eval results: {e}")
    
    return res, general_avg_data


def tb_step(exp_name, temperature, save_path_dir, base_model_dir=None, tb_path_dir=None):
    """
    将按 step 组织的评测结果上传到 TensorBoard
    
    Args:
        exp_name: 实验名称
        temperature: 采样温度
        save_path_dir: 评测结果目录
        base_model_dir: base model 的评测结果目录
        tb_path_dir: TensorBoard 日志目录
    """
    if tb_path_dir is None:
        raise ValueError('tb_path_dir is None')
    
    os.makedirs(tb_path_dir, exist_ok=True)
    
    # 获取所有 steps
    steps = get_all_steps(save_path_dir, base_model_dir)
    
    if not steps:
        print(f"Warning: No step directories found in {save_path_dir}")
        return
    
    print(f"Found {len(steps)} steps: {[s[0] for s in steps]}")
    
    # 设置 TensorBoard
    tb_path = os.path.join(tb_path_dir, f'{exp_name}-temperature_{temperature}')
    os.environ['TENSORBOARD_DIR'] = tb_path
    logger = Tracking(project_name='Se', experiment_name=exp_name, default_backend=['console', 'tensorboard'])
    
    for step_num, step_dir in steps:
        print(f"Processing step {step_num}: {step_dir}")
        
        eval_data = {}
        
        # 读取数学数据集结果
        greedy_data_path = os.path.join(step_dir, 'greedy_data_Overall_results.jsonl')
        temp_data_path = os.path.join(step_dir, 'temp_data_Overall_results.jsonl')
        
        d1, avg_d1 = read_math_eval(greedy_data_path, greedy=True)
        d2, avg_d2 = read_math_eval(temp_data_path, greedy=False)
        
        # 计算 Math AVG
        math_avg_data = avg_d1 + avg_d2
        math_avg = sum(math_avg_data) / len(math_avg_data) if len(math_avg_data) > 0 else 0.0
        
        eval_data.update(d1)
        eval_data.update(d2)
        eval_data['Math_AVG'] = round(math_avg, 2)
        
        # 读取通用数据集的评测结果
        additional_eval_data, general_avg_data = read_aggregated_eval(step_dir)
        eval_data.update(additional_eval_data)
        
        # 计算 General AVG
        general_avg = sum(general_avg_data) / len(general_avg_data) if len(general_avg_data) > 0 else 0.0
        eval_data['General_AVG'] = round(general_avg, 2)
        
        # 计算 Overall AVG
        all_avg_data = math_avg_data + general_avg_data
        overall_avg = sum(all_avg_data) / len(all_avg_data) if len(all_avg_data) > 0 else 0.0
        eval_data['Overall_AVG'] = round(overall_avg, 2)
        
        # 记录到 TensorBoard，x 轴为训练 step
        logger.log(data=eval_data, step=step_num)
        print(f"  Logged step {step_num}: Math_AVG={math_avg:.2f}, General_AVG={general_avg:.2f}, Overall_AVG={overall_avg:.2f}")
    
    print(f"\nTensorBoard logs saved to: {tb_path}")


def generate_results_table_step(exp_name, temperature, save_path_dir, base_model_dir=None, output_file=None):
    """
    生成结果表格
    横轴是数据集和 avg，纵轴是 step
    """
    # 获取所有 steps
    steps = get_all_steps(save_path_dir, base_model_dir)
    
    if not steps:
        print(f"Warning: No step directories found in {save_path_dir}")
        return None
    
    # 收集所有数学数据集名称
    math_datasets_set = set()
    all_general_datasets = ['bbeh', 'mmlupro', 'supergpqa', 'gpqa']
    
    # 先遍历一次收集数学数据集名称
    for step_num, step_dir in steps:
        greedy_data_path = os.path.join(step_dir, 'greedy_data_Overall_results.jsonl')
        temp_data_path = os.path.join(step_dir, 'temp_data_Overall_results.jsonl')
        
        if os.path.exists(greedy_data_path):
            try:
                eval_data = pd.read_json(greedy_data_path, lines=True).to_dict(orient='records')
                for data_item in eval_data:
                    data_source = data_item.get('data_source')
                    if data_source:
                        math_datasets_set.add((1, data_source))
            except:
                pass
        
        if os.path.exists(temp_data_path):
            try:
                eval_data = pd.read_json(temp_data_path, lines=True).to_dict(orient='records')
                for data_item in eval_data:
                    data_source = data_item.get('data_source')
                    if data_source:
                        math_datasets_set.add((2, data_source))
            except:
                pass
    
    # 排序数学数据集
    sorted_math_datasets = sorted(math_datasets_set, key=lambda x: (x[1], x[0]))
    all_math_datasets = []
    for dataset_type, data_source in sorted_math_datasets:
        if dataset_type == 1:
            all_math_datasets.append(f'checked_mean@1/{data_source}')
        else:
            all_math_datasets.append(f'checked_mean@32/{data_source}')
    
    # 构建表格数据
    table_data = []
    
    for step_num, step_dir in steps:
        row = OrderedDict()
        row['Step'] = step_num
        
        # 读取数学数据集结果
        greedy_data_path = os.path.join(step_dir, 'greedy_data_Overall_results.jsonl')
        temp_data_path = os.path.join(step_dir, 'temp_data_Overall_results.jsonl')
        
        math_values = []
        d1, avg_d1 = read_math_eval(greedy_data_path, greedy=True)
        d2, avg_d2 = read_math_eval(temp_data_path, greedy=False)
        
        # 添加数学数据集结果
        for key in all_math_datasets:
            dataset_name = key.split('/')[-1]
            if key.startswith('checked_mean@1/'):
                value = d1.get(key)
                row[dataset_name] = round(float(value), 2) if value is not None else None
                if value is not None:
                    math_values.append(float(value))
            elif key.startswith('checked_mean@32/'):
                value = d2.get(key)
                row[dataset_name] = round(float(value), 2) if value is not None else None
                if value is not None:
                    math_values.append(float(value))
        
        # Math AVG
        math_avg = sum(math_values) / len(math_values) if len(math_values) > 0 else None
        row['Math_AVG'] = round(math_avg, 2) if math_avg is not None else None
        
        # General 数据集结果
        additional_eval_data, general_avg_data = read_aggregated_eval(step_dir)
        
        for dataset_name in all_general_datasets:
            key = f'additional_eval/{dataset_name}_accuracy'
            value = additional_eval_data.get(key)
            row[dataset_name] = round(value, 2) if value is not None else None
        
        # General AVG
        general_avg = sum(general_avg_data) / len(general_avg_data) if len(general_avg_data) > 0 else None
        row['General_AVG'] = round(general_avg, 2) if general_avg is not None else None
        
        # Overall AVG
        all_values = math_values + general_avg_data
        overall_avg = sum(all_values) / len(all_values) if len(all_values) > 0 else None
        row['Overall_AVG'] = round(overall_avg, 2) if overall_avg is not None else None
        
        table_data.append(row)
    
    # 创建 DataFrame
    df = pd.DataFrame(table_data)
    
    # 设置列顺序
    column_order = ['Step']
    math_dataset_names = [key.split('/')[-1] for key in all_math_datasets]
    column_order.extend(math_dataset_names)
    column_order.append('Math_AVG')
    column_order.extend(all_general_datasets)
    column_order.append('General_AVG')
    column_order.append('Overall_AVG')
    
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # 按 step 排序
    df = df.sort_values('Step').reset_index(drop=True)
    
    # 保存结果
    if output_file is None:
        output_file = os.path.join(save_path_dir, f'{exp_name}_results_table.csv')
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Results table saved to: {output_file}")
    
    # 保存 Markdown 格式
    md_file = output_file.replace('.csv', '.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Evaluation Results by Step\n\n")
        f.write(f"Experiment: {exp_name}, Temperature: {temperature}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Results table (Markdown) saved to: {md_file}")
    
    return df


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Upload step-based evaluation results to TensorBoard')
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--save_path_dir",
        type=str,
        required=True,
        help="Directory containing step_* subdirectories with evaluation results"
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default=None,
        help="Directory containing base model evaluation results (as step 0)"
    )
    parser.add_argument(
        "--tb_path_dir",
        type=str,
        required=True,
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--generate_table",
        action="store_true",
        default=True,
        help="Generate results table after uploading to TensorBoard"
    )
    parser.add_argument(
        "--table_output",
        type=str,
        default=None,
        help="Output file path for results table (optional)"
    )
    
    args = parser.parse_args()
    
    # 上传到 TensorBoard
    tb_step(args.exp_name, args.temperature, args.save_path_dir, args.base_model_dir, args.tb_path_dir)
    
    # 生成结果表格
    if args.generate_table:
        generate_results_table_step(args.exp_name, args.temperature, args.save_path_dir, 
                                    args.base_model_dir, args.table_output)
