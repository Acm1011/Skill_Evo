#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将单个模型多个版本在单个数据集上的评测结果上传到 TensorBoard
x 轴是训练 step (V1=step_interval, V2=2*step_interval, ...)
base model 的结果作为 step 0

目录结构：
    eval_results_dir/
        {prefix}-V1/
            {dataset}_Overall_results.jsonl
        {prefix}-V2/
            {dataset}_Overall_results.jsonl
        ...
    base_model_dir/
        {base_model}/
            {dataset}_Overall_results.jsonl
"""
from verl.utils.tracking import Tracking
import os
import re
import json
import pandas as pd
from collections import OrderedDict


# 数据集配置
DATASET_CONFIG = {
    "AIME24": {"n_samples": 32, "temperature": 0.6, "metric_key": "checked_mean@32"},
    "AIME25": {"n_samples": 32, "temperature": 0.6, "metric_key": "checked_mean@32"},
    "AMC23": {"n_samples": 1, "temperature": 0.0, "metric_key": "checked_mean@1"},
    "MATH500": {"n_samples": 1, "temperature": 0.0, "metric_key": "checked_mean@1"},
    "Minerva": {"n_samples": 1, "temperature": 0.0, "metric_key": "checked_mean@1"},
    "OlympiadBench": {"n_samples": 1, "temperature": 0.0, "metric_key": "checked_mean@1"},
}


def get_all_versions(eval_results_dir, prefix, base_model=None, base_model_dir=None, dataset=None):
    """
    获取所有版本目录
    
    Args:
        eval_results_dir: 评测结果目录
        prefix: 模型前缀
        base_model: base model 名称
        base_model_dir: base model 结果目录
        dataset: 数据集名称，用于验证结果文件是否存在
    
    Returns:
        versions: [(version_number, version_dir_path), ...]，按版本号排序
    """
    versions = []
    
    # 添加 base model 作为 version 0
    if base_model:
        if base_model_dir:
            base_path = os.path.join(base_model_dir, base_model)
        else:
            base_path = os.path.join(eval_results_dir, base_model)
        
        if os.path.exists(base_path):
            # 验证 base model 有该数据集的结果
            if dataset:
                result_file = os.path.join(base_path, f'{dataset}_Overall_results.jsonl')
                if os.path.exists(result_file):
                    versions.append((0, base_path))
                else:
                    print(f"Warning: Base model result not found: {result_file}")
            else:
                versions.append((0, base_path))
        else:
            print(f"Warning: Base model directory not found: {base_path}")
    
    # 扫描 {prefix}-V{数字} 目录
    pattern = re.compile(rf'^{re.escape(prefix)}-V(\d+)$')
    if os.path.exists(eval_results_dir):
        for name in os.listdir(eval_results_dir):
            path = os.path.join(eval_results_dir, name)
            if os.path.isdir(path):
                match = pattern.match(name)
                if match:
                    version_num = int(match.group(1))
                    # 验证该版本有该数据集的结果
                    if dataset:
                        result_file = os.path.join(path, f'{dataset}_Overall_results.jsonl')
                        if os.path.exists(result_file):
                            versions.append((version_num, path))
                        else:
                            print(f"Warning: Version {version_num} result not found: {result_file}")
                    else:
                        versions.append((version_num, path))
    
    # 按版本号排序
    versions.sort(key=lambda x: x[0])
    return versions


def read_single_dataset_eval(path, dataset, metric_key="checked_mean@32"):
    """
    读取单个数据集的评测结果
    
    Args:
        path: 结果目录路径
        dataset: 数据集名称
        metric_key: 指标 key，如 checked_mean@1 或 checked_mean@32
    
    Returns:
        res: {metric_name: value} 字典
        values: [value] 列表
    """
    res = {}
    values = []
    
    result_file = os.path.join(path, f'{dataset}_Overall_results.jsonl')
    
    if not os.path.exists(result_file):
        print(f"Warning: Result file not found: {result_file}")
        return res, values
    
    try:
        eval_data = pd.read_json(result_file, lines=True).to_dict(orient='records')
        for data_item in eval_data:
            data_source = data_item.get('data_source', 'unknown')
            
            # 尝试获取指定的 metric
            value = data_item.get(metric_key)
            if value is None:
                # 尝试其他可能的 key 格式
                n_samples = metric_key.split('@')[-1]
                alt_key = f'checked_mean@{n_samples}'
                value = data_item.get(alt_key, 0)
            
            if value is not None:
                # 确保是数值
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = 0.0
                
                res[f'{metric_key}/{data_source}'] = value
                values.append(value)
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
    
    return res, values


def tb(prefix, dataset, eval_results_dir=None, tb_path_dir=None, base_model=None, base_model_dir=None, step_interval=10):
    """
    将单个数据集的评测结果上传到 TensorBoard
    
    Args:
        prefix: 实验前缀名称
        dataset: 数据集名称
        eval_results_dir: 评测结果目录
        tb_path_dir: TensorBoard 日志目录
        base_model: base model 名称
        base_model_dir: base model 的评测结果目录
        step_interval: 每个版本对应的 step 间隔 (V1 = step_interval, V2 = 2*step_interval, ...)
    """
    if tb_path_dir is None:
        raise ValueError('tb_path_dir is None')
    
    if eval_results_dir is None:
        raise ValueError('eval_results_dir is None')
    
    os.makedirs(tb_path_dir, exist_ok=True)
    
    # 获取数据集配置
    config = DATASET_CONFIG.get(dataset, {"n_samples": 32, "temperature": 0.6, "metric_key": "checked_mean@32"})
    metric_key = config["metric_key"]
    temperature = config["temperature"]
    n_samples = config["n_samples"]
    
    print(f"Dataset: {dataset}")
    print(f"  n_samples: {n_samples}")
    print(f"  temperature: {temperature}")
    print(f"  metric_key: {metric_key}")
    
    # 获取所有版本
    versions = get_all_versions(eval_results_dir, prefix, base_model, base_model_dir, dataset)
    
    if not versions:
        print(f"Warning: No version directories found in {eval_results_dir}")
        return
    
    print(f"Found {len(versions)} versions: {[v[0] for v in versions]}")
    
    # 设置 TensorBoard
    tb_path = os.path.join(tb_path_dir, f'{prefix}-{dataset}')
    os.environ['TENSORBOARD_DIR'] = tb_path
    logger = Tracking(project_name='Se', experiment_name=f'{prefix}_{dataset}', default_backend=['console', 'tensorboard'])
    
    for version_num, version_dir in versions:
        print(f"Processing version {version_num}: {version_dir}")
        
        eval_data = {}
        
        # 读取数据集结果
        data, values = read_single_dataset_eval(version_dir, dataset, metric_key)
        
        # 直接记录数据集的值（单个数据集只有一个值）
        value = values[0] if values else 0.0
        eval_data[dataset] = round(value, 2)
        
        # 使用 step_interval 计算实际的 step 值
        actual_step = version_num * step_interval
        
        # 记录到 TensorBoard，x 轴为训练 step
        logger.log(data=eval_data, step=actual_step)
        print(f"  Logged step {actual_step} (V{version_num}): {dataset}={value:.2f}")
    
    print(f"\nTensorBoard logs saved to: {tb_path}")


def generate_results_table(prefix, dataset, eval_results_dir=None, base_model=None, base_model_dir=None, 
                           output_file=None, step_interval=10):
    """
    生成单个数据集的结果表格
    横轴是数据集，纵轴是版本/step
    """
    if eval_results_dir is None:
        raise ValueError('eval_results_dir is None')
    
    # 获取数据集配置
    config = DATASET_CONFIG.get(dataset, {"n_samples": 32, "temperature": 0.6, "metric_key": "checked_mean@32"})
    metric_key = config["metric_key"]
    temperature = config["temperature"]
    n_samples = config["n_samples"]
    
    # 获取所有版本
    versions = get_all_versions(eval_results_dir, prefix, base_model, base_model_dir, dataset)
    
    if not versions:
        print(f"Warning: No version directories found in {eval_results_dir}")
        return None
    
    # 构建表格数据
    table_data = []
    
    for version_num, version_dir in versions:
        row = OrderedDict()
        
        if version_num == 0:
            row['Model'] = base_model if base_model else 'Base'
        else:
            row['Model'] = f'{prefix}-V{version_num}'
        
        row['Step'] = version_num * step_interval
        
        # 读取结果
        data, values = read_single_dataset_eval(version_dir, dataset, metric_key)
        
        # 直接记录数据集的值（单个数据集只有一个值）
        value = values[0] if values else None
        row[dataset] = round(float(value), 2) if value is not None else None
        
        table_data.append(row)
    
    # 创建 DataFrame
    df = pd.DataFrame(table_data)
    
    # 设置列顺序
    column_order = ['Model', 'Step', dataset]
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # 按 Step 排序
    df = df.sort_values('Step').reset_index(drop=True)
    
    # 保存结果
    if output_file is None:
        output_file = os.path.join(eval_results_dir, f'{prefix}_{dataset}_results_table.csv')
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Results table saved to: {output_file}")
    
    # 保存 Markdown 格式
    md_file = output_file.replace('.csv', '.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Evaluation Results by Version\n\n")
        f.write(f"Prefix: {prefix}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"n_samples: {n_samples}, temperature: {temperature}\n")
        f.write(f"Metric: {metric_key}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Results table (Markdown) saved to: {md_file}")
    
    # 打印表格预览
    print(f"\n{dataset} Results Table:")
    print(df.to_string(index=False))
    
    return df


def generate_summary_json(prefix, dataset, eval_results_dir=None, base_model=None, base_model_dir=None, 
                          output_file=None, step_interval=10):
    """
    生成 JSON 格式的摘要结果
    """
    if eval_results_dir is None:
        raise ValueError('eval_results_dir is None')
    
    # 获取数据集配置
    config = DATASET_CONFIG.get(dataset, {"n_samples": 32, "temperature": 0.6, "metric_key": "checked_mean@32"})
    metric_key = config["metric_key"]
    temperature = config["temperature"]
    n_samples = config["n_samples"]
    
    # 获取所有版本
    versions = get_all_versions(eval_results_dir, prefix, base_model, base_model_dir, dataset)
    
    if not versions:
        print(f"Warning: No version directories found in {eval_results_dir}")
        return None
    
    summary = {
        "prefix": prefix,
        "dataset": dataset,
        "n_samples": n_samples,
        "temperature": temperature,
        "metric_key": metric_key,
        "step_interval": step_interval,
        "versions": []
    }
    
    for version_num, version_dir in versions:
        data, values = read_single_dataset_eval(version_dir, dataset, metric_key)
        value = values[0] if values else 0.0
        
        version_result = {
            "version": version_num,
            "step": version_num * step_interval,
            "model": base_model if version_num == 0 else f'{prefix}-V{version_num}',
            "value": round(value, 2)
        }
        summary["versions"].append(version_result)
    
    # 保存结果
    if output_file is None:
        output_file = os.path.join(eval_results_dir, f'{prefix}_{dataset}_summary.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON saved to: {output_file}")
    
    return summary


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Upload single dataset multi-version evaluation results to TensorBoard')
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix for model directories (e.g., data_AIME24_R_Zero_ref_q_gqttrl_icl_Qwen2.5-Math-7B)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (AIME24, AIME25, AMC23, MATH500, Minerva, OlympiadBench)"
    )
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        required=True,
        help="Directory containing {prefix}-V* subdirectories with evaluation results"
    )
    parser.add_argument(
        "--tb_path_dir",
        type=str,
        required=True,
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name for step 0 (e.g., Qwen2.5-Math-7B)"
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default='/home/ycy/data3/ttrl_saved/evaluation',
        help="Directory containing base model evaluation results (if different from eval_results_dir)"
    )
    parser.add_argument(
        "--step_interval",
        type=int,
        default=10,
        help="Step interval for each version (V1 = step_interval, V2 = 2*step_interval, etc.). Default: 10."
    )
    parser.add_argument(
        "--generate_table",
        action="store_true",
        default=True,
        help="Generate results table after uploading to TensorBoard"
    )
    parser.add_argument(
        "--generate_summary",
        action="store_true",
        default=True,
        help="Generate summary JSON file"
    )
    parser.add_argument(
        "--table_output",
        type=str,
        default=None,
        help="Output file path for results table (optional)"
    )
    
    args = parser.parse_args()
    
    # 验证数据集
    if args.dataset not in DATASET_CONFIG:
        print(f"Warning: Unknown dataset '{args.dataset}', using default config")
    
    # 上传到 TensorBoard
    tb(args.prefix, args.dataset, args.eval_results_dir, args.tb_path_dir, 
       args.base_model, args.base_model_dir, args.step_interval)
    
    # 生成结果表格
    if args.generate_table:
        generate_results_table(args.prefix, args.dataset, args.eval_results_dir, 
                               args.base_model, args.base_model_dir, args.table_output, args.step_interval)
    
    # 生成摘要 JSON
    if args.generate_summary:
        generate_summary_json(args.prefix, args.dataset, args.eval_results_dir, 
                              args.base_model, args.base_model_dir, None, args.step_interval)
