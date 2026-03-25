#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理weakness_data_pool.json文件
将数据重新组织为：每个元素包含3个example，每个example包含problem, difficulty, topic
要求：每个元素的三个example的topic必须一致，difficulty尽可能接近
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Any
import numpy as np


def load_data(file_path: str) -> Dict[str, Any]:
    """加载JSON数据文件"""
    print(f"正在加载数据文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载完成，共 {len(data)} 条数据")
    return data


def group_by_topic(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """按topic分组数据"""
    topic_groups = defaultdict(list)
    
    for key, item in data.items():
        if 'extra_info' in item and 'topic' in item['extra_info']:
            topic = item['extra_info']['topic']
            example = {
                'problem': item['extra_info'].get('problem', ''),
                'difficulty': item['extra_info'].get('difficulty', 0.0),
                'topic': topic,
                'idx': item['extra_info'].get('idx', key)
            }
            topic_groups[topic].append(example)
    
    print(f"按topic分组完成，共 {len(topic_groups)} 个不同的topic")
    return topic_groups


def find_closest_triplets(examples: List[Dict[str, Any]], num_examples: int = 3) -> List[List[Dict[str, Any]]]:
    """
    找到difficulty最接近的3个example组合
    使用贪心算法：优先选择difficulty方差最小的组合，且不重叠
    """
    if len(examples) < num_examples:
        return []
    
    # 按difficulty排序
    sorted_examples = sorted(examples, key=lambda x: x['difficulty'])
    
    triplets = []
    # 使用滑动窗口找到difficulty最接近的组合
    for i in range(len(sorted_examples) - num_examples + 1):
        triplet = sorted_examples[i:i + num_examples]
        # 计算difficulty的方差和标准差
        difficulties = [ex['difficulty'] for ex in triplet]
        variance = np.var(difficulties)
        std = np.std(difficulties)
        # 同时考虑方差和difficulty范围
        diff_range = max(difficulties) - min(difficulties)
        # 综合评分：方差越小越好，范围越小越好
        score = variance + diff_range * 0.1
        triplets.append((score, triplet))
    
    # 按评分排序，返回评分最小的组合（difficulty最接近）
    triplets.sort(key=lambda x: x[0])
    
    result = []
    used_indices = set()
    
    # 贪心选择不重叠的组合
    for score, triplet in triplets:
        triplet_indices = tuple(sorted(ex['idx'] for ex in triplet))
        # 检查是否有重叠
        if not any(idx in used_indices for idx in [ex['idx'] for ex in triplet]):
            result.append(triplet)
            used_indices.update([ex['idx'] for ex in triplet])
    
    return result


def process_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """处理数据，生成符合要求的格式"""
    # 按topic分组
    topic_groups = group_by_topic(data)
    
    processed_data = []
    
    # 统计信息
    skipped_topics = 0
    total_triplets = 0
    
    # 处理每个topic
    for topic, examples in topic_groups.items():
        if len(examples) < 3:
            skipped_topics += 1
            continue
        
        # 找到difficulty最接近的3个example组合
        triplets = find_closest_triplets(examples, num_examples=3)
        
        for triplet in triplets:
            # 计算difficulty最大值和topic（三个example的topic应该一致）
            difficulties = [ex['difficulty'] for ex in triplet]
            difficulty_max = max(difficulties)
            # topic应该一致，取第一个即可
            topic = triplet[0]['topic']
            
            # 构建新的数据格式
            processed_item = {
                'examples': [
                    {
                        'problem': ex['problem'],
                        'difficulty': ex['difficulty'],
                        'topic': ex['topic']
                    }
                    for ex in triplet
                ],
                'difficulty_max': difficulty_max,
                'topic': topic
            }
            processed_data.append(processed_item)
            total_triplets += 1
    
    print(f"\n处理完成:")
    print(f"  - 跳过的topic（example数量<3）: {skipped_topics}")
    print(f"  - 生成的元素数量: {total_triplets}")
    
    return processed_data


def save_processed_data(data: List[Dict[str, Any]], output_path: str):
    """保存处理后的数据"""
    print(f"\n正在保存处理后的数据到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"保存完成")


def main():
    # 输入文件路径
    input_file = '/root/users/ycy/Self-evolving-Agent/se_code/prompt_weakness/weakness_data_pool.json'
    
    # 输出文件路径
    output_file = '/root/users/ycy/Self-evolving-Agent/se_code/prompt_weakness/weakness_data_pool_processed.json'
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 加载数据
    data = load_data(input_file)
    
    # 处理数据
    processed_data = process_data(data)
    
    # 保存处理后的数据
    save_processed_data(processed_data, output_file)
    
    # 显示一些示例
    if processed_data:
        print(f"\n示例数据（前3个元素）:")
        for i, item in enumerate(processed_data[:3]):
            print(f"\n元素 {i+1}:")
            print(f"  Topic: {item['topic']}")
            print(f"  Difficulty Max: {item['difficulty_max']}")
            for j, ex in enumerate(item['examples']):
                print(f"  Example {j+1}:")
                print(f"    Difficulty: {ex['difficulty']}")
                print(f"    Problem: {ex['problem'][:100]}..." if len(ex['problem']) > 100 else f"    Problem: {ex['problem']}")


if __name__ == '__main__':
    main()

