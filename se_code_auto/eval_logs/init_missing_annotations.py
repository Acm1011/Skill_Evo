#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理 all_topic_annotations.json 文件
对于每个 topic 中，difficulty_question_counts 为 0 且在 annotations 中没有对应信息的，
初始化1个最近 difficulty level 的数量和信息
"""

import json
import os


def find_nearest_difficulty(target_difficulty, available_difficulties):
    """
    找到数值上最接近目标 difficulty 的 available difficulty
    
    Args:
        target_difficulty: 目标 difficulty (字符串，如 "1.5")
        available_difficulties: 可用的 difficulty 列表 (字符串列表)
    
    Returns:
        最接近的 difficulty (字符串)，如果没有可用的则返回 None
    """
    if not available_difficulties:
        return None
    
    target_float = float(target_difficulty)
    
    # 计算所有可用 difficulty 与目标 difficulty 的距离
    distances = []
    for diff in available_difficulties:
        diff_float = float(diff)
        distance = abs(diff_float - target_float)
        distances.append((distance, diff))
    
    # 按距离排序，返回距离最小的
    distances.sort(key=lambda x: x[0])
    return distances[0][1]


def process_annotations_file(input_file, output_file=None):
    """
    处理 JSON 文件，为缺失的 difficulty 初始化 annotation
    """
    if output_file is None:
        output_file = input_file
    
    # 读取 JSON 文件
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计信息
    total_topics = 0
    total_initialized = 0
    
    # 遍历所有 topic
    for topic, topic_data in data.items():
        total_topics += 1
        difficulty_counts = topic_data.get('difficulty_question_counts', {})
        annotations = topic_data.get('annotations', {})
        
        # 获取所有已有 annotation 的 difficulty 列表
        available_difficulties = list(annotations.keys())
        
        if not available_difficulties:
            print(f"警告: Topic '{topic}' 没有任何 annotations，跳过")
            continue
        
        # 遍历所有 difficulty_question_counts
        initialized_count = 0
        for difficulty, count in difficulty_counts.items():
            # 如果 count 为 0 且 annotations 中没有对应的信息
            if count == 0 and difficulty not in annotations:
                # 找到最接近的 difficulty
                nearest_difficulty = find_nearest_difficulty(difficulty, available_difficulties)
                
                if nearest_difficulty:
                    # 复制最接近 difficulty 的 annotation 信息
                    nearest_annotation = annotations[nearest_difficulty].copy()
                    annotations[difficulty] = nearest_annotation
                    
                    # 设置 count 为最近 difficulty 的 count
                    nearest_count = difficulty_counts.get(nearest_difficulty, 0)
                    difficulty_counts[difficulty] = max(nearest_count,1)
                    
                    initialized_count += 1
                    print(f"已初始化: {topic} - difficulty {difficulty} (从 {nearest_difficulty} 复制, count={difficulty_counts[difficulty]})")
                else:
                    print(f"警告: Topic '{topic}', difficulty {difficulty} 无法找到可用的 annotation 源")
        
        total_initialized += initialized_count
        if initialized_count > 0:
            print(f"  Topic '{topic}' 共初始化了 {initialized_count} 个 difficulty")
        
        # 重新计算并更新 total_question_count
        total_count = sum(difficulty_counts.values())
        topic_data['total_question_count'] = total_count
        if initialized_count > 0:
            print(f"  Topic '{topic}' 的 total_question_count 已更新为: {total_count}")
    
    # 保存回文件
    print(f"\n正在保存到文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"处理完成！")
    print(f"总 topic 数: {total_topics}")
    print(f"总初始化 difficulty 数: {total_initialized}")
    print(f"输出文件: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    input_file = '/root/users/ycy/Self-evolving-Agent/se_code/prompt2/all_topic_annotations.json'
    
    # 可以选择备份原文件
    backup_file = input_file + '.backup_init'
    if not os.path.exists(backup_file):
        print(f"创建备份文件: {backup_file}")
        import shutil
        shutil.copy2(input_file, backup_file)
    
    # 处理文件
    process_annotations_file(input_file)

