#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 all_topic_annotations.json 中的 desc 字段的三个子字段合并为一个字符串
"""

import json
import os


def format_list_to_string(value):
    """
    将列表或字符串格式化为字符串
    如果是列表，用逗号和空格连接
    如果是字符串，直接返回
    """
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    elif isinstance(value, str):
        return value
    else:
        return str(value)


def analyze_all_keys(all_keys_set):
    """
    分析所有出现的 key，确定它们应该对应模板中的哪个位置
    返回: (setting_keys, challenge_keys, solution_keys) - 每个都是可能的 key 名称列表
    """
    setting_keywords = ['scope', 'setting', 'boundary', 'domain', 'range']
    challenge_keywords = ['concept', 'challenge', 'core', 'idea', 'principle', 'key']
    solution_keywords = ['tool', 'theorem', 'solution', 'path', 'method', 'technique']
    
    setting_keys = []
    challenge_keys = []
    solution_keys = []
    
    for key in all_keys_set:
        key_lower = key.lower()
        
        # 检查是否匹配 Setting
        if any(word in key_lower for word in setting_keywords):
            setting_keys.append(key)
        # 检查是否匹配 Core Challenge
        elif any(word in key_lower for word in challenge_keywords):
            challenge_keys.append(key)
        # 检查是否匹配 Solution Path
        elif any(word in key_lower for word in solution_keywords):
            solution_keys.append(key)
    
    return setting_keys, challenge_keys, solution_keys


def identify_key_positions(keys, key_mapping):
    """
    根据全局 key 映射关系，识别当前 keys 应该对应模板中的哪个位置
    key_mapping: dict, 格式为 {key_name: position}，position 为 'setting', 'challenge', 'solution'
    返回: (setting_key, challenge_key, solution_key)
    """
    # 根据映射关系确定每个 key 的位置
    key_positions = {}
    for key in keys:
        if key in key_mapping:
            key_positions[key] = key_mapping[key]
    
    # 如果映射关系不完整，使用关键词匹配作为补充
    keys_lower = [k.lower() for k in keys]
    
    # 第一个位置：Setting
    setting_key = None
    for key, pos in key_positions.items():
        if pos == 'setting':
            setting_key = key
            break
    if not setting_key:
        for i, key_lower in enumerate(keys_lower):
            if any(word in key_lower for word in ['scope', 'setting', 'boundary', 'domain', 'range']):
                setting_key = keys[i]
                break
    
    # 第二个位置：Core Challenge
    challenge_key = None
    for key, pos in key_positions.items():
        if pos == 'challenge':
            challenge_key = key
            break
    if not challenge_key:
        for i, key_lower in enumerate(keys_lower):
            if any(word in key_lower for word in ['concept', 'challenge', 'core', 'idea', 'principle', 'key']):
                challenge_key = keys[i]
                break
    
    # 第三个位置：Solution Path
    solution_key = None
    for key, pos in key_positions.items():
        if pos == 'solution':
            solution_key = key
            break
    if not solution_key:
        for i, key_lower in enumerate(keys_lower):
            if any(word in key_lower for word in ['tool', 'theorem', 'solution', 'path', 'method', 'technique']):
                solution_key = keys[i]
                break
    
    # 如果仍然无法匹配，按照key的字母顺序分配
    if not setting_key:
        setting_key = sorted(keys)[0]
    if not challenge_key:
        remaining = [k for k in sorted(keys) if k != setting_key]
        challenge_key = remaining[0] if remaining else keys[0]
    if not solution_key:
        remaining = [k for k in sorted(keys) if k != setting_key and k != challenge_key]
        solution_key = remaining[0] if remaining else keys[-1]
    
    return setting_key, challenge_key, solution_key


def merge_desc_fields(desc, key_mapping):
    """
    将 desc 的三个字段合并为指定格式的字符串
    desc 是一个字典，包含三个key（key名称不固定）
    key_mapping: dict, 全局的 key 到位置的映射关系
    """
    keys = list(desc.keys())
    
    if len(keys) != 3:
        raise ValueError(f"desc 字典应该包含3个key，但找到了 {len(keys)} 个: {keys}")
    
    # 识别三个key对应的位置
    setting_key, challenge_key, solution_key = identify_key_positions(keys, key_mapping)
    
    # 格式化值
    setting_value = format_list_to_string(desc[setting_key])
    challenge_value = format_list_to_string(desc[challenge_key])
    solution_value = format_list_to_string(desc[solution_key])
    
    merged = f"A problem blueprint for this topic: The problem's Setting is defined by [{setting_value}];its Core Challenge must test [{challenge_value}]; and its Intended Solution Path must be constructed by applying [{solution_value}]."
    #"A problem blueprint for this topic: The problem's Setting is defined by [Scope]; its Core Challenge must test [Key Concepts]; and its Intended Solution Path must be constructed by applying [Tools/Theorems]."
    
    return merged


def process_annotations_file(input_file, output_file=None):
    """
    处理 JSON 文件，合并所有 desc 字段
    """
    if output_file is None:
        output_file = input_file
    
    # 读取 JSON 文件
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 第一步：收集所有 desc 字典中的所有 key
    print("\n第一步：扫描所有 desc 字典，收集所有 key...")
    all_keys_set = set()
    for topic, topic_data in data.items():
        annotations = topic_data.get('annotations', {})
        for difficulty, annotation in annotations.items():
            if annotation and 'desc' in annotation:
                desc = annotation['desc']
                if isinstance(desc, dict):
                    all_keys_set.update(desc.keys())
    
    print(f"找到的所有 key: {sorted(all_keys_set)}")
    
    # 第二步：分析所有 key，确定它们的映射关系
    print("\n第二步：分析 key 的映射关系...")
    setting_keys, challenge_keys, solution_keys = analyze_all_keys(all_keys_set)
    
    # 构建 key 到位置的映射字典
    key_mapping = {}
    for key in setting_keys:
        key_mapping[key] = 'setting'
    for key in challenge_keys:
        key_mapping[key] = 'challenge'
    for key in solution_keys:
        key_mapping[key] = 'solution'
    
    print(f"Setting keys: {setting_keys}")
    print(f"Core Challenge keys: {challenge_keys}")
    print(f"Solution Path keys: {solution_keys}")
    print(f"Key mapping: {key_mapping}")
    
    # 第三步：处理所有 desc 字段
    print("\n第三步：处理所有 desc 字段...")
    total_topics = 0
    total_annotations = 0
    processed_annotations = 0
    
    # 遍历所有 topic
    for topic, topic_data in data.items():
        total_topics += 1
        annotations = topic_data.get('annotations', {})
        
        # 遍历该 topic 下的所有 difficulty
        for difficulty, annotation in annotations.items():
            total_annotations += 1
            
            if annotation and 'desc' in annotation:
                desc = annotation['desc']
                
                # 检查 desc 是否是字典格式（包含三个字段）
                if isinstance(desc, dict):
                    try:
                        # 合并字段
                        merged_desc = merge_desc_fields(desc, key_mapping)
                        # 替换 desc 字段（直接是字符串）
                        annotation['desc'] = merged_desc
                        processed_annotations += 1
                        print(f"已处理: {topic} - {difficulty}")
                    except ValueError as e:
                        print(f"错误: {topic} - {difficulty}: {e}")
                elif isinstance(desc, str):
                    # 如果已经是字符串，跳过
                    print(f"跳过（已是字符串）: {topic} - {difficulty}")
                else:
                    print(f"警告: {topic} - {difficulty}: desc 类型不是字典或字符串: {type(desc)}")
    
    # 保存回文件
    print(f"\n正在保存到文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"处理完成！")
    print(f"总 topic 数: {total_topics}")
    print(f"总 annotation 数: {total_annotations}")
    print(f"已处理 annotation 数: {processed_annotations}")
    print(f"输出文件: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    input_file = '/root/users/ycy/Self-evolving-Agent/se_code/prompt2/all_topic_annotations.json'
    
    # 可以选择备份原文件
    backup_file = input_file + '.backup'
    if not os.path.exists(backup_file):
        print(f"创建备份文件: {backup_file}")
        import shutil
        shutil.copy2(input_file, backup_file)
    
    # 处理文件
    process_annotations_file(input_file)

