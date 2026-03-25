#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 get_prompts_weakness 函数
"""

import sys
import os
import json

# 添加父目录到路径，以便导入 Challenger_dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Challenger_dataset import get_prompts_weakness

def main():
    # 测试生成5个prompt
    num_querys = 5
    print(f"正在生成 {num_querys} 个prompt...")
    
    try:
        result = get_prompts_weakness(num_querys)
        print(f"成功生成 {len(result)} 个prompt")
        
        # 保存结果
        output_file = '/root/users/ycy/Self-evolving-Agent/se_code/prompt_weakness/test_prompts_output.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
        
        # 显示第一个prompt的详细信息
        if result:
            print("\n第一个prompt的详细信息:")
            first = result[0]
            print(f"  idx: {first['idx']}")
            print(f"  data_source: {first['data_source']}")
            print(f"  topic: {first['topic']}")
            print(f"  target_level: {first['target_level']}")
            print(f"  ability: {first['ability']}")
            print(f"  prompt长度: {len(first['prompt'])}")
            print(f"\n  System content (前200字符):")
            print(f"    {first['prompt'][0]['content'][:200]}...")
            print(f"\n  User content (前300字符):")
            print(f"    {first['prompt'][1]['content'][:300]}...")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
