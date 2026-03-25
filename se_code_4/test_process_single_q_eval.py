#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 process_single_q_eval 函数的各种情况
"""

import json
import re


def process_single_q_eval(idx, question, response):
    """
    从模型响应中提取 JSON 分数。
    根据新的 prompt (q_eval_system.txt 和 q_eval_user.txt)，模型应该只输出 JSON 格式：{"score": <integer from 0 to 10>}
    """
    prompt = response.prompt
    assert question in prompt, f"question {question=} not in prompt {prompt=}"
    
    scores = []
    responses_str = []
    
    for out in response.outputs:
        text = out.text.strip()
        responses_str.append(text)
        
        # 尝试从文本中提取 JSON 分数
        score = None
        
        # 方法1: 直接解析整个响应文本为 JSON（根据新 prompt，整个响应应该就是 JSON）
        try:
            obj = json.loads(text)
            if 'score' in obj and isinstance(obj['score'], (int, float)):
                score = float(obj['score'])
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # 方法2: 如果直接解析失败，尝试清理文本后解析（移除可能的尾随空白、标点等）
        if score is None:
            # 移除可能的尾随标点符号和空白
            cleaned_text = text.rstrip('.,;!?\n\r\t ')
            try:
                obj = json.loads(cleaned_text)
                if 'score' in obj and isinstance(obj['score'], (int, float)):
                    score = float(obj['score'])
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # 方法3: 如果还是失败，尝试使用正则表达式提取（作为 fallback）
        if score is None:
            # 查找 JSON 格式的 score 字段，支持整数和浮点数
            # 使用 findall 找到所有匹配，取最后一个（通常最后一个可能是最终答案）
            json_matches = re.findall(r'\{[\s]*"score"[\s]*:[\s]*(\d+(?:\.\d+)?)[\s]*\}', text)
            if json_matches:
                try:
                    # 取最后一个匹配的分数
                    score = float(json_matches[-1])
                except (ValueError, IndexError):
                    pass
        
        # 验证分数范围 (0-10)
        if score is not None:
            score = max(0.0, min(10.0, score))
        else:
            # 如果无法提取分数，设置为 0.0（表示解析失败）
            score = 0.0
        
        scores.append(score)
    
    # 计算统计信息
    
    avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0.0
    reward = avg_score / 10.0
    return {
        'idx': idx,
        'question': question,
        'reward_info': {
            'scores': scores,
            'responses_str': responses_str,
            'avg_score': avg_score,
        },
        'reward': reward,  # 将分数归一化到 0-1 范围
    }


class MockOutput:
    """模拟 vLLM 的 output 对象"""
    def __init__(self, text):
        self.text = text


class MockResponse:
    """模拟 vLLM 的 response 对象"""
    def __init__(self, prompt, outputs_text):
        self.prompt = prompt
        self.outputs = [MockOutput(text) for text in outputs_text]


def test_case(name, question, prompt, outputs_text, expected_scores, expected_avg=None):
    """运行单个测试用例"""
    print(f"\n{'='*60}")
    print(f"测试用例: {name}")
    print(f"{'='*60}")
    print(f"问题: {question}")
    print(f"输出文本: {outputs_text}")
    
    response = MockResponse(prompt, outputs_text)
    result = process_single_q_eval(0, question, response)
    
    print(f"\n结果:")
    print(f"  提取的分数: {result['reward_info']['scores']}")
    print(f"  平均分数: {result['reward_info']['avg_score']}")
    print(f"  奖励值: {result['reward']}")
    
    # 验证结果
    success = True
    if expected_scores is not None:
        if result['reward_info']['scores'] != expected_scores:
            print(f"  ❌ 失败: 期望分数 {expected_scores}, 实际 {result['reward_info']['scores']}")
            success = False
        else:
            print(f"  ✅ 分数匹配")
    
    if expected_avg is not None:
        if abs(result['reward_info']['avg_score'] - expected_avg) > 0.001:
            print(f"  ❌ 失败: 期望平均分数 {expected_avg}, 实际 {result['reward_info']['avg_score']}")
            success = False
        else:
            print(f"  ✅ 平均分数匹配")
    
    return success


def main():
    """运行所有测试用例"""
    question = "What is 2+2?"
    prompt = f"Question: {question}"
    
    test_results = []
    
    # 测试用例 1: 标准 JSON 格式（整数）
    test_results.append(test_case(
        "标准 JSON 格式（整数）",
        question,
        prompt,
        ['{"score": 7}'],
        [7.0],
        7.0
    ))
    
    # 测试用例 2: 标准 JSON 格式（浮点数）
    test_results.append(test_case(
        "标准 JSON 格式（浮点数）",
        question,
        prompt,
        ['{"score": 8.5}'],
        [8.5],
        8.5
    ))
    
    # 测试用例 3: 带尾随空白
    test_results.append(test_case(
        "带尾随空白",
        question,
        prompt,
        ['{"score": 6}   '],
        [6.0],
        6.0
    ))
    
    # 测试用例 4: 带尾随标点符号
    test_results.append(test_case(
        "带尾随标点符号",
        question,
        prompt,
        ['{"score": 5}...'],
        [5.0],
        5.0
    ))
    
    # 测试用例 5: 带换行符
    test_results.append(test_case(
        "带换行符",
        question,
        prompt,
        ['{"score": 9}\n'],
        [9.0],
        9.0
    ))
    
    # 测试用例 6: 多个输出
    test_results.append(test_case(
        "多个输出",
        question,
        prompt,
        ['{"score": 7}', '{"score": 8}', '{"score": 9}'],
        [7.0, 8.0, 9.0],
        8.0
    ))
    
    # 测试用例 7: 边界值 - 最小值
    test_results.append(test_case(
        "边界值 - 最小值 0",
        question,
        prompt,
        ['{"score": 0}'],
        [0.0],
        0.0
    ))
    
    # 测试用例 8: 边界值 - 最大值
    test_results.append(test_case(
        "边界值 - 最大值 10",
        question,
        prompt,
        ['{"score": 10}'],
        [10.0],
        10.0
    ))
    
    # 测试用例 9: 超出范围 - 负数（应该被限制为0）
    test_results.append(test_case(
        "超出范围 - 负数",
        question,
        prompt,
        ['{"score": -5}'],
        [0.0],
        0.0
    ))
    
    # 测试用例 10: 超出范围 - 超过10（应该被限制为10）
    test_results.append(test_case(
        "超出范围 - 超过10",
        question,
        prompt,
        ['{"score": 15}'],
        [10.0],
        10.0
    ))
    
    # 测试用例 11: 正则表达式提取 - JSON 在文本中
    test_results.append(test_case(
        "正则表达式提取 - JSON 在文本中",
        question,
        prompt,
        ['Some text before {"score": 7} some text after'],
        [7.0],
        7.0
    ))
    
    # 测试用例 12: 正则表达式提取 - 浮点数
    test_results.append(test_case(
        "正则表达式提取 - 浮点数",
        question,
        prompt,
        ['Some text {"score": 6.5} more text'],
        [6.5],
        6.5
    ))
    
    # 测试用例 13: 无效 JSON - 缺少引号
    test_results.append(test_case(
        "无效 JSON - 缺少引号",
        question,
        prompt,
        ['{score: 7}'],
        [0.0],  # 应该fallback到0.0
        0.0
    ))
    
    # 测试用例 14: 无效 JSON - 完全无法解析
    test_results.append(test_case(
        "无效 JSON - 完全无法解析",
        question,
        prompt,
        ['This is not JSON at all'],
        [0.0],
        0.0
    ))
    
    # 测试用例 15: 空输出
    test_results.append(test_case(
        "空输出",
        question,
        prompt,
        [''],
        [0.0],
        0.0
    ))
    
    # 测试用例 16: 空 outputs 列表
    test_results.append(test_case(
        "空 outputs 列表",
        question,
        prompt,
        [],
        [],
        0.0
    ))
    
    # 测试用例 17: JSON 缺少 score 字段
    test_results.append(test_case(
        "JSON 缺少 score 字段",
        question,
        prompt,
        ['{"other": "value"}'],
        [0.0],
        0.0
    ))
    
    # 测试用例 18: score 字段不是数字
    test_results.append(test_case(
        "score 字段不是数字",
        question,
        prompt,
        ['{"score": "seven"}'],
        [0.0],
        0.0
    ))
    
    # 测试用例 19: 混合情况 - 有效和无效
    test_results.append(test_case(
        "混合情况 - 有效和无效",
        question,
        prompt,
        ['{"score": 8}', 'invalid json', '{"score": 6}'],
        [8.0, 0.0, 6.0],
        (8.0 + 0.0 + 6.0) / 3.0
    ))
    
    # 测试用例 20: 带空格的 JSON
    test_results.append(test_case(
        "带空格的 JSON",
        question,
        prompt,
        ['{ "score" : 7 }'],
        [7.0],
        7.0
    ))
    
    # 测试用例 21: 多个 JSON 对象（正则表达式会匹配最后一个）
    test_results.append(test_case(
        "多个 JSON 对象（正则提取最后一个）",
        question,
        prompt,
        ['{"score": 5} {"score": 8}'],
        [8.0],  # 正则表达式会匹配最后一个（通常最后一个可能是最终答案）
        8.0
    ))
    
    # 测试用例 22: 小数边界 - 0.0
    test_results.append(test_case(
        "小数边界 - 0.0",
        question,
        prompt,
        ['{"score": 0.0}'],
        [0.0],
        0.0
    ))
    
    # 测试用例 23: 小数边界 - 10.0
    test_results.append(test_case(
        "小数边界 - 10.0",
        question,
        prompt,
        ['{"score": 10.0}'],
        [10.0],
        10.0
    ))
    
    # 测试用例 24: 小数 - 中间值
    test_results.append(test_case(
        "小数 - 中间值",
        question,
        prompt,
        ['{"score": 7.25}'],
        [7.25],
        7.25
    ))
    
    # 测试用例 25: 尾随多个标点
    test_results.append(test_case(
        "尾随多个标点",
        question,
        prompt,
        ['{"score": 6}!!!'],
        [6.0],
        6.0
    ))
    
    # 统计结果
    print(f"\n{'='*60}")
    print(f"测试总结")
    print(f"{'='*60}")
    passed = sum(test_results)
    total = len(test_results)
    print(f"通过: {passed}/{total}")
    print(f"失败: {total - passed}/{total}")
    
    if passed == total:
        print("✅ 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败！")
        return 1


if __name__ == '__main__':
    exit(main())
