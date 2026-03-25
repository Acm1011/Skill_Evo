# Requires vllm>=0.8.5
import torch
import vllm
from vllm import LLM
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json
from datasets import load_dataset

def collection_questions(data_path):
    """
    读取 DeepMath-103K 数据集，按 topic 收集所有问题
    
    Args:
        data_path: 数据集路径，例如 '/root/users/ycy/data/DeepMath-103K'
    
    Returns:
        dict: {topic: [question1, question2, ...]}
    """
    dataset = load_dataset(data_path, split='train')
    topic_questions = defaultdict(list)
    topic_question_cnt = defaultdict(int)
    for line in dataset:
        topic_questions[line['topic']].append(line['question'])
        topic_question_cnt[line['topic']] += 1

    print(f'{topic_question_cnt=}')
    return topic_questions


def save_topic_questions(topic_questions, output_path):
    """
    保存 topic-question 字典到 JSON 文件
    
    Args:
        topic_questions: {topic: [questions]} 字典
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(topic_questions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved topic-questions to {output_path}")



def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def test_embedding():
    """测试 embedding 功能"""
    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    input_texts = queries + documents

    model = LLM(model="Qwen/Qwen3-Embedding-0.6B", task="embed")

    outputs = model.embed(input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    scores = (embeddings[:2] @ embeddings[2:].T)
    print(scores.tolist())
    # [[0.7620252966880798, 0.14078938961029053], [0.1358368694782257, 0.6013815999031067]]


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 读取数据
    data_path = '/root/users/ycy/data/DeepMath-103K'
    topic_questions = read_data(data_path)
    
    # 可选：保存到 JSON 文件
    save_topic_questions(topic_questions, 'se_code/topic_questions.json')
    
    # 示例：查看某个 topic 的问题
    example_topic = list(topic_questions.keys())[0]
    print(f"\nExample topic: {example_topic}")
    print(f"Number of questions: {len(topic_questions[example_topic])}")
    print(f"First question: {topic_questions[example_topic][0][:100]}...")
    
    # 如果需要测试 embedding，取消下面的注释
    # test_embedding()
