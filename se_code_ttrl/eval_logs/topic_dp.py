from datasets import load_dataset
import os
import json
import random
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_fixed
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

system_prompt="""You are an expert-level "Mathematical Problem Deconstructor." Your primary function is to analyze a given mathematics topic, a set of example questions, and a target difficulty level.

Your goal is to synthesize this information into a compact "Topic Profile" in valid JSON format. This profile will serve as a high-level blueprint for a "Challenger" LLM to generate new, high-quality problems.

You MUST follow this two-step process:

**1. Internal Analysis (Private):**
First, you MUST think step-by-step to deconstruct the inputs.
* **Analyze Topic & Examples:** What are the common themes, required tools (e.g., Vieta's, modular arithmetic), and underlying problem structures of the example questions?
* **Analyze Difficulty:** How does the specified difficulty level (e.g., "AIME-level") interact with these themes? (e.g., "Introductory" implies direct application; "Advanced" implies a non-obvious combination of multiple concepts or a clever insight).
* **Synthesize `desc`:** Generalize from the examples to define the topic's boundaries, core concepts, and essential tools.
* **Synthesize `cue`:** Formulate an *actionable archetype* for a new problem. This must fuse the *style* of the examples with the *complexity* of the difficulty level.

**2. Final JSON Output:**
After your internal analysis, your entire response MUST be a single, valid JSON object. Do not include any text, explanations, or markdown formatting before or after the JSON block.

---
**JSON FIELD SPECIFICATIONS:**

* **`"desc"` (Topic Description):**
    * This field must be a concise, technical description of the topic.
    * It must define the topic's **Scope** (what is in-bounds), **Key Concepts** (the core ideas), and common **Tools/Theorems** (how solutions are built), generalized from the provided examples.

* **`"cue"` (Generative Cue):**
    * This is the most critical field. It must be an **actionable archetype** or "generative seed" for the Challenger LLM.
    * It must describe *how* to construct a new problem, synthesizing the problem-solving *style* from the examples with the *complexity constraint* from the difficulty level.
    * It must be specific and generative (e.g., "A multi-step problem requiring the combination of [Concept X] with [Constraint Y] to find a unique parameter Z").
    * It must **NOT** be a generic command (e.g., "Make a hard problem").
"""
user_prompt="""Analyze the following inputs and generate your analysis in the required JSON format.

**Topic:**
{TOPIC_NAME}

**Difficulty Level:**
{DIFFICULTY_LEVEL}

**Example Questions:**
{Question}

**Required Output Format (JSON only):**
{{
  "desc": "...",
  "cue": "..."
}}"""

data_path_dir='/root/users/ycy/data'
data_name='DeepMath-103K'
data = load_dataset(os.path.join(data_path_dir, data_name),split='train')
topic2dif2data=defaultdict(lambda: defaultdict(list))
for line in data:
    q = line['question']
    topic = line['topic']
    difficulty=line['difficulty']
    topic2dif2data[topic][str(difficulty)].append(q)

# 先收集所有可能的难度等级
all_difficulties = set()
for dif2data in topic2dif2data.values():
    all_difficulties.update(dif2data.keys())
all_difficulties = sorted(all_difficulties)  # 排序以便统一顺序

# 统计每个topic的数量和每个难度等级的数量
topic_stats = {}
topic_difficulty_stats = {}
for topic, dif2data in topic2dif2data.items():
    # 初始化每个难度等级的数量为0（对于不存在的难度等级，保持为0）
    difficulty_counts = {dif: 0 for dif in all_difficulties}
    
    # 对于存在的难度等级，直接从dif2data中获取其问题数量（预先得到的）
    for dif, questions in dif2data.items():
        difficulty_counts[dif] += len(questions)  # 直接获取该难度等级的问题数量
    
    # 计算总数量
    total_count = sum(difficulty_counts.values())
    topic_stats[topic] = total_count
    topic_difficulty_stats[topic] = difficulty_counts
    
    print(f"Topic: {topic}, 总题目数: {total_count}")
    for dif, count in difficulty_counts.items():
        print(f"  Difficulty {dif}: {count} 个问题")

# 从每个topic的每个difficulty中随机抽取min(len(difficulty), 10)个question
topic2dif2sampled = defaultdict(lambda: defaultdict(list))
for topic, dif2data in topic2dif2data.items():
    for difficulty, questions in dif2data.items():
        if len(questions) == 0:
            print(f"警告: Topic: {topic}, Difficulty: {difficulty} 没有题目，跳过采样")
            continue
        sample_size = min(len(questions), 10)
        sampled_questions = random.sample(questions, sample_size)
        topic2dif2sampled[topic][difficulty] = sampled_questions
        print(f"Topic: {topic}, Difficulty: {difficulty}, 原始数量: {len(questions)}, 抽取数量: {sample_size}")

# 保存抽取后的数据
output_dir = '/root/users/ycy/Self-evolving-Agent/se_code/prompt2'
os.makedirs(output_dir, exist_ok=True)

# 保存抽取后的JSON文件
json_path = os.path.join(output_dir, 'topic_difficulty_questions_sampled.json')
topic2dif2sampled_dict = {topic: {dif: questions for dif, questions in dif2data.items()} 
                          for topic, dif2data in topic2dif2sampled.items()}
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(topic2dif2sampled_dict, f, ensure_ascii=False, indent=2)
print(f"已保存抽取后的JSON文件: {json_path}")

# 保存统计信息（包含每个topic的总数和每个difficulty的原始数量及抽取情况）
stats = {}
for topic, dif2data in topic2dif2sampled.items():
    stats[topic] = {
        'total_count': topic_stats[topic],
        'difficulty_counts': topic_difficulty_stats[topic],  # 每个难度等级的原始数量
        'sampled_by_difficulty': {dif: len(questions) for dif, questions in dif2data.items()}  # 每个难度等级的采样数量
    }
stats_path = os.path.join(output_dir, 'topic_difficulty_stats_sampled.json')
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"已保存统计信息: {stats_path}")

@retry(stop=stop_after_attempt(10), wait=wait_fixed(5))  # 最多尝试10次，每次重试间隔5秒
def process_example(topic, difficulty_level, questions):
    """
    调用LLM生成topic信息的标注结果
    Args:
        topic: topic名称
        difficulty_level: difficulty级别
        questions: questions列表
    Returns:
        (response_content, error): LLM返回的内容和错误信息
    """
    api_urls=['https://fast.ominiai.cn/v1/chat/completions']
    api_keys=['sk-m7dpHEcWQkQlgoRuOm9S0mYeurwC9BTMpmLnXDrpPOmmkn98','sk-Lplron1sCtErNjSaUDxhgwFuwCyjGKfwJpa9cwFNXcJsMNyB']
    try:
        # 将questions格式化为字符串（每个question一行，带编号）
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        example = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(
                    TOPIC_NAME=topic,
                    DIFFICULTY_LEVEL=difficulty_level,
                    Question=questions_text
                )}
            ],
            "temperature": 0.1
        }
        api_index = random.randint(0, len(api_urls)-1)
        key_index = random.randint(0, len(api_keys)-1)
        api_url = api_urls[api_index]
        api_key = api_keys[key_index]
        gpt_response = requests.post(api_url, headers={"Authorization": f'Bearer {api_key}',"Content-Type": "application/json"}, json=example, timeout=20)
        gpt_response.raise_for_status()  # 检查HTTP状态码，如果不是200会抛出异常
        return gpt_response.json()['choices'][0]['message']['content'],None
    except Exception as e:
        print(f"Error in process_example: {e}")
        
        return "No",{'error':f"Time out: {str(e)}"}


def parse_json_response(response_content):
    """
    从LLM响应中解析JSON
    """
    if not response_content:
        return None
    
    try:
        # 尝试直接解析
        response_content = response_content.strip()
        if response_content.startswith('{'):
            return json.loads(response_content)
        
        # 尝试提取JSON块（如果被markdown代码块包裹，如 ```json ... ```）
        # 先尝试找到第一个 { 和最后一个 }
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_content[start_idx:end_idx+1]
            return json.loads(json_str)
        
        return None
    except Exception as e:
        print(f"解析JSON失败: {e}, 响应内容: {response_content[:200]}")
        return None

# 存储所有topic的标注结果
topic_annotations = {}

# 收集所有任务，保持顺序信息
tasks = []
for topic_idx, (topic, dif2data) in enumerate(topic2dif2sampled.items()):
    # 获取该topic的总问题数量（原始数据集中该topic的总数，不是采样后的）
    topic_total_count = topic_stats[topic]
    # 获取该topic下每个难度等级的问题数量（原始数据集中每个难度等级的总数）
    topic_difficulty_counts = topic_difficulty_stats[topic]
    
    # 存储该topic下所有difficulty的标注结果
    topic_annotation = {
        'total_question_count': topic_total_count,
        'difficulty_question_counts': topic_difficulty_counts,
        'annotations': {}
    }
    topic_annotations[topic] = topic_annotation
    
    # 收集该topic下的所有difficulty任务
    # 注意：只处理采样后实际存在的difficulty（即dif2data中的difficulty）
    for diff_idx, (difficulty, questions) in enumerate(dif2data.items()):
        if len(questions) == 0:
            print(f"警告: Topic: {topic}, Difficulty: {difficulty} 采样后没有题目，跳过")
            continue
        tasks.append({
            'topic': topic,
            'difficulty': difficulty,
            'questions': questions,
            'topic_idx': topic_idx,
            'diff_idx': diff_idx
        })

# 定义处理单个任务的函数
def process_task(task):
    """处理单个任务并返回结果"""
    topic = task['topic']
    difficulty = task['difficulty']
    questions = task['questions']
    topic_idx = task['topic_idx']
    diff_idx = task['diff_idx']
    
    print(f"正在处理 Topic: {topic}, Difficulty: {difficulty}...")
    
    # 调用LLM生成标注结果
    response_content, error = process_example(topic, difficulty, questions)
    
    if error:
        print(f"错误: {error}")
        result = {
            'topic': topic,
            'difficulty': difficulty,
            'topic_idx': topic_idx,
            'diff_idx': diff_idx,
            'error': error,
            'raw_response': None,
            'parsed_json': None
        }
    else:
        # 解析JSON响应
        parsed_json = parse_json_response(response_content)
        
        result = {
            'topic': topic,
            'difficulty': difficulty,
            'topic_idx': topic_idx,
            'diff_idx': diff_idx,
            'error': None,
            'raw_response': response_content,
            'parsed_json': parsed_json
        }
        
        print(f"成功处理 Topic: {topic}, Difficulty: {difficulty}")
    
    return result

# 使用多线程并行处理，保持顺序
max_workers = min(os.cpu_count(),100)  # 可以根据需要调整线程数
results = [None] * len(tasks)  # 预分配结果列表，保持顺序

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务，并记录索引
    future_to_index = {executor.submit(process_task, task): idx for idx, task in enumerate(tasks)}
    
    # 收集结果，按完成顺序处理
    for future in as_completed(future_to_index):
        idx = future_to_index[future]
        try:
            result = future.result()
            results[idx] = result  # 按原始索引保存结果，保持顺序
        except Exception as e:
            print(f"任务执行异常: {e}")
            # 获取原始任务信息
            task = tasks[idx]
            results[idx] = {
                'topic': task['topic'],
                'difficulty': task['difficulty'],
                'topic_idx': task['topic_idx'],
                'diff_idx': task['diff_idx'],
                'error': {'error': str(e)},
                'raw_response': None,
                'parsed_json': None
            }

# 按照原始顺序将结果保存到对应的topic_annotation中
for result in results:
    if result is None:
        print("警告: 发现None结果，跳过")
        continue
    topic = result['topic']
    difficulty = result['difficulty']
    
    # 确保topic存在于topic_annotations中
    if topic not in topic_annotations:
        print(f"警告: Topic {topic} 不在topic_annotations中，跳过")
        continue
    
    # 只保存解析后的JSON内容（desc和cue），如果解析失败或出错则保存None
    if result['error'] or not result['parsed_json']:
        topic_annotations[topic]['annotations'][difficulty] = None
    else:
        # 直接保存parsed_json的内容（包含desc和cue）
        topic_annotations[topic]['annotations'][difficulty] = result['parsed_json']

# 保存所有topic的标注结果到一个JSON文件
all_annotations_path = os.path.join(output_dir, 'all_topic_annotations.json')
with open(all_annotations_path, 'w', encoding='utf-8') as f:
    json.dump(topic_annotations, f, ensure_ascii=False, indent=2)
print(f"\n所有标注结果已保存到: {all_annotations_path}")


