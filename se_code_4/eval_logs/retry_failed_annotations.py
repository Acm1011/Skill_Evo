import os
import json
import random
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from concurrent.futures import ThreadPoolExecutor, as_completed

# 复用原始文件中的 prompts
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
  "desc": {{
  "Scope":"..."
  "Key Concepts":"..."
  "Tools/Theorems":"..."
  }},
  "cue": "..."
}}"""


@retry(stop=stop_after_attempt(10), wait=wait_fixed(5))
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


def find_failed_annotations(annotations_file, sampled_questions_file):
    """
    找出所有解析失败的样本
    条件：difficulty_question_counts[difficulty] > 0 但是 annotations[difficulty] 不存在或为 None
    """
    # 读取标注文件
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations_data = json.load(f)
    
    # 读取采样的问题文件
    with open(sampled_questions_file, 'r', encoding='utf-8') as f:
        sampled_questions = json.load(f)
    
    failed_tasks = []
    
    for topic, topic_data in annotations_data.items():
        difficulty_counts = topic_data.get('difficulty_question_counts', {})
        annotations = topic_data.get('annotations', {})
        
        # 检查每个难度等级
        for difficulty, count in difficulty_counts.items():
            # 如果该难度有问题（count > 0），但标注不存在或为 None
            if count > 0:
                if difficulty not in annotations or annotations[difficulty] is None:
                    # 检查采样数据中是否有对应的问题
                    if topic in sampled_questions and difficulty in sampled_questions[topic]:
                        questions = sampled_questions[topic][difficulty]
                        if len(questions) > 0:
                            failed_tasks.append({
                                'topic': topic,
                                'difficulty': difficulty,
                                'questions': questions,
                                'count': count
                            })
                            print(f"发现失败样本: Topic={topic}, Difficulty={difficulty}, Count={count}, Questions={len(questions)}")
                        else:
                            print(f"警告: Topic={topic}, Difficulty={difficulty} 在采样数据中没有问题，跳过")
                    else:
                        print(f"警告: Topic={topic}, Difficulty={difficulty} 在采样数据中不存在，跳过")
    
    return failed_tasks


def build_prompt(topic, difficulty, questions):
    """构建完整的 prompt"""
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    full_user_prompt = user_prompt.format(
        TOPIC_NAME=topic,
        DIFFICULTY_LEVEL=difficulty,
        Question=questions_text
    )
    full_prompt = f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n=== USER PROMPT ===\n{full_user_prompt}"
    return full_prompt


def sanitize_filename(filename):
    """
    清理文件名，移除或替换不安全的字符
    """
    # 替换不安全的字符
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '->']
    safe_filename = filename
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, '_')
    # 移除多余的空格和下划线
    safe_filename = '_'.join(safe_filename.split())
    return safe_filename


def save_failed_prompt(output_dir, topic, difficulty, full_prompt, questions):
    """
    保存失败的 prompt 到 txt 文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 清理 topic 和 difficulty 作为文件名
    safe_topic = sanitize_filename(topic)
    safe_difficulty = sanitize_filename(str(difficulty))
    
    # 构建文件名
    filename = f"{safe_topic}_{safe_difficulty}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # 构建文件内容
    content = f"Topic: {topic}\n"
    content += f"Difficulty: {difficulty}\n"
    content += f"Questions Count: {len(questions)}\n"
    content += "=" * 80 + "\n\n"
    content += full_prompt
    content += "\n\n" + "=" * 80 + "\n"
    content += "=== QUESTIONS ===\n"
    for i, q in enumerate(questions, 1):
        content += f"{i}. {q}\n"
    
    # 保存文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已保存失败的 prompt 到: {filepath}")




def main():
    # 文件路径
    base_dir = '/root/users/ycy/Self-evolving-Agent/se_code/prompt2'
    annotations_file = os.path.join(base_dir, 'all_topic_annotations.json')
    sampled_questions_file = os.path.join(base_dir, 'topic_difficulty_questions_sampled.json')
    
    # 输出目录
    output_dir = os.path.join(base_dir, 'failed_prompts')
    
    print("=" * 80)
    print("开始查找解析失败的样本...")
    print("=" * 80)
    
    # 找出所有失败的样本
    failed_tasks = find_failed_annotations(annotations_file, sampled_questions_file)
    
    if len(failed_tasks) == 0:
        print("\n没有发现失败的样本，所有标注都已成功！")
        return
    
    print(f"\n共发现 {len(failed_tasks)} 个失败的样本")
    print("=" * 80)
    print("\n失败的样本信息：")
    print("-" * 80)
    
    # 为每个失败的样本构建 prompt 并保存
    for i, task in enumerate(failed_tasks, 1):
        topic = task['topic']
        difficulty = task['difficulty']
        questions = task['questions']
        
        # 打印失败信息
        print(f"\n[{i}/{len(failed_tasks)}] 失败样本:")
        print(f"  Topic: {topic}")
        print(f"  Difficulty: {difficulty}")
        print(f"  问题数量: {len(questions)}")
        
        # 构建 prompt
        questions_text = "\n".join([f"{j+1}. {q}" for j, q in enumerate(questions)])
        full_user_prompt = user_prompt.format(
            TOPIC_NAME=topic,
            DIFFICULTY_LEVEL=difficulty,
            Question=questions_text
        )
        full_prompt = f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n=== USER PROMPT ===\n{full_user_prompt}"
        
        # 保存为单独的 txt 文件
        save_failed_prompt(
            output_dir=output_dir,
            topic=topic,
            difficulty=difficulty,
            full_prompt=full_prompt,
            questions=questions
        )
    
    print("\n" + "=" * 80)
    print(f"处理完成！共保存 {len(failed_tasks)} 个失败的 prompt 文件")
    print(f"保存目录: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

