import copy
import logging
import os
import random
import re
import json
from collections import defaultdict
from typing import Optional
from string import Template
import math

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


logger = logging.getLogger(__name__)


def get_se_base_path():
    """获取 Self-evolving-Agent 项目的基础路径，优先从环境变量读取"""
    base_dir = os.environ.get("SE_BASE_DIR", "/home/ycy/data1")
    project_name = os.environ.get("SE_PROJECT_NAME", "Self-evolving-Agent")
    code_module = os.environ.get("SE_CODE_MODULE", "se_code_auto")
    return os.path.join(base_dir, project_name, code_module)


def get_prompt_path(subdir=""):
    """获取 prompt 目录路径"""
    base = os.environ.get("SE_PROMPT_DIR", get_se_base_path())
    if subdir:
        return os.path.join(base, subdir)
    return base


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}

def get_prompts_topic(num_querys:int):
    prompt_path = get_prompt_path('prompt4')
    with open(os.path.join(prompt_path,'system.txt'),'r',encoding='utf-8') as f:
        raw_system = f.read()              # 使用 $PLACEHOLDER 占位
    with open(os.path.join(prompt_path,'user.txt'),'r',encoding='utf-8') as f:
        raw_user = f.read()                # 使用 $PLACEHOLDER 占位
    with open(os.path.join(prompt_path,'Cnt_Topic.json'),'r',encoding='utf-8') as f:
        topics = json.load(f)
    topics = ['->'.join(topic.split('->')[1:]).strip() for topic in topics.keys()]
    dataframe=[]
    for idx in range(num_querys):
        topic = random.choice(topics)
        user_prompt=raw_user.format(
            TOPIC=topic
        )
        prompt=[
            {
                'role':'system',
                'content':raw_system
            },
            {
                'role':'user',
                'content':user_prompt
            }
        ]
        dataframe.append(
            {
                'idx': idx,
                'data_source':'Challenger',
                'topic':topic,
                'prompt':prompt,
                'ability':'math'
            }
        )
        
    random.shuffle(dataframe)
    return dataframe




def get_prompts_Topic_AoPS(num_querys:int):
    # if not os.path.exists(topic_path):
    #     raise ValueError(f'{topic_path=} not exist')
    # print(f'{topic_path=}')
    prompt_path = get_prompt_path('prompt2')
    topic_path = os.path.join(prompt_path, 'all_topic_annotations.json')
    with open(os.path.join(prompt_path,'DIFFICULTY_RUBRIC.txt'),'r',encoding='utf-8') as f:
        difficulty_rubric = f.read()
    with open(os.path.join(prompt_path,'DIFFICULTY_EXAMPLES.txt'),'r',encoding='utf-8') as f:
        difficulty_example = f.read()
    with open(os.path.join(prompt_path,'system.txt'),'r',encoding='utf-8') as f:
        raw_system = f.read()              # 使用 $PLACEHOLDER 占位
    with open(os.path.join(prompt_path,'user.txt'),'r',encoding='utf-8') as f:
        raw_user = f.read()                # 使用 $PLACEHOLDER 占位
    topics = json.load(open(os.path.join(topic_path),'r',encoding='utf-8'))
    topics_cnt = sum(v['total_question_count'] for v in topics.values())
    #levels=['','5', '5.5', '6', '6.5', '7', '7.5', '8', '8.5', '9', '9.5', '10']
    levels=['4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5', '9.0']
    dataframe=[]
    # ---------- 配额分配（最大余数法 + 最低覆盖1条可选） ----------
    shares = []
    for t, info in topics.items():
        share = (info['total_question_count'] / topics_cnt) * num_querys
        shares.append((t, share))
    base_alloc = {t: int(math.floor(s)) for t,s in shares}
    total_base = sum(base_alloc.values())
    remain = num_querys - total_base
    residuals = sorted(((s - math.floor(s), t) for t,s in shares), reverse=True)
    for _, t in residuals[:remain]:
        base_alloc[t] += 1
    idx = 0
    for topic, infos in topics.items():
        #core_desc=infos['Core_desc']
        #cue_desc = infos['Cue_desc']
        n = base_alloc[topic]
        out=[]
        # 获取当前 topic 的 difficulty_question_counts 作为权重
        difficulty_counts = infos.get('difficulty_question_counts', {})
        # 构建权重列表，对应 levels 的顺序
        weights = [difficulty_counts.get(level, 0) for level in levels]
        # 如果所有权重都为0，则使用均匀分布
        if sum(weights) == 0:
            weights = None
        for _ in range(n):
            # 按权重采样
            if weights is None:
                target_level = levels[random.randint(0, len(levels)-1)]
            else:
                target_level = np.random.choice(levels, p=np.array(weights) / sum(weights))

            desc=infos['annotations'][str(target_level)]['desc']
            cue=infos['annotations'][str(target_level)]['cue']
            topic_profile=f"Topic:{topic}\\nTopic Description:{desc}\\nQuestion-Generated Cues:{cue}"
            system_prompt=raw_system
            user_prompt=raw_user.format(
                TOPIC_PROFILE=topic_profile,
                TARGET_DIFFICULTY=target_level,
                AOPS_SCALE_DEFINITIONS=difficulty_rubric,
                AOPS_DIFFICULTY_EXAMPLES=difficulty_example
            )
            prompt=[
                {
                    'role':'system',
                    'content':system_prompt
                },
                {
                    'role':'user',
                    'content':user_prompt
                }
            ]
            out.append(
                {
                    'idx': idx,
                    'data_source':'Challenger',
                    'topic':topic,
                    'target_level':target_level,
                    'prompt':prompt,
                    'ability':'math'
                }
            )
            idx += 1 
        dataframe.extend(out)
    random.shuffle(dataframe)
    return dataframe




def get_prompts_weakness(num_querys: int):
    """
    从weakness_data_pool_processed.json中读取数据，使用system.txt和user.txt生成prompt
    """
    prompt_path = get_prompt_path('prompt_weakness')
    data_path = os.path.join(prompt_path, 'weakness_data_pool_processed.json')
    
    # 读取system和user模板
    with open(os.path.join(prompt_path, 'system.txt'), 'r', encoding='utf-8') as f:
        raw_system = f.read()
    with open(os.path.join(prompt_path, 'user.txt'), 'r', encoding='utf-8') as f:
        raw_user = f.read()
    
    # 将{{PLACEHOLDER}}格式转换为{PLACEHOLDER}格式以使用Python的format方法
    # 使用正则表达式精确替换，避免破坏其他内容
    raw_user = re.sub(r'\{\{(\w+)\}\}', r'{\1}', raw_user)
    
    # 读取处理后的数据
    with open(data_path, 'r', encoding='utf-8') as f:
        weakness_data = json.load(f)
    
    # 如果请求的数量超过可用数据，可以重复使用
    if num_querys > len(weakness_data):
        # 重复采样
        selected_data = random.choices(weakness_data, k=num_querys)
    else:
        # 随机选择
        selected_data = random.sample(weakness_data, k=num_querys)
    
    dataframe = []
    for idx, item in enumerate(selected_data):
        # 提取信息
        topic = item['topic']
        examples = item['examples']
        target_difficulty = item['difficulty_max']  # 使用difficulty_max作为target_difficulty
        
        # 格式化user prompt
        user_prompt = raw_user.format(
            TOPIC=topic,
            EXAMPLE_1_DIFFICULTY=examples[0]['difficulty'],
            EXAMPLE_1_CONTENT=examples[0]['problem'],
            EXAMPLE_2_DIFFICULTY=examples[1]['difficulty'],
            EXAMPLE_2_CONTENT=examples[1]['problem'],
            EXAMPLE_3_DIFFICULTY=examples[2]['difficulty'],
            EXAMPLE_3_CONTENT=examples[2]['problem'],
            TARGET_DIFFICULTY=target_difficulty
        )
        
        # 构建prompt
        prompt = [
            {
                'role': 'system',
                'content': raw_system
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
        
        dataframe.append({
            'idx': idx,
            'data_source': 'Challenger',
            'topic': topic,
            'target_level': target_difficulty,
            'prompt': prompt,
            'ability': 'math'
        })
    
    random.shuffle(dataframe)
    return dataframe


def get_prompts_weakness_icl(num_querys: int):
    """
    从weakness_data_pool.json中读取数据，按acc字段均匀采样，生成ICL prompt
    """
    prompt_path = get_prompt_path('prompt_weakness')
    template_path = os.path.join(prompt_path, '3-shot-icl-end-prompt')
    data_path = os.path.join(prompt_path, 'weakness_data_pool.json')
    
    # 读取prompt模板
    with open(template_path, 'r', encoding='utf-8') as f:
        raw_prompt = f.read()
    
    # 读取原始数据
    with open(data_path, 'r', encoding='utf-8') as f:
        weakness_data = json.load(f)
    
    # 提取所有数据及其acc值
    data_list = []
    for key, item in weakness_data.items():
        acc = item.get('extra_info', {}).get('acc', None)
        problem = item.get('extra_info', {}).get('problem', None)
        if acc is not None and problem is not None:
            data_list.append({
                'key': key,
                'acc': acc,
                'problem': problem,
                'topic': item.get('extra_info', {}).get('topic', ''),
                'difficulty': item.get('extra_info', {}).get('difficulty', 0)
            })
    
    # 按acc值分桶，实现均匀采样
    # 将acc值分成若干个区间，从每个区间均匀采样
    num_bins = 10
    bins = [[] for _ in range(num_bins)]
    for item in data_list:
        # acc范围通常是0-1，映射到0-9的桶
        bin_idx = min(int(item['acc'] * num_bins), num_bins - 1)
        bins[bin_idx].append(item)
    
    # 从每个非空桶中均匀采样
    non_empty_bins = [b for b in bins if len(b) > 0]
    samples_per_bin = num_querys // len(non_empty_bins) if non_empty_bins else 0
    remainder = num_querys % len(non_empty_bins) if non_empty_bins else 0
    
    selected_data = []
    for i, bin_data in enumerate(non_empty_bins):
        # 为前remainder个桶多分配一个样本
        n_samples = samples_per_bin + (1 if i < remainder else 0)
        if n_samples > 0:
            if n_samples >= len(bin_data):
                selected_data.extend(random.choices(bin_data, k=n_samples))
            else:
                selected_data.extend(random.sample(bin_data, k=n_samples))
    
    # 打乱顺序
    random.shuffle(selected_data)
    
    # 生成dataframe
    dataframe = []
    for idx, item in enumerate(selected_data):
        # 填充reference_question到模板
        user_prompt = raw_prompt.format(reference_question=item['problem'])
        
        prompt = [
            {
                'role': 'user',
                'content': user_prompt # 1416 token
            }
        ]
        
        dataframe.append({
            'idx': idx,
            'data_source': 'Challenger',
            'topic': item['topic'],
            'target_level': item['difficulty'],
            'reference_acc': item['acc'],
            'reference_question': item['problem'],
            'prompt': prompt,
            'ability': 'math'
        })
    
    return dataframe

def get_prompts(num_querys, get_prompts_func):
    if get_prompts_func == "R_Zero":
        return get_prompts_R_zero(num_querys)
    elif get_prompts_func == "Topic":
        return get_prompts_topic(num_querys)
    elif get_prompts_func == "Topic_AoPS":
        return get_prompts_Topic_AoPS(num_querys)
    elif get_prompts_func == "weakness":
        return get_prompts_weakness(num_querys)
    elif get_prompts_func == "weakness_icl":
        return get_prompts_weakness_icl(num_querys)
    else:
        raise ValueError(f"Invalid get_prompts_func: {get_prompts_func}")


def get_prompts_R_zero(num_querys):
    chat = [
        {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter.\n"
                "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                "Avoid re-using textbook clichés or famous contest problems.\n"
                "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                "<question>\n"
                "{The full problem statement on one or more lines}\n"
                "</question>\n\n"
                r"\boxed{final_answer}"
                "\n\n"
                "Do NOT output anything else—no explanations, no extra markup."
            )
        },
        {
            "role": "user",
            "content": (
                "Generate one new, challenging reasoning question now. "
                "Remember to format the output exactly as instructed."
            )
        }
    ]
    dataframe=[
        {
            'idx': idx,
            'data_source': 'Challenger',
            'topic':'Challenger',
            'prompt': chat,
            'ability': 'math'
           
        }
        for idx in range(num_querys)
    ]
    return dataframe


class ChallengerTopicDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
    ):
        
        self.tokenizer = tokenizer
        self.dynamic_topics = config.get('dynamic_topics',False)
        default_topic_path = os.path.join(get_prompt_path('prompt2'), 'all_topic_annotations.json')
        if self.dynamic_topics:
            self.topic_path = config.get('topics_path', default_topic_path)
        else:
            self.topic_path = default_topic_path
        print(f'{self.dynamic_topics=} ,{self.topic_path=}')
        self.num_querys = config.get("num_querys", 1000)
        self.get_prompts_func = config.get("get_prompts_func", "Topic_AoPS")
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        print(f'{self.max_prompt_length=}')
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self._build_dataset()
     
    def __len__(self):
        return len(self.dataframe)

    def _build_dataset(self):
        self.dataframe = get_prompts(num_querys=self.num_querys, get_prompts_func=self.get_prompts_func)
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages: dict = row_dict.pop('prompt')
        #print(f'{messages=}')
        model_inputs = {}
        if self.apply_chat_template_kwargs.get("chat_template") is None:
            assert hasattr(self.tokenizer, "chat_template"), (
                "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                "models like GLM can copy chat_template.jinja from instruct models"
            )
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        #print(f'{raw_prompt=}')
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        
        position_ids = compute_position_id_with_mask(attention_mask)
        #row_dict = {}
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
            row_dict['raw_inputs'] = raw_prompt

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()


def test_get_prompts_weakness_icl():
    """
    测试 get_prompts_weakness_icl 函数的正确性和边界情况
    """
    from transformers import AutoTokenizer
    from collections import Counter
    
    tokenizer_path = "/home/ycy/data1/models/Qwen3-4B-Base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print("=" * 60)
    print("测试 get_prompts_weakness_icl 函数")
    print("=" * 60)
    
    # 测试1: 基本功能测试
    print("\n[测试1] 基本功能测试 (num_querys=100)")
    try:
        result = get_prompts_weakness_icl(100)
        print(f"  ✓ 返回数据条数: {len(result)}")
        assert len(result) == 100, f"期望100条，实际{len(result)}条"
        print("  ✓ 数据条数正确")
    except Exception as e:
        print(f"  ✗ 基本功能测试失败: {e}")
        return
    
    # 测试2: 检查返回数据的字段完整性
    print("\n[测试2] 字段完整性检查")
    required_fields = ['idx', 'data_source', 'topic', 'target_level', 'reference_acc', 'prompt', 'ability']
    sample = result[0]
    for field in required_fields:
        assert field in sample, f"缺少字段: {field}"
    print(f"  ✓ 所有必需字段存在: {required_fields}")
    
    # 测试3: 检查prompt格式
    print("\n[测试3] Prompt格式检查")
    prompt = sample['prompt']
    assert isinstance(prompt, list), "prompt应该是列表"
    assert len(prompt) == 1, "prompt应该只有一个user消息"
    assert prompt[0]['role'] == 'user', "消息角色应该是user"
    assert 'content' in prompt[0], "消息应该包含content"
    assert '{reference_question}' not in prompt[0]['content'], "reference_question占位符应该被替换"
    print("  ✓ Prompt格式正确")
    
    # 测试4: 边界情况 - num_querys=0
    print("\n[测试4] 边界情况: num_querys=0")
    try:
        result_0 = get_prompts_weakness_icl(0)
        print(f"  ✓ 返回数据条数: {len(result_0)}")
    except Exception as e:
        print(f"  ⚠ num_querys=0 时出现异常: {e}")
    
    # 测试5: 边界情况 - num_querys=1
    print("\n[测试5] 边界情况: num_querys=1")
    result_1 = get_prompts_weakness_icl(1)
    print(f"  ✓ 返回数据条数: {len(result_1)}")
    assert len(result_1) == 1, "应该返回1条数据"
    
    # 测试6: 检查acc分布的均匀性
    print("\n[测试6] ACC分布均匀性检查 (num_querys=1000)")
    result_large = get_prompts_weakness_icl(1000)
    acc_bins = [0] * 10
    for item in result_large:
        bin_idx = min(int(item['reference_acc'] * 10), 9)
        acc_bins[bin_idx] += 1
    
    print("  ACC区间分布:")
    for i, count in enumerate(acc_bins):
        acc_range = f"[{i*0.1:.1f}, {(i+1)*0.1:.1f})"
        bar = "█" * (count // 10)
        print(f"    {acc_range}: {count:4d} {bar}")
    
    # 检查均匀性（每个桶的数量差异不应太大）
    non_zero_bins = [c for c in acc_bins if c > 0]
    if non_zero_bins:
        avg = sum(non_zero_bins) / len(non_zero_bins)
        max_deviation = max(abs(c - avg) for c in non_zero_bins)
        print(f"  非空桶数量: {len(non_zero_bins)}, 平均每桶: {avg:.1f}, 最大偏差: {max_deviation:.1f}")
    
    # 测试7: Token数量统计 (使用apply_chat_template)
    print("\n[测试7] Token数量统计 (使用全部1000条数据, apply_chat_template)")
    import matplotlib.pyplot as plt
    
    token_counts = []
    acc_values = []
    for idx, item in enumerate(result_large):
        # 使用 apply_chat_template 进行tokenize
        chat_messages = item['prompt']
        tokenized = tokenizer.apply_chat_template(
            chat_messages, 
            tokenize=True, 
            add_generation_prompt=True
        )
        token_counts.append(len(tokenized))
        acc_values.append(item['reference_acc'])
        
        # 打印第一个样本的详细信息进行验证
        if idx == 0:
            print(f"  [验证] 第一个样本:")
            print(f"    tokenized 类型: {type(tokenized)}")
            print(f"    tokenized 长度 (token总数): {len(tokenized)}")
            print(f"    前10个token IDs: {tokenized[:10]}")
            print(f"    解码后前100字符: {tokenizer.decode(tokenized[:50])[:100]}...")
    
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    avg_tokens = sum(token_counts) / len(token_counts)
    median_tokens = sorted(token_counts)[len(token_counts)//2]
    
    print(f"  Token数量统计:")
    print(f"    最大值: {max_tokens}")
    print(f"    最小值: {min_tokens}")
    print(f"    平均值: {avg_tokens:.2f}")
    print(f"    中位数: {median_tokens}")
    
    # 使用matplotlib绘制图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('get_prompts_weakness_icl Token Statistics', fontsize=14, fontweight='bold')
    
    # 图1: Token数量直方图
    ax1 = axes[0, 0]
    ax1.hist(token_counts, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(avg_tokens, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_tokens:.0f}')
    ax1.axvline(median_tokens, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_tokens}')
    ax1.axvline(max_tokens, color='darkred', linestyle=':', linewidth=2, label=f'Max: {max_tokens}')
    ax1.set_xlabel('Token Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Token Count Distribution')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: ACC分布直方图
    ax2 = axes[0, 1]
    ax2.hist(acc_values, bins=10, color='seagreen', edgecolor='white', alpha=0.8, range=(0, 1))
    ax2.set_xlabel('ACC Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('ACC Value Distribution (Uniform Sampling Check)')
    ax2.set_xticks([i*0.1 for i in range(11)])
    ax2.grid(axis='y', alpha=0.3)
    
    # 图3: Token数量 vs ACC散点图
    ax3 = axes[1, 0]
    scatter = ax3.scatter(acc_values, token_counts, c=token_counts, cmap='viridis', alpha=0.6, s=20)
    ax3.set_xlabel('ACC Value')
    ax3.set_ylabel('Token Count')
    ax3.set_title('Token Count vs ACC Value')
    plt.colorbar(scatter, ax=ax3, label='Token Count')
    ax3.grid(alpha=0.3)
    
    # 图4: Token数量箱线图 (按ACC区间分组)
    ax4 = axes[1, 1]
    acc_bins_data = [[] for _ in range(10)]
    for acc, tokens in zip(acc_values, token_counts):
        bin_idx = min(int(acc * 10), 9)
        acc_bins_data[bin_idx].append(tokens)
    
    # 只绘制非空的箱线图
    non_empty_data = [(i, data) for i, data in enumerate(acc_bins_data) if data]
    if non_empty_data:
        positions = [i for i, _ in non_empty_data]
        box_data = [data for _, data in non_empty_data]
        bp = ax4.boxplot(box_data, positions=positions, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
    
    ax4.set_xlabel('ACC Bin (0-9 represents 0.0-1.0)')
    ax4.set_ylabel('Token Count')
    ax4.set_title('Token Count by ACC Bin (Boxplot)')
    ax4.set_xticks(range(10))
    ax4.set_xticklabels([f'{i/10:.1f}' for i in range(10)])
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = '/home/ycy/data1/Self-evolving-Agent/se_code_auto/prompt_weakness/token_stats.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  图表已保存至: {save_path}")
    plt.close()
    
    # 测试8: 检查数据源和ability字段
    print("\n[测试8] 数据源和ability字段检查")
    data_sources = set(item['data_source'] for item in result_large)
    abilities = set(item['ability'] for item in result_large)
    print(f"  数据源: {data_sources}")
    print(f"  能力类型: {abilities}")
    
    # 测试9: 超大数量测试
    print("\n[测试9] 超大数量测试 (num_querys=5000)")
    try:
        result_xl = get_prompts_weakness_icl(5000)
        print(f"  ✓ 返回数据条数: {len(result_xl)}")
        # 检查是否有重复（由于使用choices可能有重复）
        problems = [item['prompt'][0]['content'] for item in result_xl]
        unique_problems = set(problems)
        print(f"  唯一问题数: {len(unique_problems)} / {len(problems)}")
    except Exception as e:
        print(f"  ✗ 超大数量测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print(f"最大Token数量: {max_tokens}")
    print("=" * 60)
    
    return max_tokens


if __name__ == "__main__":
    import sys
    
    # 优先检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "test_icl":
        test_get_prompts_weakness_icl()
        sys.exit(0)
    
    # 原有的测试代码
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader
    from collections import Counter
    config = OmegaConf.load("config/challenger_trainer.yaml")
    print(f'{config=}')
    tokenizer = AutoTokenizer.from_pretrained("/home/ycy/data1/models/Qwen3-4B-Base")
    config.return_raw_chat=True
    dataset = ChallengerTopicDataset(tokenizer, config.data)
    print(f'{len(dataset)=}')

    res = dataset[random.randint(0, len(dataset)-1)]
    print(f'{res["raw_inputs"]=}')
    print(f'{res["input_ids"].shape=}')
    print(f'{res["attention_mask"].shape=}')
    print(f'{res["position_ids"].shape=}')
    print(f'{res["attention_mask"].sum()=}')
    
