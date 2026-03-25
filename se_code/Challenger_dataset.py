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

def get_prompts_topic(num_querys:int, prompt_path=None):
    
    prompt_path=os.path.join(prompt_path,'prompt4')
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




def get_prompts_Topic_AoPS(num_querys:int, prompt_path=None):
    # if not os.path.exists(topic_path):
    #     raise ValueError(f'{topic_path=} not exist')
    # print(f'{topic_path=}')
    
    prompt_path=os.path.join(prompt_path,'prompt2')
    topic_path=os.path.join(prompt_path,'all_topic_annotations.json')
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


# def get_prompts(num_querys:int, topic_path):
#     if not os.path.exists(topic_path):
#         raise ValueError(f'{topic_path=} not exist')
#     print(f'{topic_path=}')
#     with open('/root/users/ycy/data/DeepMath-103K_t2q_s128.json','r',encoding='utf-8') as f:
#         topic_qs = json.load(f)
#     prompt_path='/root/users/ycy/Self-evolving-Agent/se_code/prompt3_icl'
#     with open(os.path.join(prompt_path,'system.txt'),'r',encoding='utf-8') as f:
#         raw_system = f.read()              # 使用 $PLACEHOLDER 占位
#     with open(os.path.join(prompt_path,'user.txt'),'r',encoding='utf-8') as f:
#         raw_user = f.read()                # 使用 $PLACEHOLDER 占位
#     topics = json.load(open(os.path.join(topic_path),'r',encoding='utf-8'))
#     topics_cnt = sum(v['total_question_count'] for v in topics.values())
    
    
#     dataframe=[]
#     # ---------- 配额分配（最大余数法 + 最低覆盖1条可选） ----------
#     shares = []
#     for t, info in topics.items():
#         share = (info['total_question_count'] / topics_cnt) * num_querys
#         shares.append((t, share))
#     base_alloc = {t: int(math.floor(s)) for t,s in shares}
#     total_base = sum(base_alloc.values())
#     remain = num_querys - total_base
#     residuals = sorted(((s - math.floor(s), t) for t,s in shares), reverse=True)
#     for _, t in residuals[:remain]:
#         base_alloc[t] += 1
#     idx = 0
#     for topic, infos in topics.items():
#         #core_desc=infos['Core_desc']
#         #cue_desc = infos['Cue_desc']
#         n = base_alloc[topic]
#         out=[]
#         for _ in range(n):
#             ns = len(topic_qs[topic]) 
#             if ns < 3:
#                 continue

#             qs = random.sample(topic_qs[topic], k=3)
#             system_prompt=raw_system
#             user_prompt=raw_user.format(
#                 example_question_1=qs[0],
#                 example_question_2=qs[1],
#                 example_question_3=qs[2],
#             )
#             prompt=[
#                 {
#                     'role':'system',
#                     'content':system_prompt
#                 },
#                 {
#                     'role':'user',
#                     'content':user_prompt
#                 }
#             ]
#             out.append(
#                 {
#                     'idx': idx,
#                     'data_source':'Challenger',
#                     'topic':topic,
#                     'prompt':prompt,
#                     'ability':'math'
#                 }
#             )
#             idx += 1 
#         dataframe.extend(out)
#     random.shuffle(dataframe)
#     return dataframe


def get_prompts_weakness(num_querys: int, prompt_path=None):
    """
    从weakness_data_pool_processed.json中读取数据，使用system.txt和user.txt生成prompt
    """
    prompt_path=os.path.join(prompt_path,'prompt_weakness')
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


def get_prompts(num_querys, get_prompts_func, prompt_path):
    if get_prompts_func == "R_Zero":
        return get_prompts_R_zero(num_querys, prompt_path)
    elif get_prompts_func == "Topic":
        return get_prompts_topic(num_querys, prompt_path)
    elif get_prompts_func == "Topic_AoPS":
        return get_prompts_Topic_AoPS(num_querys, prompt_path)
    elif get_prompts_func == "weakness":
        return get_prompts_weakness(num_querys, prompt_path)
    else:
        raise ValueError(f"Invalid get_prompts_func: {get_prompts_func}")


def get_prompts_R_zero(num_querys, prompt_path=None):
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
        self.prompt_path=config.get('prompt_path','/root/users/ycy/Self-evolving-Agent/se_code')
        if self.dynamic_topics:
            self.topic_path=config.get('topics_path',os.path.join(self.prompt_path,'prompt2','all_topic_annotations.json'))
        else:
            self.topic_path=os.path.join(self.prompt_path,'prompt2','all_topic_annotations.json')
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
        self.dataframe = get_prompts(num_querys=self.num_querys, get_prompts_func=self.get_prompts_func, prompt_path=self.prompt_path)
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
if __name__ == "__main__":
    
    #pass
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader
    from collections import Counter
    config = OmegaConf.load("config/challenger_trainer.yaml")
    print(f'{config=}')
    tokenizer = AutoTokenizer.from_pretrained("/root/users/ycy/models/shares/Qwen3-4B-Base")
    config.return_raw_chat=True
    dataset = ChallengerTopicDataset(tokenizer, config.data)
    print(f'{len(dataset)=}')
    #cnt=dict(Counter([item['topic'] for item in dataset.dataframe]))
    #topics = json.load(open(os.path.join('/root/users/ycy/Self-evolving-Agent/se_code/prompt2', 'all_topic_annotations.json'),'r'))
    #topics_cnt = sum([v['total_question_count'] for k,v in topics.items()])
    #print(f'{cnt=}')
    # for k,v in cnt.items():
    #     ratio = v/len(dataset)
    #     raw_ratio=topics[k]['total_question_count']/topics_cnt
    #     print(f"{((ratio-raw_ratio) < 1e-3)=}")


    res=dataset[random.randint(0,len(dataset)-1)]
    print(f'{res["raw_inputs"]=}')
    print(f'{res["input_ids"].shape=}')
    print(f'{res["attention_mask"].shape=}')
    print(f'{res["position_ids"].shape=}')
    print(f'{res["attention_mask"].sum()=}')
    