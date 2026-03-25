import copy
import logging
import os
import random
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


logger = logging.getLogger(__name__)
"""
PROMPT_DICT = {
        "algebra": {
            "topic": "代数",
            "description": "线性代数、多项式、方程求解、函数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert algebra problem setter specializing in advanced algebraic concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial algebra problem. "
                        "The problem should involve concepts like polynomial manipulation, equation solving, function analysis, "
                        "linear algebra, or advanced algebraic structures. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging algebra question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "geometry": {
            "topic": "几何",
            "description": "平面几何、立体几何、解析几何、三角学等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert geometry problem setter specializing in advanced geometric concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial geometry problem. "
                        "The problem should involve concepts like plane geometry, solid geometry, coordinate geometry, "
                        "trigonometry, or advanced geometric constructions. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging geometry question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "number_theory": {
            "topic": "数论",
            "description": "整数性质、素数、同余、数论函数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert number theory problem setter specializing in advanced number-theoretic concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial number theory problem. "
                        "The problem should involve concepts like divisibility, prime numbers, modular arithmetic, "
                        "Diophantine equations, or advanced number-theoretic functions. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging number theory question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "combinatorics": {
            "topic": "组合数学",
            "description": "排列组合、图论、计数原理、生成函数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert combinatorics problem setter specializing in advanced combinatorial concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial combinatorics problem. "
                        "The problem should involve concepts like permutations, combinations, graph theory, "
                        "counting principles, generating functions, or advanced combinatorial structures. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging combinatorics question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "probability": {
            "topic": "概率论",
            "description": "概率计算、条件概率、随机变量、概率分布等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert probability problem setter specializing in advanced probabilistic concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial probability problem. "
                        "The problem should involve concepts like probability calculations, conditional probability, "
                        "random variables, probability distributions, or advanced probabilistic reasoning. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging probability question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "statistics": {
            "topic": "统计学",
            "description": "描述统计、推断统计、假设检验、回归分析等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert statistics problem setter specializing in advanced statistical concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial statistics problem. "
                        "The problem should involve concepts like descriptive statistics, inferential statistics, "
                        "hypothesis testing, regression analysis, or advanced statistical methods. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging statistics question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "calculus": {
            "topic": "微积分",
            "description": "极限、导数、积分、微分方程等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert calculus problem setter specializing in advanced calculus concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial calculus problem. "
                        "The problem should involve concepts like limits, derivatives, integrals, "
                        "differential equations, or advanced calculus applications. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging calculus question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "trigonometry": {
            "topic": "三角学",
            "description": "三角函数、三角恒等式、三角方程、反三角函数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert trigonometry problem setter specializing in advanced trigonometric concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial trigonometry problem. "
                        "The problem should involve concepts like trigonometric functions, trigonometric identities, "
                        "trigonometric equations, inverse trigonometric functions, or advanced trigonometric applications. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging trigonometry question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "complex_numbers": {
            "topic": "复数",
            "description": "复数运算、复平面、复数方程、欧拉公式等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert complex numbers problem setter specializing in advanced complex analysis concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial complex numbers problem. "
                        "The problem should involve concepts like complex arithmetic, complex plane geometry, "
                        "complex equations, Euler's formula, or advanced complex analysis. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging complex numbers question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "sequences_series": {
            "topic": "数列与级数",
            "description": "等差数列、等比数列、数列极限、无穷级数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert sequences and series problem setter specializing in advanced sequence and series concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial sequences and series problem. "
                        "The problem should involve concepts like arithmetic sequences, geometric sequences, "
                        "sequence limits, infinite series, convergence tests, or advanced series analysis. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
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
                        "Generate one new, challenging sequences and series question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        }
    }
"""
PROMPT_DICT={
    "R-Zero": {
            "topic": "raw_math",
            "description": "R-Zero原始的prompt",
            "chat": [
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
}
}
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
        
        self.config = config
        self.num_querys = config.get("num_querys", 1000)
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.max_prompt_length = config.get("max_prompt_length", 1024)
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
        self.dataframe = [
            {
                'data_source': 'challenger',
                'prompt': PROMPT_DICT[topic]['chat'],
                'topic': PROMPT_DICT[topic]['topic'],
                'ability':'math',
            }
            for _ in range(self.num_querys)
            for topic in PROMPT_DICT.keys()
        ]
        # 打乱数据集顺序
        random.shuffle(self.dataframe)
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
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader
    config = OmegaConf.load("config/challenger_trainer.yaml")
    tokenizer = AutoTokenizer.from_pretrained("/root/users/ycy/models/shares/Qwen3-4B-Base")
    config.return_raw_chat=True
    dataset = ChallengerTopicDataset(tokenizer, config)
    print(f'{len(dataset)=}')
    res=dataset[0]
    print(f'{res["raw_inputs"]=}')
    print(f'{res["input_ids"].shape=}')
    print(f'{res["attention_mask"].shape=}')
    print(f'{res["position_ids"].shape=}')
    