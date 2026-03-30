import logging
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
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


class SynthesizerDataset(Dataset):
    """
    从 solver_offline_driver 产出的 merged parquet/jsonl 中加载数据，
    每条样本已经包含 prompt（skill_generation prompt）、extra_info.raw_q_info、
    extra_info.random_q_info 等字段。

    训练时 Synthesizer 模型的输入是 skill_generation prompt，
    输出是 JSON 格式的 skill。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = config.get("max_prompt_length", 4096)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)        
        train_files = config.get("train_files", None)
        if train_files is None:
            raise ValueError("data.train_files must be set for SynthesizerDataset")
        if isinstance(train_files, str):
            train_files = [train_files]

        self.dataframe = self._load_files(list(train_files))
        logger.info("SynthesizerDataset loaded %d samples", len(self.dataframe))

    def _load_files(self, paths):
        rows = []
        for p in paths:
            p = os.path.abspath(p)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"SynthesizerDataset: file not found: {p}")
            if p.endswith(".parquet") or p.endswith(".pq"):
                df = pd.read_parquet(p)
                rows.extend(df.to_dict(orient="records"))
            elif p.endswith(".jsonl"):
                df = pd.read_json(p, lines=True)
                rows.extend(df.to_dict(orient="records"))
            elif p.endswith(".json"):
                import json
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    rows.extend(data)
                else:
                    raise ValueError(f"JSON must be array: {p}")
            else:
                raise ValueError(f"Unsupported file: {p}")
        return rows

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict: dict = dict(self.dataframe[item])
        messages = row_dict.pop("prompt")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        model_inputs = {}
        raw_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )

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
