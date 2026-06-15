"""DeepMath task dataloader for SkillOpt."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Optional

from skillopt.datasets.base import BatchSpec, SplitDataLoader


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL line {line_no} in {path}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_problem(row: dict) -> Optional[str]:
    problem = _first_text(row.get("problem"), row.get("question"), row.get("raw_question"))
    if problem:
        return problem
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    problem = _first_text(extra.get("problem"), extra.get("question"))
    if problem:
        return problem
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = _first_text(msg.get("content"))
                if content:
                    return content
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def extract_ground_truth(row: dict) -> str:
    gt = _first_text(row.get("ground_truth"), row.get("gt"))
    if gt:
        return gt
    reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
    value = reward_model.get("ground_truth")
    if isinstance(value, list) and value:
        gt = _first_text(value[0])
    else:
        gt = _first_text(value)
    if gt:
        return gt
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return _first_text(extra.get("answer"), extra.get("solution"))


def extract_topic(row: dict) -> str:
    topic = _first_text(row.get("topic"), row.get("topic_key"))
    if topic:
        return topic
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return _first_text(extra.get("topic"), extra.get("topic_key")) or "general_math"


def normalize_item(row: dict, row_idx: int, source_path: str) -> dict | None:
    problem = extract_problem(row)
    if not problem:
        return None
    gt = extract_ground_truth(row)
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    item_id = row.get("id") or row.get("idx") or extra.get("idx") or row.get("line_idx") or row_idx
    topic = extract_topic(row)
    item = {
        "id": str(item_id),
        "question": problem,
        "problem": problem,
        "ground_truth": gt,
        "answer": gt,
        "topic": topic,
        "task_type": str(topic or "general_math"),
        "difficulty": extra.get("difficulty") or row.get("difficulty"),
        "source_path": source_path,
        "source_row_idx": row_idx,
        "raw_row": row,
    }
    if row.get("student_response"):
        item["source_student_response"] = row.get("student_response")
    if row.get("is_correct") is not None:
        item["source_is_correct"] = row.get("is_correct")
    return item


def load_items(data_path: str) -> list[dict]:
    if not data_path:
        raise ValueError("DeepMath requires data_path to point to a DeepMath/trajectory JSONL file.")
    if os.path.isdir(data_path):
        candidates = sorted(os.path.join(data_path, name) for name in os.listdir(data_path) if name.endswith(".jsonl"))
        if len(candidates) != 1:
            raise ValueError(f"DeepMath data_path directory must contain exactly one .jsonl file: {data_path}")
        data_path = candidates[0]
    rows = _read_jsonl(data_path)
    items: list[dict] = []
    for i, row in enumerate(rows):
        item = normalize_item(row, i, data_path)
        if item is not None:
            items.append(item)
    if not items:
        raise ValueError(f"No valid DeepMath items loaded from {data_path}")
    return items


class DeepMathDataLoader(SplitDataLoader):
    """DeepMath dataloader with deterministic train/selection/test batches."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._task_types: list[str] = []

    def load_raw_items(self, data_path: str) -> list[dict]:
        return load_items(data_path)

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        all_items = self.train_items + self.val_items + self.test_items
        self._task_types = sorted({str(item.get("task_type") or "general_math") for item in all_items}) or ["general_math"]

    def get_task_types(self) -> list[str]:
        return list(self._task_types)

    def plan_train_epoch(self, *, epoch: int, steps_per_epoch: int, accumulation: int, batch_size: int, seed: int, **kwargs) -> list[BatchSpec]:
        rng = random.Random(seed + epoch * 1000)
        items = list(self.train_items)
        rng.shuffle(items)
        total_batches = steps_per_epoch * accumulation
        batches: list[BatchSpec] = []
        cursor = 0
        for batch_idx in range(total_batches):
            batch_seed = seed + epoch * 1000 + batch_idx + 1
            batch_items = items[cursor: cursor + batch_size]
            cursor += len(batch_items)
            if not batch_items and items:
                batch_items = list(items)
                random.Random(batch_seed).shuffle(batch_items)
                batch_items = batch_items[:batch_size]
            batches.append(BatchSpec(phase="train", split="train", seed=batch_seed, batch_size=len(batch_items), payload=batch_items))
        return batches
