from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from agent_system.environments.math_utils import build_task_key, extract_answer_text, grade_math_answer


@dataclass
class MathTask:
    question: str
    ground_truth: str
    data_source: str
    topic: str
    index: int
    task_key: str


def _normalize_reset_kwargs(kwargs: Any) -> List[Dict[str, Any]]:
    if kwargs is None:
        return []
    if isinstance(kwargs, np.ndarray):
        kwargs = kwargs.tolist()
    if isinstance(kwargs, list):
        out: List[Dict[str, Any]] = []
        for item in kwargs:
            if isinstance(item, dict):
                out.append(dict(item))
        return out
    if isinstance(kwargs, dict):
        if "question" in kwargs:
            return [dict(kwargs)]
        keys = [k for k in kwargs.keys() if k != "is_train"]
        if not keys:
            return []
        first_val = kwargs[keys[0]]
        if isinstance(first_val, np.ndarray):
            first_val = first_val.tolist()
        if not isinstance(first_val, list):
            return [dict(kwargs)]
        size = len(first_val)
        out = []
        for idx in range(size):
            row = {}
            for key in keys:
                value = kwargs[key]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                row[key] = value[idx]
            if "is_train" in kwargs:
                row["is_train"] = kwargs["is_train"]
            out.append(row)
        return out
    raise TypeError(f"Unsupported kwargs type: {type(kwargs)!r}")


class MathSingleTurnEnvBatch:
    def __init__(self, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True, env_config=None):
        self.seed = seed
        self.env_num = env_num
        self.group_n = group_n
        self.capacity = max(1, env_num * max(group_n, 1))
        self.is_train = is_train
        self.env_config = env_config
        self._tasks: List[MathTask] = []

    def reset(self, kwargs=None):
        rows = _normalize_reset_kwargs(kwargs)
        self._tasks = []
        obs_list: List[str] = []
        info_list: List[Dict[str, Any]] = []
        for line_idx, row in enumerate(rows):
            question = str(row.get("question") or "").strip()
            ground_truth = str(row.get("ground_truth") or "").strip()
            data_source = str(row.get("data_source") or "math").strip()
            topic = str(row.get("topic") or "").strip()
            index = int(row.get("index", line_idx))
            task_key = build_task_key(question, topic)
            task = MathTask(
                question=question,
                ground_truth=ground_truth,
                data_source=data_source,
                topic=topic,
                index=index,
                task_key=task_key,
            )
            self._tasks.append(task)
            obs_list.append(question)
            info_list.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "topic": topic,
                    "index": index,
                    "task_key": task_key,
                    "won": False,
                    "task_score": 0.0,
                    "format_ok": False,
                    "pred_answer": "",
                }
            )
        return obs_list, info_list

    def step(self, actions: Sequence[str]):
        if len(actions) != len(self._tasks):
            raise ValueError(f"Expected {len(self._tasks)} actions, got {len(actions)}")

        obs_list: List[str] = []
        reward_list: List[float] = []
        done_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        for task, raw_action in zip(self._tasks, actions):
            action_text = str(raw_action or "")
            pred_answer = extract_answer_text(action_text)
            format_ok = pred_answer is not None
            reward = grade_math_answer(pred_answer, task.ground_truth) if format_ok else 0.0
            won = bool(reward >= 1.0)
            obs_list.append(task.question)
            reward_list.append(float(reward))
            done_list.append(True)
            info_list.append(
                {
                    "question": task.question,
                    "ground_truth": task.ground_truth,
                    "data_source": task.data_source,
                    "topic": task.topic,
                    "index": task.index,
                    "task_key": task.task_key,
                    "won": won,
                    "task_score": float(reward),
                    "format_ok": format_ok,
                    "pred_answer": pred_answer or "",
                    "raw_response": action_text,
                }
            )

        return obs_list, reward_list, done_list, info_list

    def close(self):
        self._tasks = []


def build_math_single_turn_envs(seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True, env_config=None):
    return MathSingleTurnEnvBatch(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )
