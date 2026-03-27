#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单条 skill 的数据模型，与 skills.jsonl 字段对齐，并附带 utility。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# JSON 键名与 skill_induction 产出的 skills.jsonl 一致
JSON_KEY_SKILL_NAME = "skill name"
JSON_KEY_PROBLEM_TYPE = "problem type"
JSON_KEY_KEY_INSIGHT = "key insight"
JSON_KEY_METHOD = "method"
JSON_KEY_SKILL_FROM = "skill_from"
JSON_KEY_ID = "id"
JSON_KEY_PROBLEM = "problem"
JSON_KEY_SAMPLED_ROLLOUT_INDICES = "sampled_rollout_indices"
JSON_KEY_PARSE_ERROR = "parse_error"
JSON_KEY_RAW_MODEL_OUTPUT = "raw_model_output"
JSON_KEY_UTILITY = "utility"


@dataclass
class SkillItem:
    skill_name: str
    problem_type: str
    key_insight: str
    method: str
    skill_from: str
    id: str
    problem: str
    sampled_rollout_indices: list[int] = field(default_factory=list)
    parse_error: str | None = None
    raw_model_output: str | None = None
    utility: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        """序列化为与 skills.jsonl 一致的键名；utility 作为扩展字段。"""
        return {
            JSON_KEY_SKILL_NAME: self.skill_name,
            JSON_KEY_PROBLEM_TYPE: self.problem_type,
            JSON_KEY_KEY_INSIGHT: self.key_insight,
            JSON_KEY_METHOD: self.method,
            JSON_KEY_SKILL_FROM: self.skill_from,
            JSON_KEY_ID: self.id,
            JSON_KEY_PROBLEM: self.problem,
            JSON_KEY_SAMPLED_ROLLOUT_INDICES: list(self.sampled_rollout_indices),
            JSON_KEY_PARSE_ERROR: self.parse_error,
            JSON_KEY_RAW_MODEL_OUTPUT: self.raw_model_output,
            JSON_KEY_UTILITY: self.utility,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> SkillItem:
        """从 jsonl 行反序列化；缺失 utility 时默认为 0.0。"""
        indices = d.get(JSON_KEY_SAMPLED_ROLLOUT_INDICES) or []
        if not isinstance(indices, list):
            indices = []
        else:
            indices = [int(x) for x in indices]
        util = d.get(JSON_KEY_UTILITY, 0.0)
        if util is None:
            util = 0.0
        else:
            util = float(util)
        return cls(
            skill_name=str(d.get(JSON_KEY_SKILL_NAME, "")),
            problem_type=str(d.get(JSON_KEY_PROBLEM_TYPE, "")),
            key_insight=str(d.get(JSON_KEY_KEY_INSIGHT, "")),
            method=str(d.get(JSON_KEY_METHOD, "")),
            skill_from=str(d.get(JSON_KEY_SKILL_FROM, "")),
            id=str(d.get(JSON_KEY_ID, "")),
            problem=str(d.get(JSON_KEY_PROBLEM, "")),
            sampled_rollout_indices=indices,
            parse_error=d.get(JSON_KEY_PARSE_ERROR),
            raw_model_output=d.get(JSON_KEY_RAW_MODEL_OUTPUT),
            utility=util,
        )

    @classmethod
    def from_jsonl_line(cls, line: str) -> SkillItem:
        line = line.strip()
        if not line:
            raise ValueError("empty jsonl line")
        return cls.from_json_dict(json.loads(line))

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_json_dict(), ensure_ascii=False)
