#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单条 skill 的数据模型，与 skills.jsonl 字段对齐，并附带 utility。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# JSON 键名与 skill_induction 产出的 skills.jsonl 一致
JSON_KEY_SKILL_NAME = "skill name"
JSON_KEY_PROBLEM_TYPE = "problem type"
JSON_KEY_KEY_INSIGHT = "key insight"
JSON_KEY_METHOD = "method"
JSON_KEY_SKILL_FROM = "skill_from"
JSON_KEY_ID = "id"
JSON_KEY_PROBLEM = "problem"
JSON_KEY_UTILITY = "utility"
JSON_KEY_USAGE_SUCCESS = "skill_usage_success"
JSON_KEY_USAGE_FAILURE = "skill_usage_failure"
# 外部传入时允许使用 reward 作为 utility 的别名
JSON_KEY_REWARD_ALIAS = "reward"


@dataclass
class SkillItem:
    skill_name: str
    problem_type: str
    key_insight: str
    method: str
    skill_from: str
    id: str
    problem: str
    utility: float = 0.0
    skill_usage_success: int = 0
    skill_usage_failure: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        """序列化为与 skills.jsonl 一致的键名；utility / usage 作为扩展字段。"""
        return {
            JSON_KEY_SKILL_NAME: self.skill_name,
            JSON_KEY_PROBLEM_TYPE: self.problem_type,
            JSON_KEY_KEY_INSIGHT: self.key_insight,
            JSON_KEY_METHOD: self.method,
            JSON_KEY_SKILL_FROM: self.skill_from,
            JSON_KEY_ID: self.id,
            JSON_KEY_PROBLEM: self.problem,
            JSON_KEY_UTILITY: self.utility,
            JSON_KEY_USAGE_SUCCESS: self.skill_usage_success,
            JSON_KEY_USAGE_FAILURE: self.skill_usage_failure,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any], *, assigned_id: str = "") -> SkillItem:
        """从 jsonl 行或外部 payload 反序列化。

        - ``utility`` 优先；若缺失则回退到 ``reward`` 字段（外部传入时的别名）；仍缺失默认 0.0。
        - ``assigned_id`` 非空时覆盖 d 中的 id 字段，用于服务端自动分配 id 的场景。
        - ``skill_usage_success`` / ``skill_usage_failure`` 缺失时默认为 0。
        """
        util_raw = d.get(JSON_KEY_UTILITY)
        if util_raw is None:
            util_raw = d.get(JSON_KEY_REWARD_ALIAS, 0.0)
        try:
            util = float(util_raw) if util_raw is not None else 0.0
        except (TypeError, ValueError):
            util = 0.0

        try:
            usage_success = int(d.get(JSON_KEY_USAGE_SUCCESS, 0) or 0)
        except (TypeError, ValueError):
            usage_success = 0
        try:
            usage_failure = int(d.get(JSON_KEY_USAGE_FAILURE, 0) or 0)
        except (TypeError, ValueError):
            usage_failure = 0

        raw_id = assigned_id if assigned_id else str(d.get(JSON_KEY_ID, ""))
        return cls(
            skill_name=str(d.get(JSON_KEY_SKILL_NAME, "")),
            problem_type=str(d.get(JSON_KEY_PROBLEM_TYPE, "")),
            key_insight=str(d.get(JSON_KEY_KEY_INSIGHT, "")),
            method=str(d.get(JSON_KEY_METHOD, "")),
            skill_from=str(d.get(JSON_KEY_SKILL_FROM, "")),
            id=raw_id,
            problem=str(d.get(JSON_KEY_PROBLEM, "")),
            utility=util,
            skill_usage_success=usage_success,
            skill_usage_failure=usage_failure,
        )

    @classmethod
    def from_jsonl_line(cls, line: str) -> SkillItem:
        line = line.strip()
        if not line:
            raise ValueError("empty jsonl line")
        return cls.from_json_dict(json.loads(line))

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_json_dict(), ensure_ascii=False)
