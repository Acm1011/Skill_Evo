#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从模型输出抽取 \\boxed{} 并与金标比对（供 rollout / skill_induction 使用）。"""
from __future__ import annotations

import re
from typing import Any, Optional


def extract_boxed(text: str) -> Optional[str]:
    """
    提取文本中最后一个 \\boxed{...} 的内容；大括号可嵌套（深度计数）。
    """
    if not text:
        return None
    idx = text.rfind(r"\boxed")
    if idx < 0:
        return None
    j = idx + len(r"\boxed")
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text) or text[j] != "{":
        return None
    depth = 0
    start = j
    i = j
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
        i += 1
    return None


def normalize_answer(s: Optional[str]) -> str:
    """去空白、逗号、常见 LaTeX 包裹，便于比较。"""
    if s is None:
        return ""
    t = str(s).strip()
    t = re.sub(r"\s+", "", t)
    t = t.replace(",", "")
    if len(t) >= 2 and t.startswith("$") and t.endswith("$"):
        t = t[1:-1]
    return t.strip()


def answers_equivalent(pred: str, gold: str) -> bool:
    """先尝试数值等价，再回退字符串比较。"""
    pn, gn = normalize_answer(pred), normalize_answer(gold)
    if pn == gn:
        return True
    try:
        if float(pn) == float(gn):
            return True
    except ValueError:
        pass
    try:
        if int(pn) == int(gn):
            return True
    except ValueError:
        pass
    return False


def is_rollout_correct(rollout_text: str, gold: Any) -> bool:
    """
    若无法抽取 \\boxed{} 或与金标不等价则判错。
    gold 可为 int/str 等，会先 str 再规范化比较。
    """
    inner = extract_boxed(rollout_text)
    if inner is None:
        return False
    gold_s = str(gold).strip() if gold is not None else ""
    return answers_equivalent(inner, gold_s)


class AnswerEvaluator:
    """可选封装，与纯函数等价。"""

    extract_boxed = staticmethod(extract_boxed)
    normalize_answer = staticmethod(normalize_answer)
    answers_equivalent = staticmethod(answers_equivalent)
    is_rollout_correct = staticmethod(is_rollout_correct)
