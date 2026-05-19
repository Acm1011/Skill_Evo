from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional


SUBTASK_NAMES = [
    "understand_problem",
    "choose_strategy",
    "derive_solution",
    "check_constraints",
    "finalize_answer",
]

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def build_task_key(question: str, topic: Optional[str]) -> str:
    q = (question or "").strip()
    t = (topic or "").strip()
    return f"[{t}] {q}" if t else q


def extract_answer_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def grade_math_answer(answer: Optional[str], ground_truth: Optional[str]) -> float:
    pred = (answer or "").strip()
    gt = (ground_truth or "").strip()
    if not pred or not gt:
        return 0.0
    try:
        from verl.utils.reward_score.prime_math import compute_score as prime_math_compute_score

        return float(bool(prime_math_compute_score(pred, gt)))
    except Exception:
        return float(pred.lower().rstrip(".") == gt.lower().rstrip("."))


def _extract_json_block(text: str) -> str:
    clean = text.strip()
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)
    start_idx = clean.find("{")
    end_idx = clean.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("no JSON object found")
    return clean[start_idx : end_idx + 1]


def parse_reflection_json(text: str) -> Dict[str, Any]:
    json_str = _extract_json_block(text)
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError("reflection payload must be an object")
    return data


def _normalize_status(status: Any) -> str:
    s = str(status or "").strip().lower()
    return "completed" if s == "completed" else "incomplete"


def normalize_reflection_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_subtasks = payload.get("subtasks")
    by_name: Dict[str, Dict[str, str]] = {}
    if isinstance(raw_subtasks, Iterable) and not isinstance(raw_subtasks, (str, bytes, dict)):
        for item in raw_subtasks:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            by_name[name] = {
                "name": name,
                "description": str(item.get("description") or "").strip(),
                "status": _normalize_status(item.get("status")),
            }

    subtasks: List[Dict[str, str]] = []
    for name in SUBTASK_NAMES:
        existing = by_name.get(name)
        if existing is None:
            subtasks.append({"name": name, "description": "", "status": "incomplete"})
        else:
            subtasks.append(existing)

    task_success = payload.get("task_success", False)
    if isinstance(task_success, str):
        task_success = task_success.strip().lower() in {"true", "1", "yes"}
    else:
        task_success = bool(task_success)

    return {
        "subtasks": subtasks,
        "task_success": task_success,
        "action_lesson": str(payload.get("action_lesson") or "").strip(),
        "reasoning_lesson": str(payload.get("reasoning_lesson") or "").strip(),
    }


def compute_subtask_potential(payload: Dict[str, Any]) -> float:
    subtasks = payload.get("subtasks") or []
    total = 0
    completed = 0
    for item in subtasks:
        if not isinstance(item, dict):
            continue
        total += 1
        if _normalize_status(item.get("status")) == "completed":
            completed += 1
    if total <= 0:
        return 0.0
    return completed / total


def build_lesson_text(payload: Dict[str, Any]) -> str:
    lessons: List[str] = []
    action = str(payload.get("action_lesson") or "").strip()
    reasoning = str(payload.get("reasoning_lesson") or "").strip()
    if action:
        lessons.append(f"Action Insight: {action}")
    if reasoning:
        lessons.append(f"Reasoning Insight: {reasoning}")
    return " | ".join(lessons)
