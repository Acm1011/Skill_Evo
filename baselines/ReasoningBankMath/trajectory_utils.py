from __future__ import annotations

from typing import Any, Dict, List, Optional

from baselines.SkillRL.student_rollout import extract_ground_truth, extract_meta, extract_problem

from .text_utils import topic_slug


def normalize_trajectory_row(row: Dict[str, Any], line_idx: int) -> Optional[Dict[str, Any]]:
    if "problem" in row and "student_response" in row:
        problem = str(row.get("problem") or "").strip()
        response = str(row.get("student_response") or "").strip()
        if not problem or not response:
            return None
        topic = row.get("topic")
        return {
            "idx": row.get("idx", line_idx),
            "line_idx": row.get("line_idx", line_idx),
            "problem": problem,
            "topic": topic,
            "topic_key": row.get("topic_key") or topic_slug(topic),
            "difficulty": row.get("difficulty"),
            "student_response": response,
            "is_correct": row.get("is_correct"),
            "ground_truth": row.get("ground_truth"),
        }

    problem = extract_problem(row)
    if not problem:
        return None
    meta = extract_meta(row, line_idx)
    response = _extract_response(row)
    if not response:
        return None
    gt = extract_ground_truth(row)
    return {
        "idx": meta["idx"],
        "line_idx": line_idx,
        "problem": problem,
        "topic": meta.get("topic"),
        "topic_key": topic_slug(meta.get("topic")),
        "difficulty": meta.get("difficulty"),
        "student_response": response,
        "is_correct": row.get("is_correct"),
        "ground_truth": gt,
    }


def _extract_response(row: Dict[str, Any]) -> str:
    for key in ("student_response", "response", "completion", "output"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    responses = row.get("responses")
    if isinstance(responses, list):
        for item in responses:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def format_math_trajectory(problem: str, student_response: str, ground_truth: Optional[str]) -> str:
    parts: List[str] = [
        "<problem>",
        str(problem).strip(),
        "</problem>",
        "<solution>",
        str(student_response).strip(),
        "</solution>",
    ]
    gt = str(ground_truth or "").strip()
    if gt:
        parts.extend(["<ground_truth>", gt, "</ground_truth>"])
    return "\n".join(parts)

