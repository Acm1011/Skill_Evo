"""DeepMath answer extraction and grading utilities."""
from __future__ import annotations

import re
from typing import Any

_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_all_boxed_content(text: str) -> list[str]:
    return [m.group(1).strip() for m in _BOXED_RE.finditer(text or "") if m.group(1).strip()]


def extract_answer(text: str) -> str:
    if not text:
        return ""
    tag = _ANSWER_TAG_RE.search(text)
    if tag:
        return tag.group(1).strip()
    boxed = extract_all_boxed_content(text)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].strip().rstrip(".") if lines else ""


def _normalize(text: Any) -> str:
    return str(text or "").strip().lower().rstrip(".")


def evaluate_response(response: str, ground_truth: str) -> dict:
    pred = extract_answer(response)
    gt = str(ground_truth or "").strip()
    hard = 0
    if pred and gt:
        try:
            from mathruler.grader import grade_answer
            hard = int(bool(grade_answer(pred, gt)))
        except Exception:
            hard = int(_normalize(pred) == _normalize(gt))
    return {
        "predicted_answer": pred,
        "ground_truth": gt,
        "hard": hard,
        "soft": float(hard),
    }
