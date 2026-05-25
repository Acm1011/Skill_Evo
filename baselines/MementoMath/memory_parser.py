from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .text_utils import normalize_memory_text, normalize_space


def _extract_json_block(raw: str) -> Dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        frag = text[start : end + 1]
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _normalize_plan_steps(value: Any) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    if isinstance(value, list):
        for i, item in enumerate(value, start=1):
            if isinstance(item, dict):
                desc = normalize_space(
                    item.get("description") or item.get("step") or item.get("content")
                )
            else:
                desc = normalize_space(item)
            if desc:
                steps.append({"id": str(i), "description": desc})
    elif isinstance(value, str):
        lines = [
            normalize_space(x)
            for x in re.split(r"(?:\n+|\s*;\s*)", value)
            if normalize_space(x)
        ]
        for i, desc in enumerate(lines, start=1):
            steps.append({"id": str(i), "description": desc})
    return steps[:6]


def parse_case_output(raw: str, *, fallback_status: str) -> Dict[str, Any]:
    obj = _extract_json_block(raw) or {}
    plan = _normalize_plan_steps(obj.get("plan") or obj.get("plan_steps") or obj.get("steps"))
    takeaway = normalize_space(
        obj.get("takeaway") or obj.get("lesson") or obj.get("principle") or ""
    )
    case_summary = normalize_space(
        obj.get("case_summary") or obj.get("summary") or obj.get("case") or ""
    )
    tags = obj.get("tags")
    if not isinstance(tags, list):
        tags = []
    clean_tags = [normalize_space(x) for x in tags if normalize_space(x)][:8]

    if not plan:
        text = normalize_space(raw)
        if text:
            plan = [{"id": "1", "description": text[:500]}]
    if not takeaway and plan:
        takeaway = plan[0]["description"]
    return {
        "plan_steps": plan,
        "takeaway": takeaway,
        "case_summary": case_summary or f"{fallback_status} math case",
        "tags": clean_tags,
    }


def plan_signature(plan_steps: List[Dict[str, Any]]) -> str:
    parts = [normalize_memory_text(x.get("description")) for x in plan_steps if isinstance(x, dict)]
    return "||".join(x for x in parts if x)
