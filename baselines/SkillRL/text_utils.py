"""Lightweight text helpers (avoid importing heavy skill_src.utils)."""
from __future__ import annotations

import re
from typing import List, Optional


def extract_all_boxed_content(text: str) -> List[str]:
    """Extract contents of \\boxed{...} with brace-depth matching."""
    boxed_contents: List[str] = []
    search_start = 0
    needle = r"\boxed{"
    while True:
        start_pos = text.find(needle, search_start)
        if start_pos == -1:
            break
        depth = 0
        content = text[start_pos + len(needle) :]
        end_pos = -1
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == -1:
                end_pos = i
                break
        if end_pos != -1:
            boxed_contents.append(content[:end_pos].strip())
            search_start = start_pos + len(needle) + end_pos + 1
        else:
            search_start = start_pos + len(needle)
    return boxed_contents


def topic_slug(topic: Optional[str]) -> str:
    """Normalize DeepMath topic string to a safe dict key."""
    if topic is None or not str(topic).strip():
        return "unknown"
    s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", str(topic).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:120] if s else "unknown")


def parse_json_object(raw: str) -> Optional[dict]:
    """Extract first top-level JSON object from model output."""
    import json

    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None
