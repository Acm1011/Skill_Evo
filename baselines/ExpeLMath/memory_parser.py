from __future__ import annotations

import re
from typing import Any, Dict, List

from .text_utils import normalize_memory_text, normalize_space


_RAW_RULE_RE = re.compile(
    r"#\s*Raw Rule\s*(?P<raw_rule>.*?)(?=(?:\n#\s*Memory Item\s*\d+)|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_ITEM_RE = re.compile(
    r"#\s*Memory Item\s*\d+\s*"
    r"##\s*Title\s*(?P<title>.*?)\s*"
    r"##\s*Description\s*(?P<description>.*?)\s*"
    r"##\s*Content\s*(?P<content>.*?)\s*"
    r"(?:##\s*Type\s*(?P<memory_type>.*?))?"
    r"(?=(?:\n#\s*Memory Item\s*\d+)|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_teacher_output(raw: str, *, default_memory_type: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    items: List[Dict[str, str]] = []
    for match in _ITEM_RE.finditer(text):
        title = normalize_space(match.group("title"))
        description = normalize_space(match.group("description"))
        content = normalize_space(match.group("content"))
        memory_type = normalize_space(match.group("memory_type")) or default_memory_type
        if not title or not content:
            continue
        items.append(
            {
                "title": title,
                "description": description,
                "content": content,
                "memory_type": memory_type,
            }
        )
    if not items:
        compact = normalize_space(text)
        if compact:
            items = [
                {
                    "title": "Recovered Memory",
                    "description": "Recovered from non-standard teacher output.",
                    "content": compact,
                    "memory_type": default_memory_type,
                }
            ]
    raw_rule_match = _RAW_RULE_RE.search(text)
    raw_rule = normalize_space(raw_rule_match.group("raw_rule")) if raw_rule_match else ""
    if not raw_rule and items:
        first = items[0]
        raw_rule = normalize_space(f"{first.get('title', '')}: {first.get('content', '')}")
    return {
        "raw_rule": raw_rule,
        "memory_items": items[:3],
    }


def memory_item_key(item: Dict[str, Any]) -> str:
    title = normalize_memory_text(item.get("title"))
    content = normalize_memory_text(item.get("content"))
    memory_type = normalize_memory_text(item.get("memory_type"))
    return f"{memory_type}||{title}||{content}"

