from __future__ import annotations

import re
from typing import Any, Dict, List

from .text_utils import normalize_memory_text, normalize_space


_ITEM_RE = re.compile(
    r"#\s*Memory Item\s*\d+\s*"
    r"##\s*Title\s*(?P<title>.*?)\s*"
    r"##\s*Description\s*(?P<description>.*?)\s*"
    r"##\s*Content\s*(?P<content>.*?)(?=(?:\n#\s*Memory Item\s*\d+)|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_memory_items(raw: str) -> List[Dict[str, str]]:
    text = str(raw or "").strip()
    items: List[Dict[str, str]] = []
    for m in _ITEM_RE.finditer(text):
        title = normalize_space(m.group("title"))
        description = normalize_space(m.group("description"))
        content = normalize_space(m.group("content"))
        if not title or not content:
            continue
        items.append(
            {
                "title": title,
                "description": description,
                "content": content,
            }
        )
    if items:
        return items[:3]
    compact = normalize_space(text)
    if compact:
        return [
            {
                "title": "Recovered Memory",
                "description": "Recovered from non-standard teacher output.",
                "content": compact,
            }
        ]
    return []


def memory_item_key(item: Dict[str, Any]) -> str:
    title = normalize_memory_text(item.get("title"))
    content = normalize_memory_text(item.get("content"))
    return f"{title}||{content}"

