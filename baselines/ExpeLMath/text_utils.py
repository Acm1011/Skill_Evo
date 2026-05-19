from __future__ import annotations

import hashlib
import re
from typing import Any, Optional


def topic_slug(topic: Optional[str]) -> str:
    if topic is None or not str(topic).strip():
        return "unknown"
    s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", str(topic).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] if s else "unknown"


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_memory_text(text: Any) -> str:
    s = normalize_space(text).lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    return normalize_space(s)


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

