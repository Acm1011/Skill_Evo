"""Minimal DeepMath jsonl slice + rollout URL resolution (no heavy skill_src deps)."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence


def resolve_rollout_server_urls(
    cli_urls: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Priority: cli_urls > SE_ROLLOUT_SERVER_URLS > SE_ROLLOUT_BASE_PORT + SE_ROLLOUT_N_SERVERS + SE_ROLLOUT_HOST
    (same semantics as skill_src.rollout_deepmath).
    """
    if cli_urls:
        out = [u.strip() for u in cli_urls if u.strip()]
        if out:
            return out
    env_urls = os.environ.get("SE_ROLLOUT_SERVER_URLS", "").strip()
    if env_urls:
        parts = re.split(r"[\s,]+", env_urls)
        return [p for p in parts if p]
    host = (os.environ.get("SE_ROLLOUT_HOST", "127.0.0.1").strip() or "127.0.0.1")
    base_s = os.environ.get(
        "SE_ROLLOUT_BASE_PORT",
        os.environ.get("ROLLOUT_BASE_PORT", "8760"),
    ).strip()
    n_s = os.environ.get("SE_ROLLOUT_N_SERVERS", "").strip()
    if not n_s:
        raise ValueError(
            "未提供 --server-urls，且环境中无 SE_ROLLOUT_SERVER_URLS。"
            "请设置 SE_ROLLOUT_SERVER_URLS，或设置 SE_ROLLOUT_N_SERVERS，"
            "并设置 SE_ROLLOUT_BASE_PORT（或 ROLLOUT_BASE_PORT）。"
        )
    base_port = int(base_s)
    n = int(n_s)
    return [f"http://{host}:{base_port + i}" for i in range(n)]


def load_records_in_range(data_file: str, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
    """Load records [start_idx, end_idx) from a jsonl file."""
    records: List[Dict[str, Any]] = []
    with open(data_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
