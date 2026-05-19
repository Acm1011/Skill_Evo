from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import httpx


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_success_instruction() -> str:
    return (_prompt_dir() / "success_memory.txt").read_text(encoding="utf-8")


def load_failure_instruction() -> str:
    return (_prompt_dir() / "failure_memory.txt").read_text(encoding="utf-8")


def teacher_env() -> Dict[str, str]:
    return {
        "base_url": os.environ.get("RBM_TEACHER_BASE_URL", "").strip().rstrip("/"),
        "api_key": os.environ.get("RBM_TEACHER_API_KEY", "").strip(),
        "model": os.environ.get("RBM_TEACHER_MODEL", "").strip(),
    }


def chat_complete(
    messages: List[Dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 600.0,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    if not base_url:
        raise ValueError("RBM_TEACHER_BASE_URL is empty")
    if not model:
        raise ValueError("RBM_TEACHER_MODEL is empty")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in teacher response: {data.keys()}")
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    return content if isinstance(content, str) else str(content)

