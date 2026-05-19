from __future__ import annotations

import os
import random
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
        "rollout_urls": os.environ.get("SE_ROLLOUT_SERVER_URLS", "").strip(),
        "rollout_base_port": (
            os.environ.get("RBM_ROLLOUT_BASE_PORT", "").strip()
            or os.environ.get("SE_ROLLOUT_BASE_PORT", "").strip()
        ),
        "rollout_n_servers": (
            os.environ.get("RBM_ROLLOUT_N_SERVERS", "").strip()
            or os.environ.get("SE_ROLLOUT_N_SERVERS", "").strip()
            or os.environ.get("SE_N_GPUS", "").strip()
        ),
        "rollout_host": (
            os.environ.get("RBM_ROLLOUT_HOST", "").strip()
            or os.environ.get("SE_ROLLOUT_HOST", "").strip()
            or "127.0.0.1"
        ),
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


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = str(m.get("content", "")).strip()
        if content:
            parts.append(f"[{role}]\n{content}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


def rollout_urls_from_env_or_args(
    *,
    cli_urls: str,
    rollout_host: str,
    rollout_base_port: str,
    rollout_n_servers: str,
    env: Dict[str, str],
) -> List[str]:
    raw = (cli_urls or env["rollout_urls"]).strip()
    if raw:
        return [u.strip().rstrip("/") for u in raw.replace(",", " ").split() if u.strip()]
    base_port = (rollout_base_port or env["rollout_base_port"]).strip()
    n_servers = (rollout_n_servers or env["rollout_n_servers"]).strip()
    host = (rollout_host or env["rollout_host"]).strip() or "127.0.0.1"
    if not base_port or not n_servers:
        return []
    try:
        base = int(base_port)
        n = int(n_servers)
    except ValueError:
        return []
    if n <= 0:
        return []
    return [f"http://{host}:{base + i}" for i in range(n)]


def rollout_complete(
    messages: List[Dict[str, str]],
    *,
    server_urls: List[str],
    timeout: float = 600.0,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    if not server_urls:
        raise ValueError("rollout server_urls is empty")
    base_url = random.choice(server_urls).rstrip("/")
    prompt = messages_to_prompt(messages)
    payload: Dict[str, Any] = {
        "data_records": [{"prompt": prompt, "question": "reasoningbank_math", "gt": "0"}],
        "num_questions": 1,
        "suffix": "reasoningbank_math_distill",
        "rollout_n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{base_url}/rollout", json=payload)
        r.raise_for_status()
        data = r.json()
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"No results in rollout response: {data.keys()}")
    responses = (results[0] or {}).get("responses") or []
    if not responses:
        raise RuntimeError("No responses in rollout result")
    return str(responses[0])
