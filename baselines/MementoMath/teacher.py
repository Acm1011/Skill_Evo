from __future__ import annotations

import os
from pathlib import Path

from baselines.ReasoningBankMath.teacher import (
    chat_complete,
    rollout_complete,
    rollout_urls_from_env_or_args,
)


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_success_instruction() -> str:
    return (_prompt_dir() / "success_case.txt").read_text(encoding="utf-8")


def load_failure_instruction() -> str:
    return (_prompt_dir() / "failure_case.txt").read_text(encoding="utf-8")


def teacher_env() -> dict[str, str]:
    return {
        "base_url": (
            os.environ.get("MMM_TEACHER_BASE_URL", "").strip().rstrip("/")
            or os.environ.get("RBM_TEACHER_BASE_URL", "").strip().rstrip("/")
        ),
        "api_key": (
            os.environ.get("MMM_TEACHER_API_KEY", "").strip()
            or os.environ.get("RBM_TEACHER_API_KEY", "").strip()
        ),
        "model": (
            os.environ.get("MMM_TEACHER_MODEL", "").strip()
            or os.environ.get("RBM_TEACHER_MODEL", "").strip()
        ),
        "rollout_urls": os.environ.get("SE_ROLLOUT_SERVER_URLS", "").strip(),
        "rollout_base_port": (
            os.environ.get("MMM_ROLLOUT_BASE_PORT", "").strip()
            or os.environ.get("RBM_ROLLOUT_BASE_PORT", "").strip()
            or os.environ.get("SE_ROLLOUT_BASE_PORT", "").strip()
        ),
        "rollout_n_servers": (
            os.environ.get("MMM_ROLLOUT_N_SERVERS", "").strip()
            or os.environ.get("RBM_ROLLOUT_N_SERVERS", "").strip()
            or os.environ.get("SE_ROLLOUT_N_SERVERS", "").strip()
            or os.environ.get("SE_N_GPUS", "").strip()
        ),
        "rollout_host": (
            os.environ.get("MMM_ROLLOUT_HOST", "").strip()
            or os.environ.get("RBM_ROLLOUT_HOST", "").strip()
            or os.environ.get("SE_ROLLOUT_HOST", "").strip()
            or "127.0.0.1"
        ),
    }


__all__ = [
    "chat_complete",
    "load_failure_instruction",
    "load_success_instruction",
    "rollout_complete",
    "rollout_urls_from_env_or_args",
    "teacher_env",
]
