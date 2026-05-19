from __future__ import annotations

from pathlib import Path

from baselines.ReasoningBankMath.teacher import (
    chat_complete,
    rollout_complete,
    rollout_urls_from_env_or_args,
    teacher_env,
)


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_compare_instruction() -> str:
    return (_prompt_dir() / "compare_memory.txt").read_text(encoding="utf-8")


def load_success_instruction() -> str:
    return (_prompt_dir() / "success_memory.txt").read_text(encoding="utf-8")


def load_failure_instruction() -> str:
    return (_prompt_dir() / "failure_memory.txt").read_text(encoding="utf-8")


def load_solve_instruction() -> str:
    return (_prompt_dir() / "solve_with_memory.txt").read_text(encoding="utf-8")

