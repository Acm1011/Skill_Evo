"""Distill layered skills from trajectories using a strong teacher (OpenAI-compatible chat)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

import httpx

from .layered_skill_bank import LayeredSkillBank, empty_bank
from .merge_skill_bank import cap_mistakes, merge_partial_into_bank
from .text_utils import parse_json_object, topic_slug


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_teacher_template() -> str:
    return (_prompt_dir() / "teacher_layered_skills.txt").read_text(encoding="utf-8")


def teacher_env() -> Dict[str, str]:
    return {
        "base_url": os.environ.get("SKILLRL_TEACHER_BASE_URL", "").strip().rstrip("/"),
        "api_key": os.environ.get("SKILLRL_TEACHER_API_KEY", "").strip(),
        "model": os.environ.get("SKILLRL_TEACHER_MODEL", "").strip(),
    }


def chat_complete(
    messages: List[Dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 600.0,
    temperature: float = 0.2,
    max_tokens: int = 8192,
) -> str:
    if not base_url:
        raise ValueError("SKILLRL_TEACHER_BASE_URL is empty")
    if not model:
        raise ValueError("SKILLRL_TEACHER_MODEL is empty")
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


def format_trajectory_block(rows: List[Dict[str, Any]], max_chars: int) -> str:
    parts: List[str] = []
    total = 0
    for i, row in enumerate(rows):
        prob = row.get("problem", "")
        resp = row.get("student_response", "")
        ic = row.get("is_correct")
        ic_s = "unknown" if ic is None else ("true" if ic else "false")
        block = (
            f"--- Example {i + 1} ---\n"
            f"Correct (if known): {ic_s}\n"
            f"Problem:\n{prob}\n"
            f"Student solution:\n{resp}\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def load_trajectories(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_distill(args: argparse.Namespace) -> int:
    env = teacher_env()
    base_url = (args.teacher_base_url or env["base_url"]).strip().rstrip("/")
    api_key = args.teacher_api_key or env["api_key"]
    model = (args.teacher_model or env["model"]).strip()
    if not base_url or not model:
        print(
            "[distill] 需要 SKILLRL_TEACHER_BASE_URL 与 SKILLRL_TEACHER_MODEL "
            "（或命令行 --teacher-base-url / --teacher-model）",
            file=sys.stderr,
        )
        return 2

    rows = load_trajectories(args.trajectories)
    by_topic: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tk = row.get("topic_key") or topic_slug(row.get("topic"))
        by_topic[tk].append(row)

    bank = LayeredSkillBank(empty_bank())
    template = load_teacher_template()
    raw_dump_dir = Path(args.raw_dump_dir) if args.raw_dump_dir else None
    if raw_dump_dir:
        raw_dump_dir.mkdir(parents=True, exist_ok=True)

    topic_keys = sorted(by_topic.keys(), key=lambda k: (-len(by_topic[k]), k))
    for tkey in topic_keys:
        bucket_rows = by_topic[tkey]
        # chunk bucket
        i = 0
        batch_num = 0
        while i < len(bucket_rows):
            chunk: List[Dict[str, Any]] = []
            char_budget = 0
            while i < len(bucket_rows) and len(chunk) < args.max_problems_per_call:
                r = bucket_rows[i]
                est = len(str(r.get("problem", ""))) + len(str(r.get("student_response", ""))) + 100
                if chunk and char_budget + est > args.max_chars_per_call:
                    break
                chunk.append(r)
                char_budget += est
                i += 1
            if not chunk:
                i += 1
                continue

            traj_block = format_trajectory_block(chunk, args.max_chars_per_call)
            user = template.format(topic_bucket=tkey, trajectories_block=traj_block)
            messages = [
                {
                    "role": "system",
                    "content": "You output only valid JSON objects for skill distillation.",
                },
                {"role": "user", "content": user},
            ]
            try:
                raw = chat_complete(
                    messages,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                print(f"[distill] teacher error topic={tkey} batch={batch_num}: {e}", file=sys.stderr)
                batch_num += 1
                continue

            if raw_dump_dir:
                (raw_dump_dir / f"{tkey}_{batch_num}.txt").write_text(raw, encoding="utf-8")

            partial = parse_json_object(raw)
            if not partial:
                print(
                    f"[distill] JSON parse failed topic={tkey} batch={batch_num}",
                    file=sys.stderr,
                )
                batch_num += 1
                continue

            merge_partial_into_bank(bank, partial, tkey)
            batch_num += 1

    cap_mistakes(bank, args.mistakes_cap)
    bank.save(args.output_skills)
    print(
        f"[distill] saved {args.output_skills} | counts={bank.skill_counts()}",
        file=sys.stderr,
    )
    return 0


def build_distill_parser(sub: Any) -> None:
    p = sub.add_parser("distill", help="Teacher distillation -> claude_style_skills.json")
    p.add_argument("--trajectories", required=True, help="trajectories.jsonl from gen-traj")
    p.add_argument("--output-skills", required=True, help="Output claude_style_skills.json")
    p.add_argument("--max-problems-per-call", type=int, default=8)
    p.add_argument("--max-chars-per-call", type=int, default=12000)
    p.add_argument("--mistakes-cap", type=int, default=50)
    p.add_argument("--raw-dump-dir", default="", help="Save raw teacher replies for debugging")
    p.add_argument("--teacher-base-url", default="")
    p.add_argument("--teacher-api-key", default="")
    p.add_argument("--teacher-model", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.set_defaults(_run=run_distill)


def run_inspect(args: argparse.Namespace) -> int:
    bank = LayeredSkillBank.from_path(args.skills_json)
    retrieved = bank.retrieve(
        task_description=args.problem or "",
        topic=args.topic,
        top_k=args.top_k,
        mistakes_cap=args.mistakes_cap,
    )
    print(bank.format_for_prompt(retrieved))
    return 0


def build_inspect_parser(sub: Any) -> None:
    p = sub.add_parser("inspect", help="Print format_for_prompt for a topic")
    p.add_argument("--skills-json", required=True)
    p.add_argument("--topic", default="unknown", help="Topic slug (topic_key)")
    p.add_argument("--problem", default="", help="Unused in template mode; placeholder")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--mistakes-cap", type=int, default=5)
    p.set_defaults(_run=run_inspect)
