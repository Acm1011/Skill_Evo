"""Distill layered skills from trajectories using a strong teacher (OpenAI-compatible chat)."""
from __future__ import annotations

import argparse
import json
import os
import random
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
        "rollout_urls": os.environ.get("SE_ROLLOUT_SERVER_URLS", "").strip(),
        "rollout_base_port": os.environ.get("SE_ROLLOUT_BASE_PORT", "").strip(),
        "rollout_n_servers": (
            os.environ.get("SE_ROLLOUT_N_SERVERS", "").strip()
            or os.environ.get("SE_N_GPUS", "").strip()
        ),
        "rollout_host": os.environ.get("SE_ROLLOUT_HOST", "").strip() or "127.0.0.1",
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


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = str(m.get("content", "")).strip()
        if content:
            parts.append(f"[{role}]\n{content}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


def _rollout_urls_from_env_or_args(args: argparse.Namespace, env: Dict[str, str]) -> List[str]:
    raw = (args.rollout_server_urls or env["rollout_urls"]).strip()
    if raw:
        return [u.strip().rstrip("/") for u in raw.replace(",", " ").split() if u.strip()]
    base_port = (args.rollout_base_port or env["rollout_base_port"]).strip()
    n_servers = (args.rollout_n_servers or env["rollout_n_servers"]).strip()
    host = (args.rollout_host or env["rollout_host"]).strip() or "127.0.0.1"
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
    max_tokens: int = 8192,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    if not server_urls:
        raise ValueError("rollout server_urls is empty")
    base_url = random.choice(server_urls).rstrip("/")
    prompt = _messages_to_prompt(messages)
    payload: Dict[str, Any] = {
        "data_records": [{"prompt": prompt, "question": "skill_distill", "gt": "0"}],
        "num_questions": 1,
        "suffix": "skillrl_distill",
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


def split_success_failure(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"success": [], "failure": [], "unknown": []}
    for row in rows:
        ic = row.get("is_correct")
        if ic is True:
            buckets["success"].append(row)
        elif ic is False:
            buckets["failure"].append(row)
        else:
            buckets["unknown"].append(row)
    return buckets


def _shorten(text: Any, limit: int = 280) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + "..."


def extract_patterns(rows: List[Dict[str, Any]], *, limit: int = 10) -> str:
    patterns: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        patterns.append(
            {
                "problem": _shorten(row.get("problem", ""), 320),
                "response_excerpt": _shorten(row.get("student_response", ""), 420),
                "topic": str(row.get("topic") or row.get("topic_key") or "unknown"),
                "is_correct": row.get("is_correct"),
                "difficulty": row.get("difficulty"),
            }
        )
    return json.dumps(patterns, ensure_ascii=False, indent=2)


def build_success_failure_block(
    success_rows: List[Dict[str, Any]],
    failure_rows: List[Dict[str, Any]],
    *,
    max_chars: int,
) -> Dict[str, str]:
    success_block = format_trajectory_block(success_rows, max_chars=max_chars // 2)
    failure_block = format_trajectory_block(failure_rows, max_chars=max_chars // 2)
    success_patterns = extract_patterns(success_rows, limit=10)
    failure_patterns = extract_patterns(failure_rows, limit=10) if failure_rows else "[]"
    return {
        "success_trajectories_block": success_block,
        "failure_trajectories_block": failure_block,
        "success_patterns": success_patterns,
        "failure_patterns": failure_patterns,
    }


def load_trajectories(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def topic_bucket_level3(row: Dict[str, Any]) -> str:
    """Use first 3 levels of topic path as distill bucket key."""
    topic = row.get("topic")
    if isinstance(topic, str) and topic.strip():
        parts = [p.strip() for p in topic.split("->") if p.strip()]
        if parts:
            return topic_slug(" -> ".join(parts[:3]))
    tk = row.get("topic_key")
    if isinstance(tk, str) and tk.strip():
        return tk.strip()
    return topic_slug(row.get("topic"))


def run_distill(args: argparse.Namespace) -> int:
    env = teacher_env()
    teacher_backend = (args.teacher_backend or "").strip().lower() or "auto"
    base_url = (args.teacher_base_url or env["base_url"]).strip().rstrip("/")
    api_key = args.teacher_api_key or env["api_key"]
    model = (args.teacher_model or env["model"]).strip()
    rollout_urls = _rollout_urls_from_env_or_args(args, env)
    if teacher_backend not in {"auto", "chat", "rollout"}:
        print(f"[distill] 非法 --teacher-backend: {teacher_backend}", file=sys.stderr)
        return 2
    if teacher_backend == "auto":
        use_rollout = bool(rollout_urls) and not (base_url and model)
    elif teacher_backend == "rollout":
        use_rollout = True
    else:
        use_rollout = False

    if use_rollout and not rollout_urls:
        print(
            "[distill] rollout 模式需要 SE_ROLLOUT_SERVER_URLS，或 "
            "SE_ROLLOUT_BASE_PORT + SE_ROLLOUT_N_SERVERS/SE_N_GPUS",
            file=sys.stderr,
        )
        return 2
    if (not use_rollout) and (not base_url or not model):
        print(
            "[distill] 需要 SKILLRL_TEACHER_BASE_URL 与 SKILLRL_TEACHER_MODEL "
            "（或命令行 --teacher-base-url / --teacher-model）",
            file=sys.stderr,
        )
        return 2
    mode_name = "rollout" if use_rollout else "chat"
    print(f"[distill] teacher backend={mode_name}", file=sys.stderr)

    rows = load_trajectories(args.trajectories)
    by_topic: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tk = topic_bucket_level3(row)
        by_topic[tk].append(row)

    bank = LayeredSkillBank(empty_bank())
    template = load_teacher_template()
    raw_dump_dir = Path(args.raw_dump_dir) if args.raw_dump_dir else None
    if raw_dump_dir:
        raw_dump_dir.mkdir(parents=True, exist_ok=True)

    topic_keys = sorted(by_topic.keys(), key=lambda k: (-len(by_topic[k]), k))
    for tkey in topic_keys:
        bucket_rows = by_topic[tkey]
        sf = split_success_failure(bucket_rows)
        success_rows = sf["success"]
        failure_rows = sf["failure"] + sf["unknown"]
        # chunk bucket
        succ_i = 0
        fail_i = 0
        batch_num = 0
        while succ_i < len(success_rows) or fail_i < len(failure_rows):
            succ_chunk: List[Dict[str, Any]] = []
            fail_chunk: List[Dict[str, Any]] = []
            char_budget = 0
            max_per_side = max(1, args.max_problems_per_call // 2)
            while (
                succ_i < len(success_rows)
                and len(succ_chunk) < max_per_side
                and len(succ_chunk) + len(fail_chunk) < args.max_problems_per_call
            ):
                r = success_rows[succ_i]
                est = len(str(r.get("problem", ""))) + len(str(r.get("student_response", ""))) + 100
                if (succ_chunk or fail_chunk) and char_budget + est > args.max_chars_per_call:
                    break
                succ_chunk.append(r)
                char_budget += est
                succ_i += 1
            while (
                fail_i < len(failure_rows)
                and len(fail_chunk) < max_per_side
                and len(succ_chunk) + len(fail_chunk) < args.max_problems_per_call
            ):
                r = failure_rows[fail_i]
                est = len(str(r.get("problem", ""))) + len(str(r.get("student_response", ""))) + 100
                if (succ_chunk or fail_chunk) and char_budget + est > args.max_chars_per_call:
                    break
                fail_chunk.append(r)
                char_budget += est
                fail_i += 1
            if not succ_chunk and succ_i < len(success_rows):
                succ_chunk.append(success_rows[succ_i])
                succ_i += 1
            if not fail_chunk and fail_i < len(failure_rows) and len(succ_chunk) < args.max_problems_per_call:
                fail_chunk.append(failure_rows[fail_i])
                fail_i += 1
            if not succ_chunk and not fail_chunk:
                continue

            blocks = build_success_failure_block(
                succ_chunk,
                fail_chunk,
                max_chars=args.max_chars_per_call,
            )
            user = template.format(
                topic_bucket=tkey,
                success_trajectories_block=blocks["success_trajectories_block"],
                failure_trajectories_block=blocks["failure_trajectories_block"],
                success_patterns=blocks["success_patterns"],
                failure_patterns=blocks["failure_patterns"],
            )
            messages = [
                {
                    "role": "system",
                    "content": "You output only valid JSON objects for skill distillation.",
                },
                {"role": "user", "content": user},
            ]
            try:
                if use_rollout:
                    raw = rollout_complete(
                        messages,
                        server_urls=rollout_urls,
                        timeout=args.timeout,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                else:
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
    p.add_argument("--teacher-backend", default="auto", choices=["auto", "chat", "rollout"])
    p.add_argument("--rollout-server-urls", default="", help="Space/comma separated, e.g. http://127.0.0.1:8760,http://127.0.0.1:8761")
    p.add_argument("--rollout-host", default="")
    p.add_argument("--rollout-base-port", default="")
    p.add_argument("--rollout-n-servers", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
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
