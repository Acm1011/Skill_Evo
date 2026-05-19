from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from .io_utils import write_jsonl
from .memory_bank import load_trajectories, make_memory_record, parse_teacher_output
from .teacher import (
    chat_complete,
    load_failure_instruction,
    load_success_instruction,
    teacher_env,
)


def _build_messages(traj: Dict[str, Any]) -> List[Dict[str, str]]:
    correctness = traj.get("is_correct")
    correctness_s = "true" if correctness is True else ("false" if correctness is False else "unknown")
    user = (
        f"Problem:\n{traj['problem']}\n\n"
        f"Topic: {traj.get('topic') or 'unknown'}\n"
        f"Correct: {correctness_s}\n"
        f"Ground truth: {traj.get('ground_truth') or ''}\n\n"
        f"Student solution:\n{traj['student_response']}\n"
    )
    return [{"role": "user", "content": user}]


def run_build_memory(args: argparse.Namespace) -> int:
    env = teacher_env()
    base_url = (args.teacher_base_url or env["base_url"]).strip().rstrip("/")
    api_key = args.teacher_api_key or env["api_key"]
    model = (args.teacher_model or env["model"]).strip()
    trajectories = load_trajectories(args.trajectories)
    out_rows: List[Dict[str, Any]] = []
    failures = 0

    for traj in trajectories:
        system_prompt = (
            load_success_instruction() if traj.get("is_correct") is True else load_failure_instruction()
        )
        try:
            raw = chat_complete(
                [{"role": "system", "content": system_prompt}] + _build_messages(traj),
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=args.timeout,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            items = parse_teacher_output(raw)
            out_rows.append(make_memory_record(traj, items, raw_teacher_output=raw))
        except Exception as e:
            failures += 1
            if args.fail_on_error:
                raise
            print(f"[build-memory] skip idx={traj.get('idx')}: {e}", file=sys.stderr)
            continue

    write_jsonl(args.output, out_rows)
    print(
        f"[build-memory] wrote {len(out_rows)} records to {args.output} | failures={failures}",
        file=sys.stderr,
    )
    return 0


def build_build_memory_parser(sub: Any) -> None:
    p = sub.add_parser("build-memory", help="Teacher distill trajectories -> memory_bank.jsonl")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--teacher-base-url", default="")
    p.add_argument("--teacher-api-key", default="")
    p.add_argument("--teacher-model", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--fail-on-error", action="store_true")
    p.set_defaults(_run=run_build_memory)
