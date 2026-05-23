from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from baselines.ReasoningBankMath.io_utils import write_jsonl

from .memory_bank import load_trajectories, make_memory_record
from .memory_parser import parse_teacher_output
from .teacher import (
    chat_complete,
    load_compare_instruction,
    load_failure_instruction,
    load_success_instruction,
    rollout_complete,
    rollout_urls_from_env_or_args,
    teacher_env,
)


def group_trajectories(
    rows: Iterable[Dict[str, Any]],
    *,
    group_by: str,
    max_success_group: int,
    max_failure_group: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_by) or row.get("problem") or row.get("idx"))
        grouped[key].append(row)

    success_take = max(1, min(2, max_success_group))
    failure_take = max(1, min(2, max_failure_group))

    compare_groups: List[Dict[str, Any]] = []
    failure_groups: List[Dict[str, Any]] = []
    success_groups: List[Dict[str, Any]] = []
    for group_rows in grouped.values():
        success_rows = [r for r in group_rows if r.get("is_correct") is True]
        failure_rows = [r for r in group_rows if r.get("is_correct") is not True]
        representative = group_rows[0]
        if success_rows and failure_rows:
            picked_success = success_rows[:success_take]
            picked_failure = failure_rows[:failure_take]
            compare_groups.append(
                {
                    "problem": representative.get("problem"),
                    "topic": representative.get("topic"),
                    "topic_key": representative.get("topic_key"),
                    "status": "mixed",
                    "memory_type": "compare_rule",
                    "rows": picked_success[:1] + picked_failure[:1],
                    "success_rows": picked_success[:1],
                    "failure_rows": picked_failure[:1],
                }
            )
            if len(failure_rows) >= 2:
                failure_groups.append(
                    {
                        "problem": representative.get("problem"),
                        "topic": representative.get("topic"),
                        "topic_key": representative.get("topic_key"),
                        "status": "failure",
                        "memory_type": "failure_rule",
                        "rows": picked_failure,
                        "failure_rows": picked_failure,
                    }
                )
        elif len(success_rows) >= 2:
            picked_success = success_rows[:success_take]
            success_groups.append(
                {
                    "problem": representative.get("problem"),
                    "topic": representative.get("topic"),
                    "topic_key": representative.get("topic_key"),
                    "status": "success",
                    "memory_type": "success_rule",
                    "rows": picked_success,
                    "success_rows": picked_success,
                }
            )
        elif len(failure_rows) >= 2:
            picked_failure = failure_rows[:failure_take]
            failure_groups.append(
                {
                    "problem": representative.get("problem"),
                    "topic": representative.get("topic"),
                    "topic_key": representative.get("topic_key"),
                    "status": "failure",
                    "memory_type": "failure_rule",
                    "rows": picked_failure,
                    "failure_rows": picked_failure,
                }
            )
    return compare_groups + success_groups + failure_groups


def _format_rows(rows: List[Dict[str, Any]], label: str) -> str:
    blocks: List[str] = []
    for i, row in enumerate(rows, start=1):
        correctness = "true" if row.get("is_correct") is True else "false"
        blocks.append(
            "\n".join(
                [
                    f"{label} {i}",
                    f"Problem: {row.get('problem')}",
                    f"Topic: {row.get('topic') or 'unknown'}",
                    f"Correct: {correctness}",
                    f"Ground truth: {row.get('ground_truth') or ''}",
                    "Student solution:",
                    str(row.get("student_response") or "").strip(),
                ]
            ).strip()
        )
    return "\n\n".join(blocks)


def _build_messages(group: Dict[str, Any]) -> List[Dict[str, str]]:
    memory_type = str(group.get("memory_type") or "")
    if memory_type == "compare_rule":
        system_prompt = load_compare_instruction()
        user = (
            f"Topic: {group.get('topic') or 'unknown'}\n\n"
            f"Successful trajectories:\n{_format_rows(group.get('success_rows', []), 'Success')}\n\n"
            f"Failed trajectories:\n{_format_rows(group.get('failure_rows', []), 'Failure')}\n"
        )
    elif memory_type == "success_rule":
        system_prompt = load_success_instruction()
        user = (
            f"Topic: {group.get('topic') or 'unknown'}\n\n"
            f"Successful trajectories:\n{_format_rows(group.get('success_rows', []), 'Success')}\n"
        )
    else:
        system_prompt = load_failure_instruction()
        user = (
            f"Topic: {group.get('topic') or 'unknown'}\n\n"
            f"Failed trajectories:\n{_format_rows(group.get('failure_rows', []), 'Failure')}\n"
        )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def _resolve_teacher_backend(args: argparse.Namespace) -> Tuple[bool, str, str, str, List[str]]:
    env = teacher_env()
    teacher_backend = (args.teacher_backend or "").strip().lower() or "auto"
    base_url = (args.teacher_base_url or env["base_url"]).strip().rstrip("/")
    api_key = args.teacher_api_key or env["api_key"]
    model = (args.teacher_model or env["model"]).strip()
    rollout_urls = rollout_urls_from_env_or_args(
        cli_urls=args.rollout_server_urls,
        rollout_host=args.rollout_host,
        rollout_base_port=args.rollout_base_port,
        rollout_n_servers=args.rollout_n_servers,
        env=env,
    )
    if teacher_backend not in {"auto", "chat", "rollout"}:
        raise SystemExit(f"invalid --teacher-backend: {teacher_backend}")
    if teacher_backend == "auto":
        use_rollout = bool(rollout_urls) and not (base_url and model)
    elif teacher_backend == "rollout":
        use_rollout = True
    else:
        use_rollout = False
    if use_rollout and not rollout_urls:
        raise SystemExit("build-memory: rollout backend requires rollout server urls or host/port settings")
    if (not use_rollout) and (not base_url or not model):
        raise SystemExit("build-memory: chat backend requires --teacher-base-url and --teacher-model")
    return use_rollout, base_url, api_key, model, rollout_urls


def run_build_memory(args: argparse.Namespace) -> int:
    use_rollout, base_url, api_key, model, rollout_urls = _resolve_teacher_backend(args)
    trajectories = load_trajectories(args.trajectories)
    grouped = group_trajectories(
        trajectories,
        group_by=args.group_by,
        max_success_group=args.max_success_group,
        max_failure_group=args.max_failure_group,
    )
    out_rows: List[Dict[str, Any]] = []
    failures = 0
    for group in grouped:
        try:
            messages = _build_messages(group)
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
            parsed = parse_teacher_output(raw, default_memory_type=group["memory_type"])
            out_rows.append(make_memory_record(group, parsed, raw_teacher_output=raw))
        except Exception as e:
            failures += 1
            if args.fail_on_error:
                raise
            print(
                f"[build-memory] skip group topic={group.get('topic_key')} problem={group.get('problem')!r}: {e}",
                file=sys.stderr,
            )
    write_jsonl(args.output, out_rows)
    print(
        f"[build-memory] wrote {len(out_rows)} records to {args.output} | groups={len(grouped)} failures={failures}",
        file=sys.stderr,
    )
    return 0


def build_build_memory_parser(sub: Any) -> None:
    p = sub.add_parser("build-memory", help="Build ExpeL-style math memory bank from trajectories")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--teacher-base-url", default="")
    p.add_argument("--teacher-api-key", default="")
    p.add_argument("--teacher-model", default="")
    p.add_argument("--teacher-backend", default="auto", choices=["auto", "chat", "rollout"])
    p.add_argument("--rollout-server-urls", default="")
    p.add_argument("--rollout-host", default="")
    p.add_argument("--rollout-base-port", default="")
    p.add_argument("--rollout-n-servers", default="")
    p.add_argument("--group-by", default="problem")
    p.add_argument("--max-success-group", type=int, default=4)
    p.add_argument("--max-failure-group", type=int, default=4)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--fail-on-error", action="store_true")
    p.set_defaults(_run=run_build_memory)
