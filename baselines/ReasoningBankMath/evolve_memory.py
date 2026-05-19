from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from .build_embeddings import run_build_embeddings
from .io_utils import read_jsonl, write_jsonl
from .memory_bank import dedupe_records, load_trajectories, make_memory_record, parse_teacher_output
from .teacher import (
    chat_complete,
    load_failure_instruction,
    load_success_instruction,
    rollout_complete,
    rollout_urls_from_env_or_args,
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


def run_evolve_memory(args: argparse.Namespace) -> int:
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
        raise SystemExit("evolve-memory: rollout backend requires rollout server urls or host/port settings")
    if (not use_rollout) and (not base_url or not model):
        raise SystemExit("evolve-memory: chat backend requires --teacher-base-url and --teacher-model")
    old_rows = read_jsonl(args.memory_bank)
    trajectories = load_trajectories(args.trajectories)

    new_records: List[Dict[str, Any]] = []
    for traj in trajectories:
        system_prompt = (
            load_success_instruction() if traj.get("is_correct") is True else load_failure_instruction()
        )
        try:
            messages = [{"role": "system", "content": system_prompt}] + _build_messages(traj)
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
            items = parse_teacher_output(raw)
            new_records.append(make_memory_record(traj, items, raw_teacher_output=raw))
        except Exception as e:
            if args.fail_on_error:
                raise
            print(f"[evolve-memory] skip idx={traj.get('idx')}: {e}", file=sys.stderr)
            continue

    merged_rows, duplicate_map = dedupe_records(
        old_rows,
        new_records,
        similarity_threshold=args.similarity_threshold,
        embed_backend=args.embed_backend,
        embed_base_url=args.embed_base_url,
        embed_api_key=args.embed_api_key,
        embed_model=args.embed_model,
        embed_timeout=args.timeout,
        embed_dim=args.hash_dim,
    )
    write_jsonl(args.output_memory_bank, merged_rows)

    if args.output_embeddings:
        emb_args = argparse.Namespace(
            memory_bank=args.output_memory_bank,
            output=args.output_embeddings,
            existing_embeddings=args.existing_embeddings,
            backend=args.embed_backend,
            embed_base_url=args.embed_base_url,
            embed_api_key=args.embed_api_key,
            embed_model=args.embed_model,
            timeout=args.timeout,
            hash_dim=args.hash_dim,
        )
        run_build_embeddings(emb_args)

    print(
        "[evolve-memory] "
        f"old={len(old_rows)} new={len(new_records)} merged={len(merged_rows)} "
        f"duplicates={len(duplicate_map)} output={args.output_memory_bank}",
        file=sys.stderr,
    )
    return 0


def build_evolve_memory_parser(sub: Any) -> None:
    p = sub.add_parser("evolve-memory", help="Incrementally update memory bank with new trajectories")
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--output-memory-bank", required=True)
    p.add_argument("--existing-embeddings", default="")
    p.add_argument("--output-embeddings", default="")
    p.add_argument("--teacher-base-url", default="")
    p.add_argument("--teacher-api-key", default="")
    p.add_argument("--teacher-model", default="")
    p.add_argument("--teacher-backend", default="auto", choices=["auto", "chat", "rollout"])
    p.add_argument("--rollout-server-urls", default="")
    p.add_argument("--rollout-host", default="")
    p.add_argument("--rollout-base-port", default="")
    p.add_argument("--rollout-n-servers", default="")
    p.add_argument("--embed-backend", default="hash", choices=["hash", "openai"])
    p.add_argument("--embed-base-url", default="")
    p.add_argument("--embed-api-key", default="")
    p.add_argument("--embed-model", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--hash-dim", type=int, default=256)
    p.add_argument("--similarity-threshold", type=float, default=0.98)
    p.add_argument("--fail-on-error", action="store_true")
    p.set_defaults(_run=run_evolve_memory)
