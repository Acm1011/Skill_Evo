from __future__ import annotations

import argparse
import sys
from typing import Any, List

from baselines.ReasoningBankMath.io_utils import read_jsonl, write_jsonl

from .build_embeddings import run_build_embeddings
from .build_memory import _resolve_teacher_backend, group_trajectories, _build_messages
from .memory_bank import dedupe_records, load_trajectories, make_memory_record
from .memory_parser import parse_teacher_output
from .teacher import chat_complete, rollout_complete


def run_evolve_memory(args: argparse.Namespace) -> int:
    use_rollout, base_url, api_key, model, rollout_urls = _resolve_teacher_backend(args)
    old_rows = read_jsonl(args.memory_bank)
    trajectories = load_trajectories(args.trajectories)
    grouped = group_trajectories(
        trajectories,
        group_by=args.group_by,
        max_success_group=args.max_success_group,
        max_failure_group=args.max_failure_group,
    )
    new_records: List[dict] = []
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
            new_records.append(make_memory_record(group, parsed, raw_teacher_output=raw))
        except Exception as e:
            if args.fail_on_error:
                raise
            print(f"[evolve-memory] skip group {group.get('problem')!r}: {e}", file=sys.stderr)

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
    p = sub.add_parser("evolve-memory", help="Incrementally update ExpeLMath memory bank")
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
    p.add_argument("--group-by", default="problem")
    p.add_argument("--max-success-group", type=int, default=4)
    p.add_argument("--max-failure-group", type=int, default=4)
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

