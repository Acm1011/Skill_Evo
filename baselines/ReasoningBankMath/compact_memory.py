from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from .build_embeddings import run_build_embeddings
from .io_utils import read_jsonl, write_jsonl
from .memory_bank import build_embedding_text, dedupe_records
from .text_utils import utc_now_iso


def _normalize_record(row: Dict[str, Any], row_idx: int) -> Dict[str, Any]:
    rec = dict(row)
    rec.setdefault("memory_id", f"legacy_mem_{row_idx:08d}")
    rec.setdefault("source_idx", rec.get("source_idx", row_idx))
    rec.setdefault("query", str(rec.get("query") or "").strip())
    rec.setdefault("topic", rec.get("topic"))
    rec.setdefault("topic_key", rec.get("topic_key") or "unknown")
    rec.setdefault("status", rec.get("status") or "unknown")
    rec.setdefault("trajectory", str(rec.get("trajectory") or ""))
    rec.setdefault("memory_items", [])
    rec.setdefault("created_at", rec.get("created_at") or utc_now_iso())
    rec.setdefault("updated_at", rec.get("updated_at") or rec["created_at"])
    rec.setdefault("raw_teacher_output", str(rec.get("raw_teacher_output") or ""))
    rec.setdefault("provenance", [])
    rec.setdefault("duplicate_count", int(rec.get("duplicate_count") or 0))
    if not str(rec.get("embedding_text") or "").strip():
        rec["embedding_text"] = build_embedding_text(
            str(rec.get("query") or ""),
            rec.get("topic"),
            list(rec.get("memory_items") or []),
        )
    return rec


def run_compact_memory(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.memory_bank)
    normalized = [_normalize_record(row, i) for i, row in enumerate(rows)]

    merged, duplicate_map = dedupe_records(
        [],
        normalized,
        similarity_threshold=args.similarity_threshold,
        embed_backend=args.embed_backend,
        embed_base_url=args.embed_base_url,
        embed_api_key=args.embed_api_key,
        embed_model=args.embed_model,
        embed_timeout=args.timeout,
        embed_dim=args.hash_dim,
    )
    write_jsonl(args.output_memory_bank, merged)

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
        "[compact-memory] "
        f"input={len(rows)} merged={len(merged)} duplicates={len(duplicate_map)} "
        f"output={args.output_memory_bank}",
        file=sys.stderr,
    )
    return 0


def build_compact_memory_parser(sub: Any) -> None:
    p = sub.add_parser("compact-memory", help="Deduplicate/merge an existing memory bank without new trajectories")
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--output-memory-bank", required=True)
    p.add_argument("--existing-embeddings", default="")
    p.add_argument("--output-embeddings", default="")
    p.add_argument("--embed-backend", default="hash", choices=["hash", "openai"])
    p.add_argument("--embed-base-url", default="")
    p.add_argument("--embed-api-key", default="")
    p.add_argument("--embed-model", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--hash-dim", type=int, default=256)
    p.add_argument("--similarity-threshold", type=float, default=0.98)
    p.set_defaults(_run=run_compact_memory)

