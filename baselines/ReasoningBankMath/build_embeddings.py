from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from .embedding_backend import embed_texts, embedding_env
from .io_utils import read_jsonl, write_jsonl
from .text_utils import utc_now_iso


def run_build_embeddings(args: argparse.Namespace) -> int:
    env = embedding_env()
    base_url = (args.embed_base_url or env["base_url"]).strip().rstrip("/")
    api_key = args.embed_api_key or env["api_key"]
    model = (args.embed_model or env["model"]).strip()

    memory_rows = read_jsonl(args.memory_bank)
    old_rows: Dict[str, Dict[str, Any]] = {}
    if args.existing_embeddings:
        for row in read_jsonl(args.existing_embeddings):
            mid = str(row.get("memory_id") or "")
            if mid:
                old_rows[mid] = row

    pending_ids: List[str] = []
    pending_texts: List[str] = []
    out_rows: List[Dict[str, Any]] = []
    for row in memory_rows:
        mid = str(row.get("memory_id") or "")
        text = str(row.get("embedding_text") or "")
        prev = old_rows.get(mid)
        if (
            prev
            and str(prev.get("text") or "") == text
            and str(prev.get("backend") or "") == args.backend
            and str(prev.get("model") or "") == model
        ):
            out_rows.append(prev)
            continue
        pending_ids.append(mid)
        pending_texts.append(text)

    new_embeddings = embed_texts(
        pending_texts,
        backend=args.backend,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=args.timeout,
        dim=args.hash_dim,
    )
    now = utc_now_iso()
    for mid, text, emb in zip(pending_ids, pending_texts, new_embeddings):
        out_rows.append(
            {
                "memory_id": mid,
                "text": text,
                "embedding": emb,
                "backend": args.backend,
                "model": model if args.backend != "hash" else f"hash-{args.hash_dim}",
                "updated_at": now,
            }
        )

    out_rows.sort(key=lambda x: str(x.get("memory_id") or ""))
    write_jsonl(args.output, out_rows)
    print(
        f"[build-embeddings] wrote {len(out_rows)} rows to {args.output} | refreshed={len(pending_ids)}",
        file=sys.stderr,
    )
    return 0


def build_build_embeddings_parser(sub: Any) -> None:
    p = sub.add_parser("build-embeddings", help="Build or refresh memory_embeddings.jsonl")
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--existing-embeddings", default="")
    p.add_argument("--backend", default="hash", choices=["hash", "openai"])
    p.add_argument("--embed-base-url", default="")
    p.add_argument("--embed-api-key", default="")
    p.add_argument("--embed-model", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--hash-dim", type=int, default=256)
    p.set_defaults(_run=run_build_embeddings)

