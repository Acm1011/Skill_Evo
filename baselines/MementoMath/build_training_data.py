from __future__ import annotations

import argparse
import random
import sys
from typing import Any, Dict, List

from .embedding_backend import cosine_similarity, embed_texts, embedding_env
from .io_utils import read_jsonl, write_jsonl
from .memory_bank import export_case_pool_row


def _select_top_neighbors(
    row: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    embeddings: Dict[str, List[float]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    src = embeddings.get(str(row.get("memory_id") or ""))
    if src is None:
        return candidates[:limit]
    scored = []
    for cand in candidates:
        vec = embeddings.get(str(cand.get("memory_id") or ""))
        if vec is None:
            continue
        scored.append((cand, cosine_similarity(src, vec)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cand for cand, _ in scored[:limit]]


def run_build_training_data(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    memory_rows = read_jsonl(args.memory_bank)
    env = embedding_env()
    base_url = (args.embed_base_url or env["base_url"]).strip().rstrip("/")
    api_key = args.embed_api_key or env["api_key"]
    model = (args.embed_model or env["model"]).strip()

    embeddings_by_id: Dict[str, List[float]] = {}
    if args.embeddings:
        for row in read_jsonl(args.embeddings):
            mid = str(row.get("memory_id") or "")
            emb = row.get("embedding")
            if mid and isinstance(emb, list):
                embeddings_by_id[mid] = [float(x) for x in emb]
    else:
        new_embs = embed_texts(
            [str(row.get("embedding_text") or "") for row in memory_rows],
            backend=args.backend,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=args.timeout,
            dim=args.hash_dim,
        )
        for row, emb in zip(memory_rows, new_embs):
            embeddings_by_id[str(row.get("memory_id") or "")] = emb

    out_rows: List[Dict[str, Any]] = []
    for row in memory_rows:
        same_topic = [
            x for x in memory_rows
            if x.get("memory_id") != row.get("memory_id")
            and x.get("topic_key") == row.get("topic_key")
            and x.get("case_label") == row.get("case_label")
        ]
        cross_topic = [
            x for x in memory_rows
            if x.get("memory_id") != row.get("memory_id")
            and (
                x.get("topic_key") != row.get("topic_key")
                or x.get("case_label") != row.get("case_label")
            )
        ]
        pos_rows = _select_top_neighbors(
            row,
            same_topic,
            embeddings_by_id,
            limit=args.num_positive,
        )
        neg_rows = _select_top_neighbors(
            row,
            cross_topic,
            embeddings_by_id,
            limit=max(args.num_negative * 3, args.num_negative),
        )
        if len(neg_rows) > args.num_negative:
            rng.shuffle(neg_rows)
            neg_rows = neg_rows[: args.num_negative]
        for cand in pos_rows:
            case_row = export_case_pool_row(cand)
            out_rows.append(
                {
                    "query": row.get("query"),
                    "case": case_row["case"],
                    "case_label": case_row["case_label"],
                    "plan": case_row["plan"],
                    "truth_label": True,
                }
            )
        for cand in neg_rows:
            case_row = export_case_pool_row(cand)
            out_rows.append(
                {
                    "query": row.get("query"),
                    "case": case_row["case"],
                    "case_label": case_row["case_label"],
                    "plan": case_row["plan"],
                    "truth_label": False,
                }
            )

    write_jsonl(args.output, out_rows)
    print(
        f"[build-training-data] wrote {len(out_rows)} rows to {args.output}",
        file=sys.stderr,
    )
    return 0


def build_build_training_data_parser(sub: Any) -> None:
    p = sub.add_parser("build-training-data", help="Build Memento retriever training_data.jsonl")
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--embeddings", default="")
    p.add_argument("--num-positive", type=int, default=4)
    p.add_argument("--num-negative", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", default="hash", choices=["hash", "openai"])
    p.add_argument("--embed-base-url", default="")
    p.add_argument("--embed-api-key", default="")
    p.add_argument("--embed-model", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--hash-dim", type=int, default=256)
    p.set_defaults(_run=run_build_training_data)
