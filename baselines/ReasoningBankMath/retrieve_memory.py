from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

from .embedding_backend import cosine_similarity, embed_texts, embedding_env
from .io_utils import read_jsonl
from .text_utils import topic_slug


def _format_memory_items(items: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for i, item in enumerate(items, start=1):
        blocks.append(
            "\n".join(
                [
                    f"# Memory Item {i}",
                    f"## Title {item.get('title', '')}",
                    f"## Description {item.get('description', '')}",
                    f"## Content {item.get('content', '')}",
                ]
            ).strip()
        )
    return "\n\n".join(b for b in blocks if b)


def format_retrieved_prompt(rows: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for i, row in enumerate(rows, start=1):
        out.append(f"### Retrieved Memory {i}")
        out.append(f"- memory_id: {row.get('memory_id')}")
        out.append(f"- topic_key: {row.get('topic_key')}")
        out.append(f"- status: {row.get('status')}")
        out.append(_format_memory_items(list(row.get("memory_items") or [])))
    return "\n\n".join(out).strip()


def retrieve_records(
    *,
    question: str,
    memory_rows: List[Dict[str, Any]],
    embedding_rows: List[Dict[str, Any]],
    top_k: int,
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    hash_dim: int,
    query_topic: Optional[str] = None,
    topic_bonus: float = 0.05,
) -> List[Tuple[Dict[str, Any], float]]:
    emb_by_id = {str(r.get("memory_id") or ""): r for r in embedding_rows}
    q_vec = embed_texts(
        [question],
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        dim=hash_dim,
    )[0]
    q_topic_key = topic_slug(query_topic)
    scored: List[Tuple[Dict[str, Any], float]] = []
    for row in memory_rows:
        mid = str(row.get("memory_id") or "")
        emb = emb_by_id.get(mid, {}).get("embedding")
        if not isinstance(emb, list):
            continue
        score = cosine_similarity(q_vec, emb)
        if q_topic_key != "unknown" and str(row.get("topic_key") or "") == q_topic_key:
            score += topic_bonus
        scored.append((row, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(0, top_k)]


def run_retrieve(args: argparse.Namespace) -> int:
    env = embedding_env()
    base_url = (args.embed_base_url or env["base_url"]).strip().rstrip("/")
    api_key = args.embed_api_key or env["api_key"]
    model = (args.embed_model or env["model"]).strip()
    memory_rows = read_jsonl(args.memory_bank)
    embedding_rows = read_jsonl(args.embeddings)
    scored = retrieve_records(
        question=args.question,
        memory_rows=memory_rows,
        embedding_rows=embedding_rows,
        top_k=args.top_k,
        backend=args.backend,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=args.timeout,
        hash_dim=args.hash_dim,
        query_topic=args.topic,
        topic_bonus=args.topic_bonus,
    )
    rows = []
    for row, score in scored:
        item = dict(row)
        item["_score"] = score
        rows.append(item)
    if args.output_format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(format_retrieved_prompt(rows))
    return 0


def build_retrieve_parser(sub: Any) -> None:
    p = sub.add_parser("retrieve", help="Retrieve relevant memories for a math question")
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--topic", default="")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--output-format", default="prompt", choices=["prompt", "json"])
    p.add_argument("--backend", default="hash", choices=["hash", "openai"])
    p.add_argument("--embed-base-url", default="")
    p.add_argument("--embed-api-key", default="")
    p.add_argument("--embed-model", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--hash-dim", type=int, default=256)
    p.add_argument("--topic-bonus", type=float, default=0.05)
    p.set_defaults(_run=run_retrieve)

