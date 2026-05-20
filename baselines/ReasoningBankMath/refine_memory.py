from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Set

from .build_embeddings import run_build_embeddings
from .embedding_backend import cosine_similarity, embed_texts
from .io_utils import read_jsonl, write_jsonl
from .memory_bank import build_embedding_text
from .memory_parser import parse_memory_items
from .teacher import (
    chat_complete,
    rollout_complete,
    rollout_urls_from_env_or_args,
    teacher_env,
)
from .text_utils import short_hash, utc_now_iso


def load_parallel_instruction() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parent / "prompts" / "parallel_memory.txt").read_text(
        encoding="utf-8"
    )


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


def _record_summary(rec: Dict[str, Any], idx: int) -> str:
    lines = [
        f"## Memory Record {idx + 1}",
        f"memory_id: {rec.get('memory_id')}",
        f"topic_key: {rec.get('topic_key')}",
        f"status: {rec.get('status')}",
        f"query: {rec.get('query')}",
    ]
    for j, item in enumerate(rec.get("memory_items") or [], start=1):
        lines.extend(
            [
                f"# Input Memory Item {j}",
                f"## Title {item.get('title', '')}",
                f"## Description {item.get('description', '')}",
                f"## Content {item.get('content', '')}",
            ]
        )
    return "\n".join(lines).strip()


def cluster_records(
    rows: List[Dict[str, Any]],
    *,
    embed_backend: str,
    embed_base_url: str,
    embed_api_key: str,
    embed_model: str,
    timeout: float,
    hash_dim: int,
    similarity_threshold: float,
) -> List[List[int]]:
    if not rows:
        return []
    embs = embed_texts(
        [str(r.get("embedding_text") or "") for r in rows],
        backend=embed_backend,
        base_url=embed_base_url,
        api_key=embed_api_key,
        model=embed_model,
        timeout=timeout,
        dim=hash_dim,
    )
    visited: Set[int] = set()
    clusters: List[List[int]] = []
    for i, rec in enumerate(rows):
        if i in visited:
            continue
        visited.add(i)
        cluster = [i]
        for j in range(i + 1, len(rows)):
            if j in visited:
                continue
            if str(rows[j].get("topic_key") or "") != str(rec.get("topic_key") or ""):
                continue
            score = cosine_similarity(embs[i], embs[j])
            if score >= similarity_threshold:
                visited.add(j)
                cluster.append(j)
        clusters.append(cluster)
    return clusters


def _build_merged_record(cluster_rows: List[Dict[str, Any]], memory_items: List[Dict[str, str]]) -> Dict[str, Any]:
    base = cluster_rows[0]
    provenance: List[Dict[str, Any]] = []
    total_duplicates = 0
    queries: List[str] = []
    statuses: List[str] = []
    for row in cluster_rows:
        provenance.extend(list(row.get("provenance") or []))
        total_duplicates += int(row.get("duplicate_count") or 0)
        q = str(row.get("query") or "").strip()
        if q:
            queries.append(q)
        s = str(row.get("status") or "").strip()
        if s:
            statuses.append(s)
    uniq_queries = []
    seen = set()
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq_queries.append(q)
    query = uniq_queries[0] if uniq_queries else str(base.get("query") or "")
    topic = base.get("topic")
    topic_key = base.get("topic_key") or "unknown"
    status = "mixed" if len(set(statuses)) > 1 else (statuses[0] if statuses else str(base.get("status") or "unknown"))
    raw_prompt = "\n\n".join(_record_summary(r, i) for i, r in enumerate(cluster_rows))
    return {
        "memory_id": f"refined_mem_{short_hash(topic_key + '|' + raw_prompt)}",
        "source_idx": base.get("source_idx"),
        "query": query,
        "topic": topic,
        "topic_key": topic_key,
        "status": status,
        "trajectory": "",
        "memory_items": memory_items,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "embedding_text": build_embedding_text(query, topic, memory_items),
        "raw_teacher_output": raw_prompt,
        "provenance": provenance,
        "duplicate_count": total_duplicates + max(0, len(cluster_rows) - 1),
    }


def run_refine_memory(args: argparse.Namespace) -> int:
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
        raise SystemExit("refine-memory: rollout backend requires rollout server urls or host/port settings")
    if (not use_rollout) and (not base_url or not model):
        raise SystemExit("refine-memory: chat backend requires --teacher-base-url and --teacher-model")

    rows = [_normalize_record(row, i) for i, row in enumerate(read_jsonl(args.memory_bank))]
    clusters = cluster_records(
        rows,
        embed_backend=args.embed_backend,
        embed_base_url=args.embed_base_url,
        embed_api_key=args.embed_api_key,
        embed_model=args.embed_model,
        timeout=args.timeout,
        hash_dim=args.hash_dim,
        similarity_threshold=args.cluster_similarity_threshold,
    )
    instruction = load_parallel_instruction()
    out_rows: List[Dict[str, Any]] = []
    failures = 0
    for cluster in clusters:
        cluster_rows = [rows[i] for i in cluster]
        if len(cluster_rows) == 1 and not args.refine_singletons:
            out_rows.append(cluster_rows[0])
            continue
        prompt = "\n\n".join(_record_summary(r, i) for i, r in enumerate(cluster_rows))
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
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
            items = parse_memory_items(raw)
            if not items:
                raise RuntimeError("teacher output did not contain any memory items")
            rec = _build_merged_record(cluster_rows, items)
            rec["raw_teacher_output"] = raw
            out_rows.append(rec)
        except Exception as e:
            failures += 1
            if args.fail_on_error:
                raise
            print(f"[refine-memory] cluster size={len(cluster_rows)} failed: {e}", file=sys.stderr)
            out_rows.extend(cluster_rows)

    write_jsonl(args.output_memory_bank, out_rows)
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
        f"[refine-memory] input={len(rows)} clusters={len(clusters)} output={len(out_rows)} failures={failures}",
        file=sys.stderr,
    )
    return 0


def build_refine_memory_parser(sub: Any) -> None:
    p = sub.add_parser("refine-memory", help="LLM-based memory evolution by clustering and rewriting existing memories")
    p.add_argument("--memory-bank", required=True)
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
    p.add_argument("--cluster-similarity-threshold", type=float, default=0.92)
    p.add_argument("--refine-singletons", action="store_true")
    p.add_argument("--fail-on-error", action="store_true")
    p.set_defaults(_run=run_refine_memory)

