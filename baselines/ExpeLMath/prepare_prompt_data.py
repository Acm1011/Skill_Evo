"""Prepare jsonl/parquet with retriever-ranked ExpeL memories injected into prompts."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from baselines.ReasoningBankMath.io_utils import read_jsonl

from .retrieve_memory import format_retrieved_prompt
from .text_utils import normalize_space, topic_slug


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_solve_template() -> str:
    return (_prompt_dir() / "solve_with_memory.txt").read_text(encoding="utf-8")


def extract_problem(row: Dict[str, Any]) -> Optional[str]:
    for key in ("problem", "question", "raw_question"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in ("problem", "question"):
            val = extra.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def extract_topic(row: Dict[str, Any]) -> Optional[str]:
    val = row.get("topic")
    if isinstance(val, str) and val.strip():
        return val.strip()
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        val = extra.get("topic")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def build_candidate_text(row: Dict[str, Any]) -> str:
    text = normalize_space(row.get("embedding_text"))
    if text:
        return text
    parts: List[str] = []
    for key in ("query", "topic", "status", "memory_type", "raw_rule"):
        val = normalize_space(row.get(key))
        if val:
            parts.append(val)
    for item in row.get("memory_items") or []:
        if not isinstance(item, dict):
            continue
        for key in ("title", "description", "content"):
            val = normalize_space(item.get(key))
            if val:
                parts.append(val)
    return "\n".join(parts).strip()


def build_memory_candidates(memory_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in memory_rows:
        mid = str(row.get("memory_id") or "").strip()
        if not mid:
            continue
        text = build_candidate_text(row)
        if not text:
            continue
        utility = float(row.get("duplicate_count") or 0.0)
        payload = dict(row)
        payload["_retrieval_id"] = mid
        out.append(
            {
                "id": mid,
                "problem_type": text,
                "utility": utility,
                "_item": payload,
            }
        )
    return out


def sync_docs_to_retriever(memory_rows: List[Dict[str, Any]], retriever_url: str) -> Dict[str, Any]:
    candidates = build_memory_candidates(memory_rows)
    items = [{"id": c["id"], "text": c["problem_type"]} for c in candidates]
    body = json.dumps({"items": items}).encode("utf-8")
    req = urlrequest.Request(
        f"{retriever_url.rstrip('/')}/docs/replace",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"/docs/replace failed: {data}")
    return data


def _post_rank(
    *,
    retriever_url: str,
    question: str,
    candidates: List[Dict[str, Any]],
    mode: str,
    retrieve_lambda: float,
    top_k: int,
) -> List[int]:
    payload = {
        "question": question,
        "candidates": [
            {
                "id": c["id"],
                "problem_type": c["problem_type"],
                "utility": c["utility"],
            }
            for c in candidates
        ],
        "mode": mode,
        "retrieve_lambda": retrieve_lambda,
        "top_k": top_k,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{retriever_url.rstrip('/')}/rank",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        raise RuntimeError(f"retriever request failed: {e}") from e
    if not data.get("ok"):
        raise RuntimeError(f"retriever error: {data.get('error')}")
    ranked = data.get("ranked_indices")
    if not isinstance(ranked, list):
        raise RuntimeError("retriever error: missing ranked_indices")
    out: List[int] = []
    for x in ranked:
        if not isinstance(x, int):
            raise RuntimeError("retriever error: ranked_indices must contain ints")
        out.append(x)
    return out


def retrieve_memories(
    *,
    question: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    retriever_url: str,
    mode: str,
    retrieve_lambda: float,
    topic_key: str,
) -> List[Dict[str, Any]]:
    if top_k <= 0 or not candidates:
        return []
    topic_candidates = [c for c in candidates if str(c["_item"].get("topic_key") or "") == topic_key]
    ranked_rows: List[Dict[str, Any]] = []
    seen_ids = set()
    for pool in (topic_candidates, candidates):
        if not pool:
            continue
        ranked_indices = _post_rank(
            retriever_url=retriever_url,
            question=question,
            candidates=pool,
            mode=mode,
            retrieve_lambda=retrieve_lambda,
            top_k=top_k,
        )
        for idx in ranked_indices:
            if idx < 0 or idx >= len(pool):
                raise RuntimeError(f"retriever returned out-of-range index: {idx}")
            item = pool[idx]["_item"]
            rid = str(item.get("_retrieval_id") or item.get("memory_id") or "")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                ranked_rows.append(item)
            if len(ranked_rows) >= top_k:
                return ranked_rows
    return ranked_rows[:top_k]


def replace_prompt_content(row: Dict[str, Any], new_content: str) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    prompt = out.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and "content" in msg and msg.get("role") == "user":
                msg["content"] = new_content
                return out
        for msg in prompt:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = new_content
                return out
        raise ValueError("prompt list has no content field to replace")
    if isinstance(prompt, str):
        out["prompt"] = new_content
        return out
    raise ValueError("prompt must be a list or string")


def run_prepare_prompt_data(args: argparse.Namespace) -> int:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("prepare-prompt-data requires pandas and pyarrow") from e

    memory_rows = read_jsonl(args.memory_bank)
    candidates = build_memory_candidates(memory_rows)
    template = load_solve_template()
    rows_out: List[Dict[str, Any]] = []
    skipped = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < args.start:
                continue
            if args.end is not None and line_no >= args.end:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[prepare-prompt-data] skip line {line_no}: bad json: {e}", file=sys.stderr)
                skipped += 1
                continue
            if not isinstance(raw, dict):
                print(f"[prepare-prompt-data] skip line {line_no}: not a JSON object", file=sys.stderr)
                skipped += 1
                continue

            problem = extract_problem(raw)
            if not problem:
                print(f"[prepare-prompt-data] skip line {line_no}: no problem", file=sys.stderr)
                skipped += 1
                continue
            topic = extract_topic(raw)
            topic_key = topic_slug(topic)
            try:
                retrieved_rows = retrieve_memories(
                    question=problem,
                    candidates=candidates,
                    top_k=args.top_k,
                    retriever_url=args.retriever_url,
                    mode=args.mode,
                    retrieve_lambda=args.retrieve_lambda,
                    topic_key=topic_key,
                )
            except Exception as e:
                if args.fail_on_retrieve_error:
                    raise SystemExit(f"prepare-prompt-data: retrieve failed at line {line_no}: {e}") from e
                print(f"[prepare-prompt-data] skip line {line_no}: retrieve failed: {e}", file=sys.stderr)
                skipped += 1
                continue

            retrieved_context = format_retrieved_prompt(retrieved_rows) or "No retrieved insights."
            try:
                new_content = template.format(retrieved_context=retrieved_context, question=problem)
            except Exception as e:
                print(f"[prepare-prompt-data] skip line {line_no}: template failed: {e}", file=sys.stderr)
                skipped += 1
                continue

            try:
                rec = replace_prompt_content(raw, new_content)
            except Exception as e:
                print(f"[prepare-prompt-data] skip line {line_no}: prompt replace failed: {e}", file=sys.stderr)
                skipped += 1
                continue
            rows_out.append(rec)

    if not rows_out:
        print("[prepare-prompt-data] no rows", file=sys.stderr)
        return 1

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in rows_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[prepare-prompt-data] jsonl={len(rows_out)} -> {out_jsonl}", file=sys.stderr)

    out_parquet = Path(args.output_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_parquet(out_parquet, index=False)
    print(f"[prepare-prompt-data] parquet={len(rows_out)} -> {out_parquet}", file=sys.stderr)
    print(f"[prepare-prompt-data] skipped={skipped}", file=sys.stderr)
    return 0


def build_prepare_prompt_data_parser(sub: Any) -> None:
    p = sub.add_parser("prepare-prompt-data", help="Input jsonl + retrieved ExpeL memories -> output jsonl/parquet")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--retriever-url", default="http://127.0.0.1:8766")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--mode", default="embedding", choices=["embedding", "hybrid"])
    p.add_argument("--retrieve-lambda", type=float, default=0.5)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--fail-on-retrieve-error", action="store_true")
    p.set_defaults(_run=run_prepare_prompt_data)
