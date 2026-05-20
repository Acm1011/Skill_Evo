"""Prepare jsonl/parquet with retrieved ReasoningBank memories injected into prompts."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .retrieve_memory import retrieve_records
from .text_utils import topic_slug


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_skill_use_template() -> str:
    return (_prompt_dir() / "skill_use_math.txt").read_text(encoding="utf-8")


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
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def extract_topic(row: Dict[str, Any]) -> Optional[str]:
    for key in ("topic",):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        val = extra.get("topic")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_idx(row: Dict[str, Any], line_no: int) -> Any:
    if row.get("idx") is not None:
        return row.get("idx")
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("idx") is not None:
        return extra.get("idx")
    return line_no


def extract_ground_truth(row: Dict[str, Any]) -> Any:
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if gt is not None:
            return gt
    if row.get("gt") is not None:
        return row.get("gt")
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in ("answer", "solution"):
            val = extra.get(key)
            if val is not None:
                return val
    return ""


def format_memory_prompt(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No relevant memories found."
    parts: List[str] = []
    for i, row in enumerate(rows, start=1):
        parts.append(f"### Relevant Memory {i}")
        parts.append(f"- Topic: {row.get('topic_key')}")
        parts.append(f"- Status: {row.get('status')}")
        for j, item in enumerate(row.get("memory_items") or [], start=1):
            parts.extend(
                [
                    f"# Memory Item {j}",
                    f"## Title {item.get('title', '')}",
                    f"## Description {item.get('description', '')}",
                    f"## Content {item.get('content', '')}",
                ]
            )
        parts.append("")
    return "\n".join(parts).strip()


def run_prepare_prompt_data(args: argparse.Namespace) -> int:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("prepare-prompt-data 需要 pandas、pyarrow: pip install pandas pyarrow") from e

    from .io_utils import read_jsonl

    memory_rows = read_jsonl(args.memory_bank)
    embedding_rows = read_jsonl(args.embeddings)
    template = load_skill_use_template()

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
            gt = extract_ground_truth(raw)
            retrieved = retrieve_records(
                question=problem,
                memory_rows=memory_rows,
                embedding_rows=embedding_rows,
                top_k=args.top_k,
                backend=args.embed_backend,
                base_url=args.embed_base_url,
                api_key=args.embed_api_key,
                model=args.embed_model,
                timeout=args.timeout,
                hash_dim=args.hash_dim,
                query_topic=topic,
                topic_bonus=args.topic_bonus,
            )
            retrieved_rows = [row for row, _ in retrieved]
            memory_text = format_memory_prompt(retrieved_rows)
            try:
                user_content = template.format(skill=memory_text, question=problem)
            except Exception as e:
                print(f"[prepare-prompt-data] skip line {line_no}: template failed: {e}", file=sys.stderr)
                skipped += 1
                continue

            prompt = [{"role": "user", "content": user_content}]
            extra = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
            ex = dict(extra)
            ex["problem"] = problem
            ex["topic"] = topic
            ex["topic_key"] = topic_key
            ex["idx"] = extract_idx(raw, line_no)
            ex["retrieval_mode"] = "reasoningbank"
            ex["top_k_memory"] = args.top_k
            ex["retrieved_memory_ids"] = [str(r.get("memory_id") or "") for r in retrieved_rows]
            ex["retrieved_memory_count"] = len(retrieved_rows)

            rec: Dict[str, Any] = {
                "prompt": prompt,
                "reward_model": {"ground_truth": gt},
                "data_source": args.data_source,
                "extra_info": ex,
            }
            if args.keep_raw_prompt and isinstance(raw.get("prompt"), list):
                rec["original_prompt"] = copy.deepcopy(raw["prompt"])
            if args.keep_raw_row:
                rec["raw_row"] = copy.deepcopy(raw)
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
    p = sub.add_parser("prepare-prompt-data", help="Input jsonl + retrieved memories -> output jsonl/parquet")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--embed-backend", default="hash", choices=["hash", "openai"])
    p.add_argument("--embed-base-url", default="")
    p.add_argument("--embed-api-key", default="")
    p.add_argument("--embed-model", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--hash-dim", type=int, default=256)
    p.add_argument("--topic-bonus", type=float, default=0.05)
    p.add_argument("--data-source", default="ReasoningBankMath")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--keep-raw-prompt", action="store_true")
    p.add_argument("--keep-raw-row", action="store_true")
    p.set_defaults(_run=run_prepare_prompt_data)

