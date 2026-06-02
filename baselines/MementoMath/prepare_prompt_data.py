"""Prepare jsonl/parquet with parametric-retriever-ranked Memento skills injected into prompts."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    val = row.get("topic")
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


def extract_data_source(row: Dict[str, Any], default: str) -> str:
    val = row.get("data_source")
    if isinstance(val, str) and val.strip():
        return val.strip()
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        val = extra.get("data_source")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return default


def format_skill_prompt(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No relevant skills found."
    parts: List[str] = []
    for i, row in enumerate(rows, start=1):
        parts.append(f"### Relevant Skill {i}")
        parts.append(f"- Case Label: {row.get('case_label')}")
        parts.append(f"- Topic: {row.get('topic_key')}")
        takeaway = str(row.get("takeaway") or "").strip()
        if takeaway:
            parts.append(f"- Takeaway: {takeaway}")
        for step in row.get("plan_steps") or []:
            sid = step.get("id")
            desc = step.get("description")
            parts.append(f"{sid}. {desc}" if sid is not None else f"- {desc}")
        parts.append("")
    return "\n".join(parts).strip()


def run_prepare_prompt_data(args: argparse.Namespace) -> int:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("prepare-prompt-data 需要 pandas、pyarrow: pip install pandas pyarrow") from e

    from .io_utils import read_jsonl
    from .parametric_memory import CaseRetriever, build_icl_text

    memory_rows = read_jsonl(args.memory_bank)
    pool_rows = read_jsonl(args.case_pool)
    template = load_skill_use_template()
    retriever = CaseRetriever(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device or None,
    )

    icl_pool: List[str] = []
    metadata: List[Dict[str, Any]] = []
    by_key: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for mem, pool in zip(memory_rows, pool_rows):
        case = pool.get("case")
        plan = pool.get("plan")
        icl_pool.append(build_icl_text(case, plan))
        metadata.append(
            {
                "case": case,
                "plan": plan,
                "case_label": pool.get("case_label", "unknown"),
                "topic_key": mem.get("topic_key"),
                "memory_id": mem.get("memory_id"),
            }
        )
        key = (
            str(pool.get("case") or ""),
            str(pool.get("plan") or ""),
            str(pool.get("case_label") or ""),
        )
        by_key[key] = mem

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
            gt = extract_ground_truth(raw)
            data_source = extract_data_source(raw, args.data_source)

            try:
                ranked = retriever.retrieve(problem, icl_pool, metadata)
            except Exception as e:
                if args.fail_on_retrieve_error:
                    raise SystemExit(f"prepare-prompt-data: retrieve failed at line {line_no}: {e}") from e
                print(f"[prepare-prompt-data] skip line {line_no}: retrieve failed: {e}", file=sys.stderr)
                skipped += 1
                continue
            ranked.sort(key=lambda x: x["score"], reverse=True)
            picked: List[Dict[str, Any]] = []
            seen = set()
            for item in ranked:
                key = (
                    str(item.get("case") or ""),
                    str(item.get("plan") or ""),
                    str(item.get("case_label") or ""),
                )
                mem = by_key.get(key)
                if mem is None:
                    continue
                mid = str(mem.get("memory_id") or "")
                if mid and mid in seen:
                    continue
                seen.add(mid)
                mem_copy = dict(mem)
                mem_copy["_score"] = float(item.get("score") or 0.0)
                picked.append(mem_copy)
                if len(picked) >= args.top_k:
                    break

            skill_text = format_skill_prompt(picked)
            try:
                user_content = template.format(skill=skill_text, question=problem)
            except Exception as e:
                print(f"[prepare-prompt-data] skip line {line_no}: template failed: {e}", file=sys.stderr)
                skipped += 1
                continue

            prompt = [{"role": "user", "content": user_content}]
            extra = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
            ex = dict(extra)
            ex["problem"] = problem
            ex["topic"] = topic
            ex["idx"] = extract_idx(raw, line_no)
            ex["top_k_skill"] = args.top_k
            ex["retriever_model_path"] = args.model_path
            ex["retrieved_memory_ids"] = [str(r.get("memory_id") or "") for r in picked]
            ex["retrieved_memory_count"] = len(picked)

            rec: Dict[str, Any] = {
                "problem": problem,
                "ground_truth": gt,
                "prompt": prompt,
                "reward_model": {"ground_truth": gt},
                "data_source": data_source,
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
    p = sub.add_parser("prepare-prompt-data", help="Input jsonl + retrieved skills -> output jsonl/parquet")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--case-pool", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default="princeton-nlp/sup-simcse-roberta-base")
    p.add_argument("--device", default="")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--data-source", default="MementoMath")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--keep-raw-prompt", action="store_true")
    p.add_argument("--keep-raw-row", action="store_true")
    p.add_argument("--fail-on-retrieve-error", action="store_true")
    p.set_defaults(_run=run_prepare_prompt_data)
