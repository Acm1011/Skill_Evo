"""Build RL jsonl/parquet from DeepMath using ARISE library skills."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from baselines.SkillRL.prepare_rl_data import retrieve_bucket
from baselines.SkillRL.student_rollout import extract_ground_truth, extract_meta, extract_problem

from .skill_bank import AriseSkillBank, format_skill_prompt, load_skill_use_template


def run_prepare_rl_data(args: argparse.Namespace) -> int:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("prepare-rl-data 需要 pandas、pyarrow: pip install pandas pyarrow") from e

    bank = AriseSkillBank.from_path(args.library_json, include_reservoir=args.include_reservoir)
    candidates = bank.build_candidates()
    if not candidates:
        raise SystemExit(f"prepare-rl-data: no usable skills found in {args.library_json}")
    template = load_skill_use_template()

    rows: List[Dict[str, Any]] = []
    skipped = 0

    with open(args.deepmath_jsonl, "r", encoding="utf-8") as f:
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
                print(f"[arise-prepare-rl-data] skip line {line_no}: bad json: {e}", file=sys.stderr)
                skipped += 1
                continue
            if not isinstance(raw, dict):
                print(f"[arise-prepare-rl-data] skip line {line_no}: not a JSON object", file=sys.stderr)
                skipped += 1
                continue

            problem = extract_problem(raw)
            if not problem:
                print(f"[arise-prepare-rl-data] skip line {line_no}: no problem", file=sys.stderr)
                skipped += 1
                continue

            meta = extract_meta(raw, line_no)
            gt = extract_ground_truth(raw)

            try:
                retrieved_skills = retrieve_bucket(
                    question=problem,
                    candidates=candidates,
                    top_k=args.top_k,
                    retriever_url=args.retriever_url,
                    mode=args.mode,
                    retrieve_lambda=args.retrieve_lambda,
                )
            except Exception as e:
                if args.fail_on_retrieve_error:
                    raise SystemExit(f"prepare-rl-data: retrieve failed at line {line_no}: {e}") from e
                print(f"[arise-prepare-rl-data] skip line {line_no}: retrieve failed: {e}", file=sys.stderr)
                skipped += 1
                continue

            skill_text = format_skill_prompt(retrieved_skills)
            if not skill_text.strip():
                print(f"[arise-prepare-rl-data] skip line {line_no}: no retrieved skills", file=sys.stderr)
                skipped += 1
                continue

            try:
                user_content = template.format(skill=skill_text, question=problem)
            except Exception as e:
                print(f"[arise-prepare-rl-data] skip line {line_no}: template failed: {e}", file=sys.stderr)
                skipped += 1
                continue

            prompt = [{"role": "user", "content": user_content}]
            extra = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
            ex = dict(extra)
            ex["problem"] = problem
            ex["topic"] = meta.get("topic")
            ex["topic_key"] = meta.get("topic_key")
            ex["idx"] = meta.get("idx", line_no)
            ex["retrieval_mode"] = args.mode
            ex["retriever_url"] = args.retriever_url
            ex["top_k"] = args.top_k
            ex["skill_id"] = [str(s.get("skill_id") or "") for s in retrieved_skills]
            ex["retrieved_skill_ids"] = list(ex["skill_id"])
            ex["retrieved_skill_count"] = len(retrieved_skills)
            ex["skill_candidates_count"] = len(candidates)

            rec: Dict[str, Any] = {
                "prompt": prompt,
                "reward_model": {"ground_truth": gt.strip() if isinstance(gt, str) else ""},
                "data_source": args.data_source,
                "extra_info": ex,
            }
            if args.keep_raw_prompt and isinstance(raw.get("prompt"), list):
                rec["original_prompt"] = copy.deepcopy(raw["prompt"])
            rows.append(rec)

    if not rows:
        print("[arise-prepare-rl-data] no rows", file=sys.stderr)
        return 1

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[arise-prepare-rl-data] jsonl={len(rows)} -> {out_jsonl}", file=sys.stderr)

    out_parquet = Path(args.output_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    print(f"[arise-prepare-rl-data] parquet={len(rows)} -> {out_parquet}", file=sys.stderr)
    print(f"[arise-prepare-rl-data] skipped={skipped}", file=sys.stderr)
    return 0


def build_prepare_rl_data_parser(sub: Any) -> None:
    p = sub.add_parser("prepare-rl-data", help="DeepMath + ARISE library -> RL jsonl/parquet")
    p.add_argument("--deepmath-jsonl", default="/home/ycy/sdi/data/DeepMath-103K.jsonl")
    p.add_argument("--library-json", required=True, help="ARISE skill library checkpoint json")
    p.add_argument("--retriever-url", default="http://127.0.0.1:8766")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--mode", default="embedding", choices=["embedding", "hybrid"])
    p.add_argument("--retrieve-lambda", type=float, default=0.5)
    p.add_argument("--data-source", default="math_dapo", help="ARISE reward data_source tag")
    p.add_argument("--output-jsonl", default="baselines/ARISE/outputs/deepmath_arise_rl.jsonl")
    p.add_argument("--output-parquet", default="baselines/ARISE/outputs/deepmath_arise_rl.parquet")
    p.add_argument("--keep-raw-prompt", action="store_true")
    p.add_argument("--include-reservoir", action="store_true", help="Also retrieve from reservoir entries")
    p.add_argument("--fail-on-retrieve-error", action="store_true")
    p.set_defaults(_run=run_prepare_rl_data)
