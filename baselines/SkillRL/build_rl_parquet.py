"""Build verl-style parquet for math RL (skills injected in user message)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .layered_skill_bank import LayeredSkillBank
from .student_rollout import extract_ground_truth, extract_meta, extract_problem
from .text_utils import topic_slug


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_skill_use_template() -> str:
    return (_prompt_dir() / "skill_use_math.txt").read_text(encoding="utf-8")


def _normalize_ground_truth(gt: Optional[str]) -> Any:
    if gt is None:
        return ""
    return gt.strip()


def _validate_rl_args(args: argparse.Namespace) -> None:
    if args.val_ratio > 0 and not (args.output_val or "").strip():
        raise SystemExit("build-rl-parquet: set --output-val when --val-ratio > 0")


def run_build_rl_parquet(args: argparse.Namespace) -> int:
    _validate_rl_args(args)
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("build-rl-parquet 需要 pandas、pyarrow: pip install pandas pyarrow") from e

    bank = LayeredSkillBank.from_path(args.skills_json)
    template = load_skill_use_template()

    rows: List[Dict[str, Any]] = []
    skipped = 0

    with open(args.deepmath_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < args.start:
                continue
            if line_no >= args.end:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            problem = extract_problem(raw)
            if not problem:
                skipped += 1
                continue
            meta = extract_meta(raw, line_no)
            topic = meta.get("topic")
            gt = extract_ground_truth(raw)
            skill_text = bank.format_for_prompt(
                bank.retrieve(task_description=problem, topic=topic, top_k=args.top_k)
            )
            try:
                user_content = template.format(skill=skill_text, question=problem)
            except Exception as e:
                print(f"[build-rl-parquet] skip line {line_no}: {e}", file=sys.stderr)
                skipped += 1
                continue

            prompt = [{"role": "user", "content": user_content}]
            rm_gt = _normalize_ground_truth(gt)
            extra = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
            ex = dict(extra)
            ex["problem"] = problem
            ex["topic"] = topic
            ex["topic_key"] = topic_slug(topic)
            ex["idx"] = meta.get("idx", line_no)

            rec: Dict[str, Any] = {
                "prompt": prompt,
                "reward_model": {"ground_truth": rm_gt},
                "data_source": "DeepMath-103K",
                "extra_info": ex,
            }
            if args.keep_raw_prompt and isinstance(raw.get("prompt"), list):
                rec["original_prompt"] = copy.deepcopy(raw["prompt"])
            rows.append(rec)

    if not rows:
        print("[build-rl-parquet] no rows", file=sys.stderr)
        return 1

    if args.val_ratio > 0 and args.val_ratio < 1:
        n = len(rows)
        n_val = max(1, int(n * args.val_ratio))
        train_rows, val_rows = rows[:-n_val], rows[-n_val:]
    else:
        train_rows, val_rows = rows, []

    out_train = Path(args.output_train)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(out_train, index=False)
    print(f"[build-rl-parquet] train={len(train_rows)} -> {out_train}", file=sys.stderr)

    if val_rows:
        out_val = Path(args.output_val)
        out_val.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(val_rows).to_parquet(out_val, index=False)
        print(f"[build-rl-parquet] val={len(val_rows)} -> {out_val}", file=sys.stderr)

    print(f"[build-rl-parquet] skipped={skipped}", file=sys.stderr)
    return 0


def build_build_rl_parquet_parser(sub: Any) -> None:
    p = sub.add_parser("build-rl-parquet", help="DeepMath + skills -> train/val parquet for verl")
    p.add_argument("--deepmath-jsonl", required=True)
    p.add_argument("--skills-json", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--val-ratio", type=float, default=0.0, help="e.g. 0.05; 0 = all train")
    p.add_argument("--output-train", required=True)
    p.add_argument("--output-val", default="", help="Required if val-ratio > 0")
    p.add_argument("--keep-raw-prompt", action="store_true")
    p.set_defaults(_run=run_build_rl_parquet)
