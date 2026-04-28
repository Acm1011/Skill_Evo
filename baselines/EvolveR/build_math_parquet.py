#!/usr/bin/env python3
"""Build train/val parquet for EvolveR math RL (schema aligned with baselines/SkillRL)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_problem(raw: Dict[str, Any]) -> Optional[str]:
    ei = raw.get("extra_info")
    if isinstance(ei, dict):
        prob = ei.get("problem")
        if isinstance(prob, str) and prob.strip():
            return prob.strip()
    prompt = raw.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def _extract_ground_truth(raw: Dict[str, Any]) -> Optional[str]:
    rm = raw.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if isinstance(gt, str) and gt.strip():
            return gt.strip()
        if isinstance(gt, list) and gt and isinstance(gt[0], str):
            return gt[0].strip()
    ei = raw.get("extra_info")
    if isinstance(ei, dict):
        a = ei.get("answer") or ei.get("solution")
        if isinstance(a, str) and a.strip():
            return a.strip()
    return None


def _extract_meta(raw: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    ei = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
    idx = raw.get("idx", line_no)
    if idx is None and isinstance(ei, dict):
        rqi = ei.get("raw_q_info") or {}
        if isinstance(rqi, dict):
            idx = rqi.get("idx", line_no)
    topic = ei.get("topic") if isinstance(ei, dict) else None
    return {"idx": idx if idx is not None else line_no, "topic": topic}


def main() -> int:
    p = argparse.ArgumentParser(description="DeepMath jsonl -> EvolveR verl parquet")
    p.add_argument("--deepmath-jsonl", required=True, help="DeepMath-103K style jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--val-ratio", type=float, default=0.0)
    p.add_argument("--output-train", required=True)
    p.add_argument("--output-val", default="")
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        print("Need pandas, pyarrow: pip install pandas pyarrow", file=sys.stderr)
        raise SystemExit(1) from e

    if args.val_ratio > 0 and not (args.output_val or "").strip():
        print("Set --output-val when --val-ratio > 0", file=sys.stderr)
        return 1

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
            problem = _extract_problem(raw)
            if not problem:
                skipped += 1
                continue
            gt = _extract_ground_truth(raw)
            meta = _extract_meta(raw, line_no)
            extra = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
            ex = dict(extra)
            ex["problem"] = problem
            ex["topic"] = meta.get("topic")
            ex["idx"] = meta.get("idx", line_no)
            ex["index"] = meta.get("idx", line_no)

            gts = (gt or "").strip()
            rec: Dict[str, Any] = {
                "prompt": [{"role": "user", "content": problem}],
                "reward_model": {"ground_truth": gts},
                "data_source": "DeepMath-103K",
                "extra_info": ex,
                # Required by EvolveR verl ray_trainer / experience trajectories
                "question": problem,
                "golden_answers": [gts] if gts else [],
                "ability": "math",
            }
            rows.append(rec)

    if not rows:
        print("No rows built", file=sys.stderr)
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
    print(f"train={len(train_rows)} -> {out_train}", file=sys.stderr)

    if val_rows:
        out_val = Path(args.output_val)
        out_val.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(val_rows).to_parquet(out_val, index=False)
        print(f"val={len(val_rows)} -> {out_val}", file=sys.stderr)

    print(f"skipped={skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
