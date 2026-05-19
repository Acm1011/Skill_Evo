#!/usr/bin/env python3
"""Build RetroAgent-math parquet from DeepMath-style jsonl."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_problem(raw: Dict[str, Any]) -> Optional[str]:
    ei = raw.get("extra_info")
    if isinstance(ei, dict):
        problem = ei.get("problem")
        if isinstance(problem, str) and problem.strip():
            return problem.strip()
    prompt = raw.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
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
        answer = ei.get("answer") or ei.get("solution")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    return None


def _extract_meta(raw: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    ei = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
    idx = raw.get("idx", line_no)
    if idx is None and isinstance(ei, dict):
        raw_q_info = ei.get("raw_q_info") or {}
        if isinstance(raw_q_info, dict):
            idx = raw_q_info.get("idx", line_no)
    topic = ei.get("topic") if isinstance(ei, dict) else None
    difficulty = ei.get("difficulty") if isinstance(ei, dict) else None
    return {
        "idx": idx if idx is not None else line_no,
        "topic": str(topic or "").strip(),
        "difficulty": difficulty,
    }


def _placeholder_prompt() -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": "The actual mathematics problem will be provided by the environment. Respond using <answer>...</answer>.",
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepMath jsonl -> RetroAgent math parquet")
    parser.add_argument("--deepmath-jsonl", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-val", default="")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        print("Need pandas and pyarrow: pip install pandas pyarrow", file=sys.stderr)
        raise SystemExit(1) from e

    if args.val_ratio > 0 and not args.output_val.strip():
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
            gt = (_extract_ground_truth(raw) or "").strip()
            meta = _extract_meta(raw, line_no)
            extra = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
            extra_info = dict(extra)
            extra_info["problem"] = problem
            extra_info["topic"] = meta["topic"]
            extra_info["difficulty"] = meta["difficulty"]
            extra_info["index"] = meta["idx"]
            extra_info["idx"] = meta["idx"]

            rows.append(
                {
                    "data_source": str(raw.get("data_source") or "DeepMath-103K"),
                    "prompt": _placeholder_prompt(),
                    "ability": "math",
                    "reward_model": {"ground_truth": gt},
                    "extra_info": extra_info,
                    "env_kwargs": {
                        "question": problem,
                        "ground_truth": gt,
                        "data_source": str(raw.get("data_source") or "DeepMath-103K"),
                        "topic": meta["topic"],
                        "index": meta["idx"],
                    },
                }
            )

    if not rows:
        print("No rows built", file=sys.stderr)
        return 1

    if 0 < args.val_ratio < 1:
        n_val = max(1, int(len(rows) * args.val_ratio))
        train_rows = rows[:-n_val]
        val_rows = rows[-n_val:]
    else:
        train_rows = rows
        val_rows = []

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

