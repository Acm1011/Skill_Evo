#!/usr/bin/env python3
"""Prepare DeepMath prompt data with a trained SkillOpt skill injected.

This script is the downstream-data companion of ``scripts/train.py`` for the
DeepMath environment. The original SkillOpt training loop writes optimized
skills to ``<out_root>/best_skill.md`` and ``<out_root>/skills/skill_vXXXX.md``;
this script turns either skill file into DeepMath temp/greedy JSONL/Parquet data.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL line {line_no} in {path}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_problem(row: dict) -> str:
    problem = _first_text(row.get("problem"), row.get("question"), row.get("raw_question"))
    if problem:
        return problem
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    problem = _first_text(extra.get("problem"), extra.get("question"))
    if problem:
        return problem
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = _first_text(msg.get("content"))
                if content:
                    return content
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return ""


def extract_ground_truth(row: dict) -> Any:
    reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
    if reward_model.get("ground_truth") is not None:
        return reward_model.get("ground_truth")
    for key in ("ground_truth", "gt", "answer"):
        if row.get(key) is not None:
            return row.get(key)
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    for key in ("answer", "solution"):
        if extra.get(key) is not None:
            return extra.get(key)
    return ""


def extract_topic(row: dict) -> str:
    topic = _first_text(row.get("topic"), row.get("topic_key"))
    if topic:
        return topic
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return _first_text(extra.get("topic"), extra.get("topic_key")) or "general_math"


def extract_idx(row: dict, line_no: int) -> Any:
    if row.get("idx") is not None:
        return row.get("idx")
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    if extra.get("idx") is not None:
        return extra.get("idx")
    return line_no


def build_prompt(skill_text: str, problem: str) -> str:
    return (
        "You are solving a math problem. Use the following SkillOpt-optimized "
        "skill memory when it is relevant.\n\n"
        f"## Skill Memory\n{skill_text.strip()}\n\n"
        f"## Problem\n{problem}\n\n"
        "Think step by step, verify your reasoning, and put the final answer "
        "inside <answer>...</answer>."
    )


def prepare_prompt_data(
    *,
    input_jsonl: Path,
    skill_file: Path,
    output_jsonl: Path,
    output_parquet: Path | None,
    data_source: str,
    start: int,
    end: int | None,
    keep_raw_prompt: bool,
    keep_raw_row: bool,
) -> int:
    skill_text = skill_file.read_text(encoding="utf-8")
    rows_out: list[dict] = []
    skipped = 0
    for line_no, row in enumerate(read_jsonl(input_jsonl)):
        if line_no < start:
            continue
        if end is not None and line_no >= end:
            break
        problem = extract_problem(row)
        if not problem:
            skipped += 1
            continue
        gt = extract_ground_truth(row)
        topic = extract_topic(row)
        extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        ex = dict(extra)
        ex.update({
            "problem": problem,
            "topic": topic,
            "idx": extract_idx(row, line_no),
            "skillopt_skill_file": str(skill_file),
            "skillopt_skill_chars": len(skill_text),
        })
        rec = {
            "problem": problem,
            "ground_truth": gt,
            "prompt": [{"role": "user", "content": build_prompt(skill_text, problem)}],
            "reward_model": {"ground_truth": gt},
            "data_source": data_source,
            "extra_info": ex,
        }
        if keep_raw_prompt and isinstance(row.get("prompt"), list):
            rec["original_prompt"] = copy.deepcopy(row["prompt"])
        if keep_raw_row:
            rec["raw_row"] = copy.deepcopy(row)
        rows_out.append(rec)

    if not rows_out:
        print(f"[skillopt-prepare] no rows generated; skipped={skipped}", file=sys.stderr)
        return 1
    write_jsonl(output_jsonl, rows_out)
    print(f"[skillopt-prepare] jsonl={len(rows_out)} -> {output_jsonl}", file=sys.stderr)
    if output_parquet is not None:
        try:
            import pandas as pd
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows_out).to_parquet(output_parquet, index=False)
            print(f"[skillopt-prepare] parquet={len(rows_out)} -> {output_parquet}", file=sys.stderr)
        except ImportError:
            print("[skillopt-prepare] skip parquet: pandas/pyarrow is not installed", file=sys.stderr)
    print(f"[skillopt-prepare] skipped={skipped}", file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject trained SkillOpt skill into DeepMath prompt data")
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--skill-file", type=Path, required=True, help="Usually <SkillOpt out_root>/best_skill.md")
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--output-parquet", type=Path, default=None)
    p.add_argument("--data-source", default="SkillOptMath")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--keep-raw-prompt", action="store_true")
    p.add_argument("--keep-raw-row", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(prepare_prompt_data(
        input_jsonl=args.input_jsonl,
        skill_file=args.skill_file,
        output_jsonl=args.output_jsonl,
        output_parquet=args.output_parquet,
        data_source=args.data_source,
        start=args.start,
        end=args.end,
        keep_raw_prompt=args.keep_raw_prompt,
        keep_raw_row=args.keep_raw_row,
    ))


if __name__ == "__main__":
    main()
