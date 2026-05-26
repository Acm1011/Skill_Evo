#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MMLU_COLUMNS = [
    "question_id",
    "question",
    "options",
    "answer",
    "answer_index",
    "cot_content",
    "category",
    "src",
]

BBEH_COLUMNS = ["question", "answer", "task"]

SUPERGPQA_COLUMNS = [
    "uuid",
    "question",
    "options",
    "answer",
    "answer_letter",
    "discipline",
    "field",
    "subfield",
    "difficulty",
    "is_calculation",
]


def _fix_jsonl(path: Path, expected_columns: list[str]) -> int:
    kept_lines: list[str] = []
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fixed = {k: obj.get(k) for k in expected_columns}
            kept_lines.append(json.dumps(fixed, ensure_ascii=False))
            count += 1
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    tmp.replace(path)
    return count


def _fix_parquet(path: Path, expected_columns: list[str]) -> int:
    df = pd.read_parquet(path)
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df[expected_columns]
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix generated general skill benchmark data schema in-place")
    parser.add_argument("--data-dir", required=True, help="Root dir containing bbeh-eval/MMLU-Pro/SuperGPQA")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data dir not found: {root}")

    bbeh_path = root / "bbeh-eval" / "train.jsonl"
    mmlu_path = root / "MMLU-Pro" / "data" / "test-00000-of-00001.parquet"
    supergpqa_path = root / "SuperGPQA" / "SuperGPQA-all.jsonl"

    if bbeh_path.exists():
        count = _fix_jsonl(bbeh_path, BBEH_COLUMNS)
        print(f"fixed {bbeh_path} rows={count}")
    else:
        print(f"skip missing {bbeh_path}")

    if mmlu_path.exists():
        count = _fix_parquet(mmlu_path, MMLU_COLUMNS)
        print(f"fixed {mmlu_path} rows={count}")
    else:
        print(f"skip missing {mmlu_path}")

    if supergpqa_path.exists():
        count = _fix_jsonl(supergpqa_path, SUPERGPQA_COLUMNS)
        print(f"fixed {supergpqa_path} rows={count}")
    else:
        print(f"skip missing {supergpqa_path}")

    print(f"done: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
