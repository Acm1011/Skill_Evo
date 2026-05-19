from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


BASELINE_DIR = Path(__file__).resolve().parents[1]
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import build_math_parquet


def test_build_math_parquet_outputs_expected_columns(tmp_path, monkeypatch):
    src = tmp_path / "deepmath.jsonl"
    rows = [
        {
            "data_source": "DeepMath-103K",
            "prompt": [{"role": "user", "content": "What is 1+1?"}],
            "reward_model": {"ground_truth": "2"},
            "extra_info": {"topic": "arithmetic", "difficulty": "easy"},
        },
        {
            "data_source": "DeepMath-103K",
            "extra_info": {"problem": "Solve x^2=1", "solution": "\\pm 1", "topic": "algebra"},
        },
    ]
    with src.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    out_train = tmp_path / "train.parquet"
    out_val = tmp_path / "val.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_math_parquet.py",
            "--deepmath-jsonl",
            str(src),
            "--start",
            "0",
            "--end",
            "2",
            "--val-ratio",
            "0.5",
            "--output-train",
            str(out_train),
            "--output-val",
            str(out_val),
        ],
    )

    rc = build_math_parquet.main()
    assert rc == 0

    df = pd.read_parquet(out_train)
    assert {"data_source", "prompt", "ability", "reward_model", "extra_info", "env_kwargs"} <= set(df.columns)
    env_kwargs = df.iloc[0]["env_kwargs"]
    assert env_kwargs["question"] == "What is 1+1?"
    assert env_kwargs["ground_truth"] == "2"
    assert env_kwargs["topic"] == "arithmetic"

