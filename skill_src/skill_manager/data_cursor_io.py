# -*- coding: utf-8 -*-
"""data_cursor.txt：单行整数，与 ``train_cursor_state.json`` 的 ``cursor`` 同义（下一跳起始记录下标，0-based）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_CURSOR_FILENAME = "data_cursor.txt"


def read_data_cursor(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        first = path.read_text(encoding="utf-8").strip().split()
        if not first:
            return 0
        return int(first[0])
    except (OSError, ValueError):
        return 0


def write_data_cursor(path: Path, cursor: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(int(cursor)) + "\n", encoding="utf-8")


def read_jsonl_slice(
    data_path: str | Path,
    start: int,
    n: int,
) -> tuple[list[dict[str, Any]], int]:
    """
    从单文件取 ``[start, start+n)`` 条**非空行**记录（与
    ``solver_offline_driver.load_records_from_files`` 对 jsonl 的计数一致）；
    parquet 则按行号对应 ``df`` 中连续 ``n`` 行。

    若剩余不足 n 条则抛 ``ValueError``（严格）。"""
    p = Path(data_path)
    if n <= 0:
        raise ValueError(f"read_jsonl_slice: n 须为正, got {n}")
    if start < 0:
        raise ValueError(f"read_jsonl_slice: start 须 >=0, got start={start}")
    suf = p.suffix.lower()
    if suf in (".parquet", ".pq"):
        return _read_parquet_slice(p, start, n)
    return _read_jsonl_lines(p, start, n)


def _read_jsonl_lines(
    p: Path, start: int, n: int
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    pos = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if pos < start:
                pos += 1
                continue
            if len(out) < n:
                d = json.loads(line)
                if not isinstance(d, dict):
                    raise ValueError(f"read_jsonl_slice: pos={pos} 非 JSON 对象")
                out.append(d)
            pos += 1
            if len(out) >= n:
                break
    if len(out) < n:
        raise ValueError(
            f"read_jsonl_slice: 需要 {n} 条自 start={start}，只得到 {len(out)} 条"
            f"（{p} 语料已用尽或行数不足）"
        )
    return out, start + n


def _read_parquet_slice(
    p: Path, start: int, n: int
) -> tuple[list[dict[str, Any]], int]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        raise ImportError("read_parquet 需要 pandas: pip install pandas pyarrow") from e
    df = pd.read_parquet(p)
    total = len(df)
    if start + n > total:
        raise ValueError(
            f"read_jsonl_slice(parquet): 需要 {n} 条自 start={start}，但 total_n={total}"
        )
    rows: list[dict[str, Any]] = []
    for j in range(n):
        i = start + j
        row = df.iloc[i]
        d: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer, np.bool_)):
                d[k] = v.item() if hasattr(v, "item") else v
            else:
                d[k] = v
        rows.append(d)
    return rows, start + n
