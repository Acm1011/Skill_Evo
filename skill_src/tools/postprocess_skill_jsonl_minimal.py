#!/usr/bin/env python3
"""Post-process existing *_skill.jsonl to keep only 4 fields in SKILL block."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SKILL_BEGIN = "SKILL:"
QUESTION_BEGIN = "\nQuestion:"
SKILL_SEP = "\n\n---\n\n"
KEEP_KEYS = ("skill name", "problem type", "key insight", "method")


def _trim_skill_block(block: str) -> str:
    parts = [p.strip() for p in block.split(SKILL_SEP) if p.strip()]
    out_parts: list[str] = []
    for p in parts:
        try:
            d = json.loads(p)
        except Exception:
            # Keep original segment if parsing fails.
            out_parts.append(p)
            continue
        if not isinstance(d, dict):
            out_parts.append(p)
            continue
        trimmed = {k: d.get(k, "") for k in KEEP_KEYS}
        out_parts.append(json.dumps(trimmed, ensure_ascii=False))
    return SKILL_SEP.join(out_parts)


def _rewrite_user_content(content: str) -> str:
    i = content.find(SKILL_BEGIN)
    if i < 0:
        return content
    j = content.find(QUESTION_BEGIN, i)
    if j < 0:
        return content
    head = content[: i + len(SKILL_BEGIN)]
    block = content[i + len(SKILL_BEGIN) : j].strip()
    tail = content[j:]
    return f"{head} {_trim_skill_block(block)}{tail}"


def _process_record(rec: dict[str, Any]) -> bool:
    prompt = rec.get("prompt")
    if not isinstance(prompt, list):
        return False
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                new_content = _rewrite_user_content(content)
                if new_content != content:
                    msg["content"] = new_content
                    return True
            return False
    return False


def _process_jsonl(path: Path, out_path: Path) -> tuple[int, int]:
    total = 0
    changed = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"[postprocess] {path.name} skip line {line_no}: {e}", file=sys.stderr)
                continue
            if not isinstance(rec, dict):
                continue
            if _process_record(rec):
                changed += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return changed, total


def _write_parquet_from_jsonl(jsonl_path: Path, parquet_path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("write parquet requires pandas and pyarrow") from e

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            d = json.loads(s)
            if isinstance(d, dict):
                rows.append(d)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Existing *_skill.jsonl paths")
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input files in place (default false)",
    )
    p.add_argument(
        "--backup-suffix",
        default=".bak_before_minimal",
        help="Backup suffix used when --inplace is set",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Output directory when not in-place; keeps original filenames",
    )
    p.add_argument(
        "--write-parquet",
        action="store_true",
        help="Also write parquet with same stem as output jsonl",
    )
    args = p.parse_args()

    inputs = [Path(x) for x in args.inputs]
    for ip in inputs:
        if not ip.is_file():
            print(f"[postprocess] input not found: {ip}", file=sys.stderr)
            return 1

    if not args.inplace and not args.output_dir:
        print("[postprocess] use --inplace or set --output-dir", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir) if args.output_dir else None
    total_changed = 0
    total_rows = 0
    for ip in inputs:
        if args.inplace:
            tmp_out = ip.with_suffix(ip.suffix + ".tmp_minimal")
            changed, rows = _process_jsonl(ip, tmp_out)
            bak = Path(str(ip) + args.backup_suffix)
            shutil.copy2(ip, bak)
            tmp_out.replace(ip)
            print(f"[postprocess] wrote {ip} (changed {changed}/{rows}), backup={bak}")
            if args.write_parquet:
                pq = ip.with_suffix(".parquet")
                _write_parquet_from_jsonl(ip, pq)
                print(f"[postprocess] wrote {pq}")
        else:
            assert out_dir is not None
            out_path = out_dir / ip.name
            changed, rows = _process_jsonl(ip, out_path)
            print(f"[postprocess] wrote {out_path} (changed {changed}/{rows})")
            if args.write_parquet:
                pq = (out_dir / ip.name).with_suffix(".parquet")
                _write_parquet_from_jsonl(out_path, pq)
                print(f"[postprocess] wrote {pq}")
        total_changed += changed
        total_rows += rows
    print(f"[postprocess] done. changed {total_changed}/{total_rows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
