"""Convert DeepMath-103K jsonl into ARISE/verl-compatible parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _normalize_ground_truth(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_prompt(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = raw.get("prompt")
    if isinstance(prompt, list):
        out: List[Dict[str, str]] = []
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip() or "user"
            content = str(msg.get("content") or "").strip()
            if content:
                out.append({"role": role, "content": content})
        if out:
            return out

    extra = raw.get("extra_info")
    if isinstance(extra, dict):
        problem = str(extra.get("problem") or "").strip()
        if problem:
            return [{"role": "user", "content": problem}]
    return []


def convert(args: argparse.Namespace) -> int:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("build_deepmath_parquet 需要 pandas、pyarrow: pip install pandas pyarrow") from e

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
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(raw, dict):
                skipped += 1
                continue

            prompt = _normalize_prompt(raw)
            reward_model = raw.get("reward_model") if isinstance(raw.get("reward_model"), dict) else {}
            ground_truth = _normalize_ground_truth(reward_model.get("ground_truth"))
            if not prompt or not ground_truth:
                skipped += 1
                continue

            extra_info = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
            rows.append(
                {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": ground_truth,
                    },
                    "extra_info": {
                        **extra_info,
                        "index": int(extra_info.get("index", extra_info.get("idx", len(rows)))),
                    },
                }
            )

    if not rows:
        raise SystemExit("no valid rows converted")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(
        json.dumps(
            {
                "input": args.deepmath_jsonl,
                "output": str(output_path),
                "written_rows": len(rows),
                "skipped_rows": skipped,
                "data_source": args.data_source,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert DeepMath jsonl to ARISE-compatible parquet")
    parser.add_argument("--deepmath-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-source", default="math_dapo")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    return convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
