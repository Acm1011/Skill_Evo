#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from baselines.SkillRL.text_utils import topic_slug


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise RuntimeError(f"JSON parse failed: {path}:{line_no}: {e}") from e


def _extract_topic(row: Dict[str, Any]) -> Optional[str]:
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        t = ei.get("topic")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return None


def _extract_difficulty(row: Dict[str, Any]) -> Any:
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        return ei.get("difficulty")
    return None


def _extract_problem(row: Dict[str, Any]) -> str:
    for k in ("question", "raw_question"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        for k in ("problem", "question"):
            v = ei.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _extract_ground_truth(row: Dict[str, Any]) -> Optional[str]:
    gt = row.get("gt")
    if isinstance(gt, list) and gt:
        x = gt[0]
        return str(x).strip() if x is not None else None
    if isinstance(gt, str) and gt.strip():
        return gt.strip()
    return None


def _to_bool_or_none(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    return None


def _resolve_input_jsonl(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        cand = p / "train_data.jsonl"
        if cand.is_file():
            return [cand]
    return []


def convert(inputs: List[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_count = 0
    src_count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for src in inputs:
            for row_i, row in enumerate(_iter_jsonl(src)):
                src_count += 1
                responses = row.get("responses")
                if not isinstance(responses, list) or not responses:
                    continue
                is_right = row.get("is_right")
                if not isinstance(is_right, list):
                    is_right = []
                problem = _extract_problem(row)
                topic = _extract_topic(row)
                gt = _extract_ground_truth(row)
                idx = row.get("idx", row_i)
                diff = _extract_difficulty(row)

                for j, resp in enumerate(responses):
                    if not isinstance(resp, str) or not resp.strip():
                        continue
                    traj = {
                        "idx": idx,
                        "line_idx": row_i,
                        "problem": problem,
                        "topic": topic,
                        "topic_key": topic_slug(topic),
                        "difficulty": diff,
                        "student_response": resp,
                        "is_correct": _to_bool_or_none(is_right[j] if j < len(is_right) else None),
                        "ground_truth": gt,
                        "source_file": str(src),
                        "source_response_idx": j,
                    }
                    fout.write(json.dumps(traj, ensure_ascii=False) + "\n")
                    out_count += 1
    print(f"[convert] source_rows={src_count} trajectories={out_count} output={output_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert Synthesizer merged train_data.jsonl into SkillRL teacher_distill trajectories.jsonl"
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input dirs/files. Dir should contain train_data.jsonl",
    )
    p.add_argument("--output", required=True, help="Output trajectories.jsonl")
    args = p.parse_args()

    resolved: List[Path] = []
    for x in args.inputs:
        resolved.extend(_resolve_input_jsonl(Path(x)))
    if not resolved:
        raise SystemExit("No valid input jsonl found from --inputs")
    convert(resolved, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
