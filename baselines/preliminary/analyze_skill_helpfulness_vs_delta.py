from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from baselines.ReasoningBankMath.io_utils import read_jsonl, write_jsonl


def _join_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    checkpoint_name = str(row.get("checkpoint_name") or "")
    return (
        checkpoint_name,
        str(row.get("method") or ""),
        str(row.get("teacher_backend") or ""),
        str(row.get("source_idx")),
    )


def _delta_bucket(row: Dict[str, Any]) -> str:
    delta = row.get("delta")
    if delta is None:
        return "not_evaluated"
    value = float(delta)
    if value > 0:
        return "improved"
    if value < 0:
        return "degraded"
    return "unchanged"


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        checkpoint_name = str(row.get("checkpoint_name") or "")
        buckets[(checkpoint_name, str(row["method"]), str(row["teacher_backend"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (checkpoint_name, method, teacher_backend), items in sorted(buckets.items()):
        for bucket_name in ("improved", "degraded", "unchanged", "not_evaluated"):
            subset = [row for row in items if row["delta_bucket"] == bucket_name]
            scored = [row for row in subset if row.get("expert_score") is not None]
            scores = [float(row["expert_score"]) for row in scored]
            out.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "method": method,
                    "teacher_backend": teacher_backend,
                    "delta_bucket": bucket_name,
                    "n_questions": len(subset),
                    "n_scored_questions": len(scored),
                    "mean_expert_score": _mean(scores),
                    "score_ge_4_rate": (_mean([1.0 if s >= 4.0 else 0.0 for s in scores]) if scores else None),
                }
            )
    return out


def run_analysis(args: argparse.Namespace) -> int:
    details = read_jsonl(args.details)
    expert_scores = read_jsonl(args.expert_scores)
    expert_by_key = {_join_key(row): row for row in expert_scores}

    joined_rows: List[Dict[str, Any]] = []
    for detail in details:
        key = _join_key(detail)
        expert = expert_by_key.get(key)
        joined_rows.append(
            {
                "checkpoint_name": str(detail.get("checkpoint_name") or ""),
                "source_idx": detail.get("source_idx"),
                "problem": detail.get("problem"),
                "method": detail.get("method"),
                "teacher_backend": detail.get("teacher_backend"),
                "delta": detail.get("delta"),
                "delta_bucket": _delta_bucket(detail),
                "baseline_acc": detail.get("baseline_acc"),
                "skill_acc": detail.get("skill_acc"),
                "skip_reason": detail.get("skip_reason", ""),
                "expert_score": None if expert is None else expert.get("expert_score"),
                "expert_label": "" if expert is None else expert.get("expert_label", ""),
                "expert_rationale": "" if expert is None else expert.get("expert_rationale", ""),
                "judge_status": "missing_expert_score" if expert is None else expert.get("judge_status", ""),
            }
        )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "joined_rows.jsonl", joined_rows)
    summary = _summarize(joined_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze whether expert-rated skill helpfulness correlates with delta buckets")
    parser.add_argument("--details", required=True, help="details.jsonl or cross_checkpoint_details.jsonl")
    parser.add_argument("--expert-scores", required=True, help="expert_scores.jsonl")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(run_analysis(args))


if __name__ == "__main__":
    raise SystemExit(main())
