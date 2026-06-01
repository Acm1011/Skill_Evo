from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from baselines.ReasoningBankMath.io_utils import read_jsonl, write_jsonl


def _skill_join_keys(row: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    method = str(row.get("method") or "")
    teacher_backend = str(row.get("teacher_backend") or "")
    source_idx = row.get("source_idx")
    problem = str(row.get("problem") or "").strip()
    keys: List[Tuple[str, str, str, str]] = []
    if source_idx is not None:
        keys.append((method, teacher_backend, "source_idx", str(source_idx)))
    if problem:
        keys.append((method, teacher_backend, "problem", problem))
    return keys


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    mu = _mean(values)
    assert mu is not None
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return numer / denom


def _trend_label(delta_change: float, slope: float, threshold: float) -> str:
    if abs(delta_change) < threshold and abs(slope) < threshold:
        return "stable"
    if delta_change > 0:
        return "improving"
    if delta_change < 0:
        return "degrading"
    if slope > 0:
        return "improving"
    if slope < 0:
        return "degrading"
    return "stable"


def _sign_bucket(delta: float) -> str:
    if delta > 0:
        return "positive"
    if delta < 0:
        return "negative"
    return "zero"


def _load_skill_lookup(skills_run_dir: Optional[str]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    if not skills_run_dir:
        return {}
    root = Path(skills_run_dir) / "generated_skills"
    lookup: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    if not root.is_dir():
        return lookup
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        for path in sorted(method_dir.glob("*.jsonl")):
            rows = read_jsonl(path)
            for row in rows:
                for key in _skill_join_keys(row):
                    lookup[key] = row
    return lookup


def _build_skill_trajectories(
    details: Sequence[Dict[str, Any]],
    *,
    skill_lookup: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    min_checkpoints: int,
    change_threshold: float,
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in details:
        delta = _safe_float(row.get("delta"))
        if delta is None:
            continue
        keys = _skill_join_keys(row)
        if not keys:
            continue
        buckets[keys[0]].append(row)

    trajectories: List[Dict[str, Any]] = []
    for key, rows in sorted(buckets.items()):
        ordered = sorted(
            rows,
            key=lambda r: (
                int(r.get("checkpoint_order") or 0),
                str(r.get("checkpoint_name") or ""),
            ),
        )
        if len(ordered) < min_checkpoints:
            continue
        deltas = [float(row["delta"]) for row in ordered if row.get("delta") is not None]
        if len(deltas) < min_checkpoints:
            continue
        xs = [float(int(row.get("checkpoint_order") or 0)) for row in ordered]
        ys = [float(row["delta"]) for row in ordered]
        first = ordered[0]
        last = ordered[-1]
        start_delta = float(first["delta"])
        end_delta = float(last["delta"])
        delta_change = end_delta - start_delta
        range_value = max(ys) - min(ys)
        slope = _slope(xs, ys) or 0.0
        sign_sequence = [_sign_bucket(v) for v in ys]
        sign_flip_count = sum(1 for a, b in zip(sign_sequence, sign_sequence[1:]) if a != b)
        skill_row = skill_lookup.get(key)
        trajectories.append(
            {
                "skill_id": f"{key[0]}::{key[1]}::{key[2]}::{key[3]}",
                "method": key[0],
                "teacher_backend": key[1],
                "match_key_type": key[2],
                "match_key_value": key[3],
                "source_idx": first.get("source_idx"),
                "problem": first.get("problem"),
                "skill_status": "" if skill_row is None else str(skill_row.get("status") or ""),
                "skill_text": "" if skill_row is None else str(skill_row.get("skill_text") or ""),
                "n_checkpoints": len(ordered),
                "start_checkpoint_name": str(first.get("checkpoint_name") or ""),
                "start_checkpoint_order": first.get("checkpoint_order"),
                "end_checkpoint_name": str(last.get("checkpoint_name") or ""),
                "end_checkpoint_order": last.get("checkpoint_order"),
                "start_delta": start_delta,
                "end_delta": end_delta,
                "delta_change": delta_change,
                "mean_delta": _mean(ys),
                "delta_stdev": _stdev(ys),
                "min_delta": min(ys),
                "max_delta": max(ys),
                "delta_range": range_value,
                "trend_slope": slope,
                "trend_label": _trend_label(delta_change, slope, change_threshold),
                "large_change": abs(delta_change) >= change_threshold,
                "high_variability": range_value >= change_threshold,
                "sign_flip_count": sign_flip_count,
                "positive_checkpoints": sum(1 for v in ys if v > 0),
                "negative_checkpoints": sum(1 for v in ys if v < 0),
                "zero_checkpoints": sum(1 for v in ys if v == 0),
                "trajectory": [
                    {
                        "checkpoint_name": str(row.get("checkpoint_name") or ""),
                        "checkpoint_order": row.get("checkpoint_order"),
                        "baseline_acc": _safe_float(row.get("baseline_acc")),
                        "skill_acc": _safe_float(row.get("skill_acc")),
                        "delta": float(row["delta"]),
                    }
                    for row in ordered
                ],
            }
        )
    return trajectories


def _summarize_trajectories(
    rows: Sequence[Dict[str, Any]],
    *,
    change_threshold: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["teacher_backend"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (method, teacher_backend), items in sorted(grouped.items()):
        delta_changes = [float(row["delta_change"]) for row in items]
        ranges = [float(row["delta_range"]) for row in items]
        out.append(
            {
                "method": method,
                "teacher_backend": teacher_backend,
                "n_skills": len(items),
                "change_threshold": change_threshold,
                "mean_delta_change": _mean(delta_changes),
                "mean_delta_range": _mean(ranges),
                "n_improving": sum(1 for row in items if row["trend_label"] == "improving"),
                "n_degrading": sum(1 for row in items if row["trend_label"] == "degrading"),
                "n_stable": sum(1 for row in items if row["trend_label"] == "stable"),
                "n_large_change": sum(1 for row in items if bool(row["large_change"])),
                "n_high_variability": sum(1 for row in items if bool(row["high_variability"])),
                "n_sign_flip": sum(1 for row in items if int(row["sign_flip_count"]) > 0),
                "share_large_change": sum(1 for row in items if bool(row["large_change"])) / len(items),
                "share_high_variability": sum(1 for row in items if bool(row["high_variability"])) / len(items),
                "share_sign_flip": sum(1 for row in items if int(row["sign_flip_count"]) > 0) / len(items),
            }
        )
    return out


def _top_rows(rows: Sequence[Dict[str, Any]], *, field: str, descending: bool, limit: int) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[field]), reverse=descending)
    return [
        {
            "skill_id": row["skill_id"],
            "method": row["method"],
            "teacher_backend": row["teacher_backend"],
            "source_idx": row.get("source_idx"),
            "problem": row.get("problem"),
            "start_checkpoint_name": row["start_checkpoint_name"],
            "end_checkpoint_name": row["end_checkpoint_name"],
            "start_delta": row["start_delta"],
            "end_delta": row["end_delta"],
            "delta_change": row["delta_change"],
            "delta_range": row["delta_range"],
            "trend_label": row["trend_label"],
        }
        for row in ordered[:limit]
    ]


def _write_json(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> int:
    details = read_jsonl(args.details)
    output_dir = Path(args.output_dir)
    skill_lookup = _load_skill_lookup(getattr(args, "skills_run_dir", None))
    trajectories = _build_skill_trajectories(
        details,
        skill_lookup=skill_lookup,
        min_checkpoints=int(args.min_checkpoints),
        change_threshold=float(args.change_threshold),
    )
    summary = _summarize_trajectories(trajectories, change_threshold=float(args.change_threshold))
    output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "write_trajectories", False):
        write_jsonl(output_dir / "skill_trajectories.jsonl", trajectories)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "top_improving_skills.json", _top_rows(trajectories, field="delta_change", descending=True, limit=int(args.top_k)))
    _write_json(output_dir / "top_degrading_skills.json", _top_rows(trajectories, field="delta_change", descending=False, limit=int(args.top_k)))
    _write_json(output_dir / "top_most_variable_skills.json", _top_rows(trajectories, field="delta_range", descending=True, limit=int(args.top_k)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze whether the same skill becomes more or less useful across checkpoints")
    parser.add_argument("--details", required=True, help="cross_checkpoint_details.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skills-run-dir", default="", help="optional run1 output dir for attaching skill_text")
    parser.add_argument("--min-checkpoints", type=int, default=2)
    parser.add_argument("--change-threshold", type=float, default=0.1, help="threshold for marking a skill as changed")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--write-trajectories", action="store_true", help="write skill_trajectories.jsonl")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_checkpoints < 2:
        raise SystemExit("--min-checkpoints must be >= 2")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.change_threshold < 0:
        raise SystemExit("--change-threshold must be >= 0")
    return int(run_analysis(args))


if __name__ == "__main__":
    raise SystemExit(main())
