from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from baselines.ReasoningBankMath.io_utils import read_jsonl, write_jsonl


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _read_json_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = [data]
    else:
        raise RuntimeError(f"unsupported JSON structure in {path}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"expected object at {path}[{idx}]")
        out.append(row)
    return out


def load_details(*, run_dir: Optional[str], details_path: Optional[str]) -> List[Dict[str, Any]]:
    if details_path:
        return _read_json_rows(Path(details_path))

    if not run_dir:
        raise ValueError("either run_dir or details_path must be provided")

    root = Path(run_dir)
    cross_path = root / "cross_checkpoint_details.jsonl"
    if cross_path.is_file():
        return read_jsonl(cross_path)

    per_checkpoint_root = root / "per_checkpoint"
    if not per_checkpoint_root.is_dir():
        raise FileNotFoundError(
            f"could not find {cross_path} or per-checkpoint outputs under {per_checkpoint_root}"
        )

    rows: List[Dict[str, Any]] = []
    for child in sorted(per_checkpoint_root.iterdir()):
        if not child.is_dir():
            continue
        for candidate in (child / "details.jsonl", child / "details.json"):
            if candidate.is_file():
                rows.extend(_read_json_rows(candidate))
                break
    return rows


def summarize_checkpoint_mean_utility(
    details: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    overall_buckets: DefaultDict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    grouped_buckets: DefaultDict[Tuple[str, int, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in details:
        checkpoint_name = str(row.get("checkpoint_name") or "")
        checkpoint_order = int(row.get("checkpoint_order") or 0)
        method = str(row.get("method") or "")
        teacher_backend = str(row.get("teacher_backend") or "")
        overall_buckets[(checkpoint_name, checkpoint_order)].append(row)
        grouped_buckets[(checkpoint_name, checkpoint_order, method, teacher_backend)].append(row)

    def build_rows(
        buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]],
        *,
        with_group: bool,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for key, rows in sorted(buckets.items(), key=lambda item: (item[0][1], item[0][0], item[0][2:])):
            evaluated = [r for r in rows if _safe_float(r.get("delta")) is not None]
            deltas = [float(r["delta"]) for r in evaluated]
            baseline_accs = [_safe_float(r.get("baseline_acc")) for r in evaluated]
            skill_accs = [_safe_float(r.get("skill_acc")) for r in evaluated]
            sample = rows[0]
            item: Dict[str, Any] = {
                "checkpoint_name": str(sample.get("checkpoint_name") or ""),
                "checkpoint_path": str(sample.get("checkpoint_path") or ""),
                "checkpoint_order": int(sample.get("checkpoint_order") or 0),
                "n_total_questions": len(rows),
                "n_evaluated_questions": len(evaluated),
                "mean_baseline_acc": _mean([v for v in baseline_accs if v is not None]),
                "mean_skill_acc": _mean([v for v in skill_accs if v is not None]),
                "mean_skill_utility": _mean(deltas),
                "improved_questions": sum(1 for v in deltas if v > 0),
                "degraded_questions": sum(1 for v in deltas if v < 0),
                "unchanged_questions": sum(1 for v in deltas if v == 0),
            }
            if with_group:
                item["method"] = str(sample.get("method") or "")
                item["teacher_backend"] = str(sample.get("teacher_backend") or "")
            out.append(item)
        return out

    overall = build_rows(overall_buckets, with_group=False)
    by_group = build_rows(grouped_buckets, with_group=True)
    return overall, by_group


def _write_json(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> int:
    details = load_details(
        run_dir=getattr(args, "run_dir", None),
        details_path=getattr(args, "details", None),
    )
    overall, by_group = summarize_checkpoint_mean_utility(details)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "checkpoint_mean_skill_utility_overall.json", overall)
    _write_json(output_dir / "checkpoint_mean_skill_utility_by_group.json", by_group)
    if getattr(args, "write_jsonl", False):
        write_jsonl(output_dir / "checkpoint_mean_skill_utility_overall.jsonl", overall)
        write_jsonl(output_dir / "checkpoint_mean_skill_utility_by_group.jsonl", by_group)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute mean skill utility (delta = skill_acc - baseline_acc) for each checkpoint."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        help="Drift eval output directory. Will read cross_checkpoint_details.jsonl if present, otherwise scan per_checkpoint/*/details.jsonl.",
    )
    source.add_argument(
        "--details",
        help="Path to cross_checkpoint_details.jsonl or a compatible details.json/jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write aggregated checkpoint utility summaries.",
    )
    parser.add_argument(
        "--write-jsonl",
        action="store_true",
        help="Also write JSONL versions of the outputs.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
