#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 memory_after_syn / memory_after_sol 快照的整体变化。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


def _add_skill_src_to_path() -> None:
    root = Path(__file__).resolve().parent.parent
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


_add_skill_src_to_path()

from skill_manager.skill_item import SkillItem  # noqa: E402


SNAPSHOT_RE = re.compile(r"^memory_after_(syn|sol)_v(\d+)\.jsonl$")


@dataclass(frozen=True)
class SnapshotMeta:
    phase: str
    version: int
    path: Path

    @property
    def label(self) -> str:
        return f"after_{self.phase}_v{self.version}"

    @property
    def stage_index(self) -> int:
        return 0 if self.phase == "syn" else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--memory-dir",
        default="skill_saved/memory",
        help="包含 memory_after_syn_v*.jsonl / memory_after_sol_v*.jsonl 的目录",
    )
    p.add_argument(
        "--output-dir",
        default="Skill_Evo/skill_src/analysis/output",
        help="分析结果输出目录",
    )
    return p.parse_args()


def discover_snapshots(memory_dir: Path) -> list[SnapshotMeta]:
    metas: list[SnapshotMeta] = []
    for path in memory_dir.glob("memory_after_*.jsonl"):
        match = SNAPSHOT_RE.match(path.name)
        if not match:
            continue
        phase, version = match.group(1), int(match.group(2))
        metas.append(SnapshotMeta(phase=phase, version=version, path=path))
    metas.sort(key=lambda meta: (meta.version, meta.stage_index))
    return metas


def load_snapshot(path: Path) -> dict[str, SkillItem]:
    by_id: dict[str, SkillItem] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            item = SkillItem.from_jsonl_line(raw)
            if not item.id:
                raise ValueError(f"{path}:{line_no} missing skill id")
            by_id[item.id] = item
    return by_id


def summarize_snapshot(meta: SnapshotMeta, items_by_id: dict[str, SkillItem]) -> dict[str, Any]:
    items = list(items_by_id.values())
    utilities = [item.utility for item in items]
    successes = [item.skill_usage_success for item in items]
    failures = [item.skill_usage_failure for item in items]
    usage_totals = [s + f for s, f in zip(successes, failures)]
    return {
        "label": meta.label,
        "phase": meta.phase,
        "version": meta.version,
        "path": str(meta.path),
        "skill_count": len(items),
        "utility": summarize_numeric(utilities),
        "skill_usage_success": summarize_numeric(successes),
        "skill_usage_failure": summarize_numeric(failures),
        "skill_usage_total": summarize_numeric(usage_totals),
        "zero_usage_skill_count": sum(1 for total in usage_totals if total == 0),
        "used_skill_count": sum(1 for total in usage_totals if total > 0),
    }


def summarize_numeric(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "avg": None, "sum": 0}
    vals = [float(v) for v in values]
    return {
        "min": min(vals),
        "max": max(vals),
        "avg": mean(vals),
        "sum": sum(vals),
    }


def summarize_transition(
    prev_meta: SnapshotMeta | None,
    prev_items: dict[str, SkillItem] | None,
    curr_meta: SnapshotMeta,
    curr_items: dict[str, SkillItem],
) -> dict[str, Any]:
    if prev_meta is None or prev_items is None:
        return {
            "from": None,
            "to": curr_meta.label,
            "phase": curr_meta.phase,
            "version": curr_meta.version,
            "prev_skill_count": 0,
            "curr_skill_count": len(curr_items),
            "entered_skill_count": len(curr_items),
            "evicted_skill_count": 0,
            "retained_skill_count": 0,
            "utility_delta": empty_delta_summary(),
            "skill_usage_success_delta": empty_delta_summary(),
            "skill_usage_failure_delta": empty_delta_summary(),
        }

    prev_ids = set(prev_items)
    curr_ids = set(curr_items)
    entered_ids = sorted(curr_ids - prev_ids, key=int_if_possible)
    evicted_ids = sorted(prev_ids - curr_ids, key=int_if_possible)
    retained_ids = sorted(curr_ids & prev_ids, key=int_if_possible)

    utility_deltas = [curr_items[sid].utility - prev_items[sid].utility for sid in retained_ids]
    success_deltas = [
        curr_items[sid].skill_usage_success - prev_items[sid].skill_usage_success
        for sid in retained_ids
    ]
    failure_deltas = [
        curr_items[sid].skill_usage_failure - prev_items[sid].skill_usage_failure
        for sid in retained_ids
    ]

    return {
        "from": prev_meta.label,
        "to": curr_meta.label,
        "phase": curr_meta.phase,
        "version": curr_meta.version,
        "prev_skill_count": len(prev_items),
        "curr_skill_count": len(curr_items),
        "entered_skill_count": len(entered_ids),
        "evicted_skill_count": len(evicted_ids),
        "retained_skill_count": len(retained_ids),
        "entered_skill_ids": entered_ids,
        "evicted_skill_ids": evicted_ids,
        "utility_delta": summarize_deltas(utility_deltas),
        "skill_usage_success_delta": summarize_deltas(success_deltas),
        "skill_usage_failure_delta": summarize_deltas(failure_deltas),
    }


def empty_delta_summary() -> dict[str, Any]:
    return {
        "changed_count": 0,
        "increase_count": 0,
        "decrease_count": 0,
        "unchanged_count": 0,
        "min": None,
        "max": None,
        "avg": None,
        "sum": 0,
    }


def summarize_deltas(deltas: list[float | int]) -> dict[str, Any]:
    if not deltas:
        return empty_delta_summary()
    vals = [float(v) for v in deltas]
    changed_count = sum(1 for v in vals if v != 0)
    increase_count = sum(1 for v in vals if v > 0)
    decrease_count = sum(1 for v in vals if v < 0)
    return {
        "changed_count": changed_count,
        "increase_count": increase_count,
        "decrease_count": decrease_count,
        "unchanged_count": len(vals) - changed_count,
        "min": min(vals),
        "max": max(vals),
        "avg": mean(vals),
        "sum": sum(vals),
    }


def int_if_possible(value: str) -> tuple[int, str] | tuple[float, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (float("inf"), value)


def build_markdown_report(
    memory_dir: Path,
    snapshot_summaries: list[dict[str, Any]],
    transition_summaries: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Memory Evolution Analysis")
    lines.append("")
    lines.append(f"- memory_dir: `{memory_dir}`")
    lines.append(f"- snapshot_count: `{len(snapshot_summaries)}`")
    lines.append("")
    lines.append("## Snapshot Summary")
    lines.append("")
    lines.append(
        "| snapshot | skills | utility_min | utility_max | utility_avg | "
        "success_avg | success_sum | failure_avg | failure_sum | zero_usage_skills |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in snapshot_summaries:
        lines.append(
            "| {label} | {skill_count} | {u_min:.6f} | {u_max:.6f} | {u_avg:.6f} | "
            "{s_avg:.3f} | {s_sum:.0f} | {f_avg:.3f} | {f_sum:.0f} | {zero_usage_skill_count} |".format(
                label=item["label"],
                skill_count=item["skill_count"],
                u_min=item["utility"]["min"] or 0.0,
                u_max=item["utility"]["max"] or 0.0,
                u_avg=item["utility"]["avg"] or 0.0,
                s_avg=item["skill_usage_success"]["avg"] or 0.0,
                s_sum=item["skill_usage_success"]["sum"],
                f_avg=item["skill_usage_failure"]["avg"] or 0.0,
                f_sum=item["skill_usage_failure"]["sum"],
                zero_usage_skill_count=item["zero_usage_skill_count"],
            )
        )
    lines.append("")
    lines.append("## Transition Summary")
    lines.append("")
    lines.append(
        "| transition | prev_skills | curr_skills | entered | evicted | retained | "
        "utility_changed | utility_delta_sum | utility_delta_avg | "
        "success_delta_sum | failure_delta_sum |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in transition_summaries:
        from_label = item["from"] or "None"
        lines.append(
            "| {from_label} -> {to_label} | {prev_skill_count} | {curr_skill_count} | "
            "{entered_skill_count} | {evicted_skill_count} | {retained_skill_count} | "
            "{u_changed} | {u_sum:.6f} | {u_avg:.6f} | {s_sum:.0f} | {f_sum:.0f} |".format(
                from_label=from_label,
                to_label=item["to"],
                prev_skill_count=item["prev_skill_count"],
                curr_skill_count=item["curr_skill_count"],
                entered_skill_count=item["entered_skill_count"],
                evicted_skill_count=item["evicted_skill_count"],
                retained_skill_count=item["retained_skill_count"],
                u_changed=item["utility_delta"]["changed_count"],
                u_sum=item["utility_delta"]["sum"],
                u_avg=item["utility_delta"]["avg"] or 0.0,
                s_sum=item["skill_usage_success_delta"]["sum"],
                f_sum=item["skill_usage_failure_delta"]["sum"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    memory_dir = Path(args.memory_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = discover_snapshots(memory_dir)
    if not snapshots:
        raise FileNotFoundError(f"no memory snapshots found under {memory_dir}")

    snapshot_summaries: list[dict[str, Any]] = []
    transition_summaries: list[dict[str, Any]] = []
    prev_meta: SnapshotMeta | None = None
    prev_items: dict[str, SkillItem] | None = None

    for meta in snapshots:
        items = load_snapshot(meta.path)
        snapshot_summaries.append(summarize_snapshot(meta, items))
        transition_summaries.append(summarize_transition(prev_meta, prev_items, meta, items))
        prev_meta = meta
        prev_items = items

    report = {
        "memory_dir": str(memory_dir),
        "snapshot_summaries": snapshot_summaries,
        "transition_summaries": transition_summaries,
    }
    json_path = output_dir / "memory_evolution_summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = output_dir / "memory_evolution_report.md"
    md_path.write_text(
        build_markdown_report(memory_dir, snapshot_summaries, transition_summaries),
        encoding="utf-8",
    )

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
