#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对 rollouts.jsonl 计算：每题正确比例的全样本平均、至少一次正确的题数。"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from eval import is_rollout_correct


@dataclass
class RolloutEvalStats:
    path: str
    num_samples: int
    num_with_rollouts: int
    mean_per_sample_correct_rate: float
    num_at_least_one_correct: int

    def __str__(self) -> str:
        return (
            f"path: {self.path}\n"
            f"  样本数: {self.num_samples}\n"
            f"  含非空 rollout 的样本: {self.num_with_rollouts}\n"
            f"  指标1 平均每题正确率 (先算每题 correct/n 再对题平均): {self.mean_per_sample_correct_rate:.6f}\n"
            f"  指标2 至少一次正确的题目数: {self.num_at_least_one_correct}"
        )


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[evaluation] skip line {line_no} in {path}: {e}", file=sys.stderr)
    return rows


def _rollout_text(r: object) -> str:
    if isinstance(r, dict):
        return (r.get("text") or "").strip()
    return str(r).strip()


def evaluate_rollouts_file(path: Path) -> RolloutEvalStats:
    """
    指标1: 对每题算 correct/n（n 为该题 rollout 条数），再对所有题算术平均。
    指标2: correct >= 1 的题数。
    """
    rows = _read_jsonl(path)
    rates: list[float] = []
    at_least_one = 0
    with_rollouts = 0

    for rec in rows:
        gold = rec.get("answer")
        rollouts = rec.get("rollouts") or []
        n = len(rollouts)
        if n == 0:
            rates.append(0.0)
            continue
        with_rollouts += 1
        correct = 0
        for r in rollouts:
            if is_rollout_correct(_rollout_text(r), gold):
                correct += 1
        rates.append(correct / n)
        if correct >= 1:
            at_least_one += 1

    num_samples = len(rows)
    mean_rate = sum(rates) / len(rates) if rates else 0.0

    return RolloutEvalStats(
        path=str(path.resolve()),
        num_samples=num_samples,
        num_with_rollouts=with_rollouts,
        mean_per_sample_correct_rate=mean_rate,
        num_at_least_one_correct=at_least_one,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估两个 rollouts.jsonl（默认：baseline rollout vs solver_with_skills）。"
    )
    parser.add_argument(
        "--rollout",
        type=Path,
        default=Path("runs/rollout/AIME-24/20260324_113909_059596/rollouts.jsonl"),
        help="无技能 / baseline 的 rollouts.jsonl",
    )
    parser.add_argument(
        "--solver",
        type=Path,
        default=Path(
            "/home/xzs/data/experient/skill_zero/runs/solver_with_skills_retrieval/AIME-24/retrieve_top3/20260324_215440_951134/rollouts.jsonl"
        ),
        help="带技能的 rollouts.jsonl",
    )
    args = parser.parse_args()

    for label, p in (
        ("rollout (baseline)", args.rollout),
        ("solver_with_skills", args.solver),
    ):
        if not p.is_file():
            print(f"[evaluation] 文件不存在，跳过 [{label}]: {p}", file=sys.stderr)
            continue
        st = evaluate_rollouts_file(p)
        print(f"=== {label} ===")
        print(st)
        print()


if __name__ == "__main__":
    main()
