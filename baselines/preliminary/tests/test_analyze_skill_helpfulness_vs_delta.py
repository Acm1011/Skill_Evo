from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.preliminary.analyze_skill_helpfulness_vs_delta import run_analysis


class AnalyzeSkillHelpfulnessVsDeltaTests(unittest.TestCase):
    def test_run_analysis_joins_and_summarizes_delta_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            details_path = tmp / "details.jsonl"
            expert_path = tmp / "expert_scores.jsonl"
            out_dir = tmp / "out"

            details = [
                {
                    "source_idx": 1,
                    "problem": "p1",
                    "method": "skillrl",
                    "teacher_backend": "api_teacher",
                    "baseline_acc": 0.25,
                    "skill_acc": 0.75,
                    "delta": 0.5,
                    "skip_reason": "",
                },
                {
                    "source_idx": 2,
                    "problem": "p2",
                    "method": "skillrl",
                    "teacher_backend": "api_teacher",
                    "baseline_acc": 0.75,
                    "skill_acc": 0.25,
                    "delta": -0.5,
                    "skip_reason": "",
                },
                {
                    "source_idx": 3,
                    "problem": "p3",
                    "method": "skillrl",
                    "teacher_backend": "api_teacher",
                    "baseline_acc": 0.5,
                    "skill_acc": 0.5,
                    "delta": 0.0,
                    "skip_reason": "",
                },
            ]
            details_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8")

            expert_scores = [
                {
                    "source_idx": 1,
                    "problem": "p1",
                    "method": "skillrl",
                    "teacher_backend": "api_teacher",
                    "expert_score": 5,
                    "expert_label": "Highly helpful",
                    "expert_rationale": "direct",
                    "judge_status": "ok",
                },
                {
                    "source_idx": 2,
                    "problem": "p2",
                    "method": "skillrl",
                    "teacher_backend": "api_teacher",
                    "expert_score": 2,
                    "expert_label": "Weakly relevant",
                    "expert_rationale": "weak",
                    "judge_status": "ok",
                },
            ]
            expert_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in expert_scores), encoding="utf-8")

            args = type("Args", (), {"details": str(details_path), "expert_scores": str(expert_path), "output_dir": str(out_dir)})()
            self.assertEqual(run_analysis(args), 0)

            joined = read_jsonl(out_dir / "joined_rows.jsonl")
            self.assertEqual(len(joined), 3)
            self.assertEqual(joined[0]["delta_bucket"], "improved")
            self.assertEqual(joined[1]["delta_bucket"], "degraded")
            self.assertEqual(joined[2]["judge_status"], "missing_expert_score")

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            improved = next(row for row in summary if row["delta_bucket"] == "improved")
            degraded = next(row for row in summary if row["delta_bucket"] == "degraded")
            unchanged = next(row for row in summary if row["delta_bucket"] == "unchanged")
            self.assertEqual(improved["mean_expert_score"], 5.0)
            self.assertEqual(degraded["mean_expert_score"], 2.0)
            self.assertIsNone(unchanged["mean_expert_score"])
