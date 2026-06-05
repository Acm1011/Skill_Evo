from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from baselines.preliminary.analyze_checkpoint_mean_skill_utility import run_analysis


class AnalyzeCheckpointMeanSkillUtilityTests(unittest.TestCase):
    def test_run_analysis_from_cross_checkpoint_details(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            details_path = tmp / "cross_checkpoint_details.jsonl"
            out_dir = tmp / "out"
            rows = [
                {
                    "checkpoint_name": "global_step_50",
                    "checkpoint_path": "/tmp/global_step_50",
                    "checkpoint_order": 0,
                    "source_idx": 1,
                    "problem": "p1",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_acc": 0.25,
                    "skill_acc": 0.50,
                    "delta": 0.25,
                },
                {
                    "checkpoint_name": "global_step_50",
                    "checkpoint_path": "/tmp/global_step_50",
                    "checkpoint_order": 0,
                    "source_idx": 2,
                    "problem": "p2",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_acc": 0.50,
                    "skill_acc": 0.50,
                    "delta": 0.0,
                },
                {
                    "checkpoint_name": "global_step_100",
                    "checkpoint_path": "/tmp/global_step_100",
                    "checkpoint_order": 1,
                    "source_idx": 1,
                    "problem": "p1",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_acc": 0.25,
                    "skill_acc": 0.75,
                    "delta": 0.50,
                },
                {
                    "checkpoint_name": "global_step_100",
                    "checkpoint_path": "/tmp/global_step_100",
                    "checkpoint_order": 1,
                    "source_idx": 2,
                    "problem": "p2",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_acc": 0.50,
                    "skill_acc": None,
                    "delta": None,
                },
            ]
            details_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "run_dir": None,
                    "details": str(details_path),
                    "output_dir": str(out_dir),
                    "write_jsonl": False,
                },
            )()

            self.assertEqual(run_analysis(args), 0)

            overall = json.loads((out_dir / "checkpoint_mean_skill_utility_overall.json").read_text(encoding="utf-8"))
            self.assertEqual(len(overall), 2)
            self.assertAlmostEqual(overall[0]["mean_skill_utility"], 0.125)
            self.assertEqual(overall[0]["n_total_questions"], 2)
            self.assertEqual(overall[0]["n_evaluated_questions"], 2)
            self.assertAlmostEqual(overall[1]["mean_skill_utility"], 0.5)
            self.assertEqual(overall[1]["n_total_questions"], 2)
            self.assertEqual(overall[1]["n_evaluated_questions"], 1)

            by_group = json.loads((out_dir / "checkpoint_mean_skill_utility_by_group.json").read_text(encoding="utf-8"))
            self.assertEqual(len(by_group), 2)
            self.assertEqual(by_group[0]["method"], "skillrl")
            self.assertEqual(by_group[0]["teacher_backend"], "server_teacher")

    def test_run_analysis_falls_back_to_per_checkpoint_details(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            run_dir = tmp / "run2"
            details_dir = run_dir / "per_checkpoint" / "global_step_50__actor__huggingface"
            details_dir.mkdir(parents=True)
            details_dir.joinpath("details.jsonl").write_text(
                json.dumps(
                    {
                        "checkpoint_name": "global_step_50__actor__huggingface",
                        "checkpoint_path": "/tmp/global_step_50/actor/huggingface",
                        "checkpoint_order": 0,
                        "source_idx": 1,
                        "problem": "p1",
                        "method": "reasoningbank",
                        "teacher_backend": "api_teacher",
                        "baseline_acc": 0.2,
                        "skill_acc": 0.6,
                        "delta": 0.4,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            out_dir = tmp / "out"
            args = type(
                "Args",
                (),
                {
                    "run_dir": str(run_dir),
                    "details": None,
                    "output_dir": str(out_dir),
                    "write_jsonl": False,
                },
            )()

            self.assertEqual(run_analysis(args), 0)
            overall = json.loads((out_dir / "checkpoint_mean_skill_utility_overall.json").read_text(encoding="utf-8"))
            self.assertEqual(len(overall), 1)
            self.assertAlmostEqual(overall[0]["mean_skill_utility"], 0.4)
