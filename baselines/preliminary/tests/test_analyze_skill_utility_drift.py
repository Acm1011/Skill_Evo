from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.preliminary.analyze_skill_utility_drift import run_analysis


class AnalyzeSkillUtilityDriftTests(unittest.TestCase):
    def test_run_analysis_tracks_same_skill_across_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            details_path = tmp / "cross_checkpoint_details.jsonl"
            out_dir = tmp / "out"
            skills_dir = tmp / "run1"
            skill_file = skills_dir / "generated_skills" / "skillrl"
            skill_file.mkdir(parents=True)
            skill_file.joinpath("server_teacher.jsonl").write_text(
                json.dumps(
                    {
                        "source_idx": 1,
                        "problem": "p1",
                        "method": "skillrl",
                        "teacher_backend": "server_teacher",
                        "status": "ok",
                        "skill_text": "use factoring",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            rows = [
                {
                    "checkpoint_name": "global_step_50",
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
                    "checkpoint_name": "global_step_100",
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
                    "checkpoint_name": "global_step_200",
                    "checkpoint_order": 2,
                    "source_idx": 1,
                    "problem": "p1",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_acc": 0.50,
                    "skill_acc": 0.25,
                    "delta": -0.25,
                },
                {
                    "checkpoint_name": "global_step_50",
                    "checkpoint_order": 0,
                    "source_idx": 2,
                    "problem": "p2",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_acc": 0.25,
                    "skill_acc": 0.25,
                    "delta": 0.0,
                },
                {
                    "checkpoint_name": "global_step_100",
                    "checkpoint_order": 1,
                    "source_idx": 2,
                    "problem": "p2",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_acc": 0.25,
                    "skill_acc": 0.25,
                    "delta": 0.0,
                },
            ]
            details_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

            args = type(
                "Args",
                (),
                {
                    "details": str(details_path),
                    "output_dir": str(out_dir),
                    "skills_run_dir": str(skills_dir),
                    "min_checkpoints": 2,
                    "change_threshold": 0.1,
                    "top_k": 10,
                    "write_trajectories": False,
                },
            )()
            self.assertEqual(run_analysis(args), 0)

            self.assertFalse((out_dir / "skill_trajectories.jsonl").exists())
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary), 1)
            row = summary[0]
            self.assertEqual(row["n_skills"], 2)
            self.assertEqual(row["n_degrading"], 1)
            self.assertEqual(row["n_stable"], 1)
            self.assertEqual(row["n_high_variability"], 1)

            top_degrading = json.loads((out_dir / "top_degrading_skills.json").read_text(encoding="utf-8"))
            self.assertEqual(top_degrading[0]["source_idx"], 1)
            self.assertEqual(top_degrading[0]["delta_change"], -0.5)

            top_variable = json.loads((out_dir / "top_most_variable_skills.json").read_text(encoding="utf-8"))
            self.assertEqual(top_variable[0]["source_idx"], 1)
            self.assertEqual(top_variable[0]["trend_label"], "degrading")

    def test_run_analysis_writes_trajectories_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            details_path = tmp / "cross_checkpoint_details.jsonl"
            out_dir = tmp / "out"
            rows = [
                {
                    "checkpoint_name": "global_step_50",
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
                    "checkpoint_name": "global_step_100",
                    "checkpoint_order": 1,
                    "source_idx": 1,
                    "problem": "p1",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_acc": 0.25,
                    "skill_acc": 0.75,
                    "delta": 0.50,
                },
            ]
            details_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "details": str(details_path),
                    "output_dir": str(out_dir),
                    "skills_run_dir": "",
                    "min_checkpoints": 2,
                    "change_threshold": 0.1,
                    "top_k": 10,
                    "write_trajectories": True,
                },
            )()
            self.assertEqual(run_analysis(args), 0)
            trajectories = read_jsonl(out_dir / "skill_trajectories.jsonl")
            self.assertEqual(len(trajectories), 1)
