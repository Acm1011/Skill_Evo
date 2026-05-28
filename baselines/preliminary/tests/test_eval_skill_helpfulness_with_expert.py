from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.preliminary.eval_skill_helpfulness_with_expert import run_eval


class EvalSkillHelpfulnessWithExpertTests(unittest.TestCase):
    def test_run_eval_scores_skills_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            skills_dir = tmp / "skills_run" / "generated_skills" / "skillrl"
            skills_dir.mkdir(parents=True)
            rows = [
                {
                    "source_idx": 1,
                    "problem": "Solve x+2=5",
                    "method": "skillrl",
                    "teacher_backend": "api_teacher",
                    "status": "ok",
                    "skill_text": "Isolate the variable by subtracting 2 from both sides.",
                },
                {
                    "source_idx": 2,
                    "problem": "Solve x^2=-1",
                    "method": "skillrl",
                    "teacher_backend": "api_teacher",
                    "status": "parse_error",
                    "skill_text": "",
                },
            ]
            (skills_dir / "api_teacher.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            traj = tmp / "traj.jsonl"
            traj_rows = [
                {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "a", "is_correct": True, "ground_truth": "3"},
                {"idx": 2, "problem": "Solve x^2=-1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "b", "is_correct": False, "ground_truth": "no real solution"},
            ]
            traj.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in traj_rows) + "\n", encoding="utf-8")

            args = mock.Mock(
                skills_run_dir=str(tmp / "skills_run"),
                output_dir=str(tmp / "out"),
                trajectories=str(traj),
                sample_size=10,
                methods=["skillrl"],
                teacher_backends=["api_teacher"],
                expert_api_base_url="http://api/v1",
                expert_api_model="judge-model",
                expert_api_key="",
                expert_temperature=0.0,
                expert_max_tokens=512,
                expert_timeout=60.0,
                eval_max_workers=2,
            )

            def fake_expert(messages, args):
                self.assertIn("Scoring rubric", messages[1]["content"])
                return json.dumps({"score": 5, "label": "Highly helpful", "rationale": "The skill directly applies."})

            with mock.patch("baselines.preliminary.eval_skill_helpfulness_with_expert._call_expert_api", side_effect=fake_expert):
                self.assertEqual(run_eval(args), 0)

            scored = read_jsonl(tmp / "out" / "expert_scores.jsonl")
            self.assertEqual(len(scored), 2)
            self.assertEqual(scored[0]["expert_score"], 5)
            self.assertEqual(scored[0]["judge_status"], "ok")
            self.assertIsNone(scored[1]["expert_score"])
            self.assertEqual(scored[1]["judge_status"], "skipped_invalid_skill")

            summary = json.loads((tmp / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["mean_score"], 5.0)
            self.assertEqual(summary[0]["skipped_invalid_skill"], 1)

    def test_run_eval_marks_parse_error_when_expert_output_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            skills_dir = tmp / "skills_run" / "generated_skills" / "skillrl"
            skills_dir.mkdir(parents=True)
            (skills_dir / "api_teacher.jsonl").write_text(
                json.dumps(
                    {
                        "source_idx": 1,
                        "problem": "Solve x+2=5",
                        "method": "skillrl",
                        "teacher_backend": "api_teacher",
                        "status": "ok",
                        "skill_text": "Subtract 2 from both sides.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            traj = tmp / "traj.jsonl"
            traj.write_text(
                json.dumps(
                    {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "a", "is_correct": True, "ground_truth": "3"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            args = mock.Mock(
                skills_run_dir=str(tmp / "skills_run"),
                output_dir=str(tmp / "out"),
                trajectories=str(traj),
                sample_size=10,
                methods=["skillrl"],
                teacher_backends=["api_teacher"],
                expert_api_base_url="http://api/v1",
                expert_api_model="judge-model",
                expert_api_key="",
                expert_temperature=0.0,
                expert_max_tokens=512,
                expert_timeout=60.0,
                eval_max_workers=1,
            )

            with mock.patch("baselines.preliminary.eval_skill_helpfulness_with_expert._call_expert_api", return_value="score: 7"):
                self.assertEqual(run_eval(args), 0)

            scored = read_jsonl(tmp / "out" / "expert_scores.jsonl")
            self.assertEqual(scored[0]["judge_status"], "parse_error")
            self.assertIsNone(scored[0]["expert_score"])
