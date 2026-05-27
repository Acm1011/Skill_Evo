from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.preliminary.eval_source_linked_skills import group_questions, run_eval


class EvalSourceLinkedSkillsTests(unittest.TestCase):
    def test_group_questions_aggregates_baseline_acc(self) -> None:
        rows = [
            {"idx": 1, "problem": "p1", "topic": "Math->A", "student_response": "a", "is_correct": True, "ground_truth": "1"},
            {"idx": 1, "problem": "p1", "topic": "Math->A", "student_response": "b", "is_correct": False, "ground_truth": "1"},
            {"idx": 2, "problem": "p2", "topic": "Math->B", "student_response": "c", "is_correct": True, "ground_truth": "2"},
            {"idx": 2, "problem": "p2", "topic": "Math->B", "student_response": "d", "is_correct": True, "ground_truth": "2"},
        ]
        grouped = group_questions(rows, 10)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["baseline_acc"], 0.5)
        self.assertEqual(grouped[1]["baseline_acc"], 1.0)

    def test_run_eval_writes_dual_teacher_outputs_and_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            traj = tmp / "traj.jsonl"
            out_dir = tmp / "out"
            rows = [
                {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "good", "is_correct": True, "ground_truth": "3"},
                {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "bad", "is_correct": False, "ground_truth": "3"},
                {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "bad2", "is_correct": False, "ground_truth": "3"},
                {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "bad3", "is_correct": False, "ground_truth": "3"},
                {"idx": 2, "problem": "Solve x^2=-1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "bad", "is_correct": False, "ground_truth": "no real solution"},
                {"idx": 2, "problem": "Solve x^2=-1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "bad", "is_correct": False, "ground_truth": "no real solution"},
                {"idx": 2, "problem": "Solve x^2=-1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "bad", "is_correct": False, "ground_truth": "no real solution"},
                {"idx": 2, "problem": "Solve x^2=-1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "bad", "is_correct": False, "ground_truth": "no real solution"},
            ]
            traj.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

            def fake_api(method: str, messages, args):
                if method == "skillrl":
                    return json.dumps(
                        {
                            "general_skills": [{"title": "Check", "principle": "Verify", "when_to_apply": "Always"}],
                            "task_specific_skills": [],
                            "common_mistakes": [],
                        },
                        ensure_ascii=False,
                    )
                if method == "reasoningbank":
                    return "# Memory Item 1\n## Title Check\n## Description Verify\n## Content Verify the answer.\n"
                return "# Raw Rule\nVerify the answer.\n\n# Memory Item 1\n## Title Check\n## Description Verify\n## Content Verify the answer.\n## Type success_rule\n"

            def fake_server(method: str, messages, args, teacher_urls):
                return fake_api(method, messages, args)

            def fake_rollout(**kwargs):
                n = int(kwargs["rollout_n"])
                return ["reasoning here \\boxed{3}" for _ in range(n)]

            args = mock.Mock(
                trajectories=str(traj),
                output_dir=str(out_dir),
                method="all",
                sample_size=10,
                server_urls=["http://127.0.0.1:8760"],
                teacher_server_urls=["http://127.0.0.1:8760"],
                served_model_name="",
                teacher_api_base_url="http://api/v1",
                teacher_api_model="model",
                teacher_api_key="",
                student_rollout_n=4,
                teacher_temperature=0.2,
                teacher_top_p=0.95,
                teacher_top_k=50,
                teacher_max_tokens=4096,
                teacher_timeout=60.0,
                student_temperature=0.7,
                student_top_p=0.95,
                student_max_tokens=4096,
                student_timeout=60.0,
                student_max_retries=1,
                student_max_concurrent=0,
            )

            with mock.patch("baselines.preliminary.eval_source_linked_skills._call_api_teacher", side_effect=fake_api), \
                mock.patch("baselines.preliminary.eval_source_linked_skills._call_server_teacher", side_effect=fake_server), \
                mock.patch("baselines.preliminary.eval_source_linked_skills._rollout_prompt", side_effect=fake_rollout):
                self.assertEqual(run_eval(args), 0)

            details = read_jsonl(out_dir / "details.jsonl")
            self.assertEqual(len(details), 12)
            rbm_skipped = [r for r in details if r["method"] == "reasoningbank" and r["source_idx"] == 2]
            self.assertEqual(len(rbm_skipped), 2)
            self.assertTrue(all(r["skip_reason"] == "no_success_trajectory" for r in rbm_skipped))

            skill_rows = read_jsonl(out_dir / "generated_skills" / "skillrl" / "api_teacher.jsonl")
            self.assertEqual(len(skill_rows), 2)
            rollout_rows = read_jsonl(out_dir / "student_rollout" / "skillrl" / "api_teacher.jsonl")
            self.assertEqual(len(rollout_rows), 8)
