from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.preliminary.eval_source_linked_skills import format_resume_status, group_questions, resume_status, run_eval


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
                eval_max_workers=0,
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

    def test_run_eval_round_robins_server_requests_across_urls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            traj = tmp / "traj.jsonl"
            out_dir = tmp / "out"
            rows = [
                {"idx": 1, "problem": "p1", "topic": "Math->A", "topic_key": "Math_A", "student_response": "good", "is_correct": True, "ground_truth": "1"},
                {"idx": 2, "problem": "p2", "topic": "Math->A", "topic_key": "Math_A", "student_response": "good", "is_correct": True, "ground_truth": "2"},
            ]
            traj.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

            teacher_targets = []
            rollout_targets = []

            def fake_api(method: str, messages, args):
                return json.dumps(
                    {
                        "general_skills": [{"title": "Check", "principle": "Verify"}],
                        "task_specific_skills": [],
                        "common_mistakes": [],
                    },
                    ensure_ascii=False,
                )

            def fake_server(method: str, messages, args, teacher_urls):
                teacher_targets.append(tuple(teacher_urls))
                return fake_api(method, messages, args)

            def fake_rollout(**kwargs):
                rollout_targets.append(tuple(kwargs["server_urls"]))
                return ["reasoning here \\boxed{1}"]

            args = mock.Mock(
                trajectories=str(traj),
                output_dir=str(out_dir),
                method="skillrl",
                sample_size=10,
                server_urls=["http://127.0.0.1:8760", "http://127.0.0.1:8761"],
                teacher_server_urls=["http://127.0.0.1:8760", "http://127.0.0.1:8761"],
                served_model_name="",
                teacher_api_base_url="http://api/v1",
                teacher_api_model="model",
                teacher_api_key="",
                student_rollout_n=1,
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
                eval_max_workers=2,
            )

            with mock.patch("baselines.preliminary.eval_source_linked_skills._call_api_teacher", side_effect=fake_api), \
                mock.patch("baselines.preliminary.eval_source_linked_skills._call_server_teacher", side_effect=fake_server), \
                mock.patch("baselines.preliminary.eval_source_linked_skills._rollout_prompt", side_effect=fake_rollout):
                self.assertEqual(run_eval(args), 0)

            self.assertEqual(set(teacher_targets), {("http://127.0.0.1:8760",), ("http://127.0.0.1:8761",)})
            self.assertEqual(set(rollout_targets), {("http://127.0.0.1:8760",), ("http://127.0.0.1:8761",)})

    def test_run_eval_resume_reuses_existing_skill_and_continues_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            traj = tmp / "traj.jsonl"
            out_dir = tmp / "out"
            rows = [
                {"idx": 1, "problem": "p1", "topic": "Math->A", "topic_key": "Math_A", "student_response": "good", "is_correct": True, "ground_truth": "1"},
            ]
            traj.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

            skill_dir = out_dir / "generated_skills" / "skillrl"
            skill_dir.mkdir(parents=True, exist_ok=True)
            prebuilt_skill = {
                "source_idx": 1,
                "problem": "p1",
                "topic": "Math->A",
                "method": "skillrl",
                "teacher_backend": "api_teacher",
                "prompt_used": "cached",
                "raw_teacher_output": "{}",
                "parsed_skill": {"general_skills": []},
                "status": "ok",
                "skill_text": "cached skill",
                "skip_reason": "",
            }
            (skill_dir / "api_teacher.jsonl").write_text(json.dumps(prebuilt_skill, ensure_ascii=False) + "\n", encoding="utf-8")

            args = mock.Mock(
                trajectories=str(traj),
                output_dir=str(out_dir),
                method="skillrl",
                sample_size=10,
                server_urls=["http://127.0.0.1:8760"],
                teacher_server_urls=["http://127.0.0.1:8760"],
                served_model_name="",
                teacher_api_base_url="http://api/v1",
                teacher_api_model="model",
                teacher_api_key="",
                student_rollout_n=1,
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
                eval_max_workers=1,
                resume=True,
            )

            def fail_teacher(*_args, **_kwargs):
                raise AssertionError("teacher should not be called when cached skill exists")

            def fake_rollout(**kwargs):
                self.assertIn("cached skill", kwargs["prompt"])
                return ["reasoning here \\boxed{1}"]

            with mock.patch("baselines.preliminary.eval_source_linked_skills._call_api_teacher", side_effect=fail_teacher), \
                mock.patch("baselines.preliminary.eval_source_linked_skills._call_server_teacher", side_effect=fail_teacher), \
                mock.patch("baselines.preliminary.eval_source_linked_skills._rollout_prompt", side_effect=fake_rollout):
                self.assertEqual(run_eval(args), 0)

            skill_rows = read_jsonl(out_dir / "generated_skills" / "skillrl" / "api_teacher.jsonl")
            self.assertEqual(skill_rows, [prebuilt_skill])
            rollout_rows = read_jsonl(out_dir / "student_rollout" / "skillrl" / "api_teacher.jsonl")
            self.assertEqual(len(rollout_rows), 1)
            self.assertEqual(rollout_rows[0]["source_idx"], 1)

    def test_resume_status_reports_complete_when_all_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            traj = tmp / "traj.jsonl"
            out_dir = tmp / "out"
            row = {"idx": 1, "problem": "p1", "topic": "Math->A", "topic_key": "Math_A", "student_response": "good", "is_correct": True, "ground_truth": "1"}
            traj.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            for teacher_backend in ("api_teacher", "server_teacher"):
                skill_path = out_dir / "generated_skills" / "skillrl" / f"{teacher_backend}.jsonl"
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                skill_path.write_text(
                    json.dumps(
                        {
                            "source_idx": 1,
                            "problem": "p1",
                            "topic": "Math->A",
                            "method": "skillrl",
                            "teacher_backend": teacher_backend,
                            "status": "ok",
                            "skill_text": "cached skill",
                        },
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )
                rollout_path = out_dir / "student_rollout" / "skillrl" / f"{teacher_backend}.jsonl"
                rollout_path.parent.mkdir(parents=True, exist_ok=True)
                rollout_path.write_text(
                    json.dumps(
                        {
                            "source_idx": 1,
                            "problem": "p1",
                            "method": "skillrl",
                            "teacher_backend": teacher_backend,
                            "attempt_idx": 0,
                            "student_response": "reasoning here \\boxed{1}",
                            "is_correct": True,
                            "ground_truth": "1",
                        },
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )

            details = [
                {
                    "source_idx": 1,
                    "problem": "p1",
                    "method": "skillrl",
                    "teacher_backend": "api_teacher",
                    "baseline_correct_count": 1,
                    "baseline_rollout_count": 1,
                    "baseline_acc": 1.0,
                    "skill_correct_count": 1,
                    "skill_rollout_count": 1,
                    "skill_acc": 1.0,
                    "delta": 0.0,
                    "skip_reason": "",
                },
                {
                    "source_idx": 1,
                    "problem": "p1",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_correct_count": 1,
                    "baseline_rollout_count": 1,
                    "baseline_acc": 1.0,
                    "skill_correct_count": 1,
                    "skill_rollout_count": 1,
                    "skill_acc": 1.0,
                    "delta": 0.0,
                    "skip_reason": "",
                },
            ]
            (out_dir / "details.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in details),
                encoding="utf-8",
            )

            args = mock.Mock(
                trajectories=str(traj),
                output_dir=str(out_dir),
                method="skillrl",
                sample_size=10,
            )
            status = resume_status(args)
            self.assertTrue(status["complete"])
            self.assertTrue(all(item["complete"] for item in status["statuses"]))
            self.assertFalse(status["requires_server"])

    def test_resume_status_reports_incomplete_when_rollout_or_detail_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            traj = tmp / "traj.jsonl"
            out_dir = tmp / "out"
            row = {"idx": 1, "problem": "p1", "topic": "Math->A", "topic_key": "Math_A", "student_response": "good", "is_correct": True, "ground_truth": "1"}
            traj.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            skill_path = out_dir / "generated_skills" / "skillrl" / "api_teacher.jsonl"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(
                json.dumps(
                    {
                        "source_idx": 1,
                        "problem": "p1",
                        "topic": "Math->A",
                        "method": "skillrl",
                        "teacher_backend": "api_teacher",
                        "status": "ok",
                        "skill_text": "cached skill",
                    },
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )

            args = mock.Mock(
                trajectories=str(traj),
                output_dir=str(out_dir),
                method="skillrl",
                sample_size=10,
            )
            status = resume_status(args)
            self.assertFalse(status["complete"])
            self.assertTrue(any(not item["complete"] for item in status["statuses"]))
            self.assertTrue(status["requires_server"])

    def test_resume_status_detail_only_gap_does_not_require_server(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            traj = tmp / "traj.jsonl"
            out_dir = tmp / "out"
            row = {"idx": 1, "problem": "p1", "topic": "Math->A", "topic_key": "Math_A", "student_response": "good", "is_correct": True, "ground_truth": "1"}
            traj.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            for teacher_backend in ("api_teacher", "server_teacher"):
                skill_path = out_dir / "generated_skills" / "skillrl" / f"{teacher_backend}.jsonl"
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                skill_path.write_text(
                    json.dumps(
                        {
                            "source_idx": 1,
                            "problem": "p1",
                            "topic": "Math->A",
                            "method": "skillrl",
                            "teacher_backend": teacher_backend,
                            "status": "ok",
                            "skill_text": "cached skill",
                        },
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )
                rollout_path = out_dir / "student_rollout" / "skillrl" / f"{teacher_backend}.jsonl"
                rollout_path.parent.mkdir(parents=True, exist_ok=True)
                rollout_path.write_text(
                    json.dumps(
                        {
                            "source_idx": 1,
                            "problem": "p1",
                            "method": "skillrl",
                            "teacher_backend": teacher_backend,
                            "attempt_idx": 0,
                            "student_response": "reasoning here \\boxed{1}",
                            "is_correct": True,
                            "ground_truth": "1",
                        },
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )

            args = mock.Mock(
                trajectories=str(traj),
                output_dir=str(out_dir),
                method="skillrl",
                sample_size=10,
            )
            status = resume_status(args)
            self.assertFalse(status["complete"])
            self.assertFalse(status["requires_server"])
            self.assertTrue(all(not item["requires_server"] for item in status["statuses"]))

    def test_format_resume_status_summarizes_counts_only(self) -> None:
        text = format_resume_status(
            {
                "complete": False,
                "requires_server": False,
                "methods": ["skillrl"],
                "n_questions": 100,
                "statuses": [
                    {
                        "method": "skillrl",
                        "teacher_backend": "api_teacher",
                        "complete": False,
                        "requires_server": False,
                        "missing_skill": ["1", "2", "3"],
                        "missing_detail": ["4"],
                        "missing_rollout": [],
                    }
                ],
            }
        )
        self.assertIn("missing_skill=3", text)
        self.assertIn("missing_detail=1", text)
        self.assertIn("missing_rollout=0", text)
        self.assertNotIn('["1", "2", "3"]', text)

    def test_run_eval_resume_rebuilds_detail_without_new_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            traj = tmp / "traj.jsonl"
            out_dir = tmp / "out"
            rows = [
                {"idx": 1, "problem": "p1", "topic": "Math->A", "topic_key": "Math_A", "student_response": "good", "is_correct": True, "ground_truth": "1"},
            ]
            traj.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

            for teacher_backend in ("api_teacher", "server_teacher"):
                skill_path = out_dir / "generated_skills" / "skillrl" / f"{teacher_backend}.jsonl"
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                skill_path.write_text(
                    json.dumps(
                        {
                            "source_idx": 1,
                            "problem": "p1",
                            "topic": "Math->A",
                            "method": "skillrl",
                            "teacher_backend": teacher_backend,
                            "status": "ok",
                            "skill_text": "cached skill",
                        },
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )
                rollout_path = out_dir / "student_rollout" / "skillrl" / f"{teacher_backend}.jsonl"
                rollout_path.parent.mkdir(parents=True, exist_ok=True)
                rollout_path.write_text(
                    json.dumps(
                        {
                            "source_idx": 1,
                            "problem": "p1",
                            "method": "skillrl",
                            "teacher_backend": teacher_backend,
                            "attempt_idx": 0,
                            "student_response": "reasoning here \\boxed{1}",
                            "is_correct": True,
                            "ground_truth": "1",
                        },
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )

            args = mock.Mock(
                trajectories=str(traj),
                output_dir=str(out_dir),
                method="skillrl",
                sample_size=10,
                server_urls=["http://127.0.0.1:8760"],
                teacher_server_urls=["http://127.0.0.1:8760"],
                served_model_name="",
                teacher_api_base_url="http://api/v1",
                teacher_api_model="model",
                teacher_api_key="",
                student_rollout_n=1,
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
                eval_max_workers=1,
                resume=True,
            )

            def fail_rollout(**_kwargs):
                raise AssertionError("rollout should not be called when only detail is missing")

            with mock.patch("baselines.preliminary.eval_source_linked_skills._rollout_prompt", side_effect=fail_rollout):
                self.assertEqual(run_eval(args), 0)

            details = read_jsonl(out_dir / "details.jsonl")
            self.assertEqual(len(details), 2)
            self.assertTrue(all(row["skill_rollout_count"] == 1 for row in details))
