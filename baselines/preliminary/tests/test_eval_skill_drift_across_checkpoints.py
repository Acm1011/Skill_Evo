from __future__ import annotations

import json
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.preliminary.eval_skill_drift_across_checkpoints import (
    _default_eval_workers,
    RolloutServerManager,
    discover_checkpoints,
    evaluate_checkpoint,
    load_skill_sets,
    run_eval,
)


class EvalSkillDriftAcrossCheckpointsTests(unittest.TestCase):
    def test_default_eval_workers_uses_gpu_count(self) -> None:
        self.assertEqual(_default_eval_workers(n_gpus=8, requested=0), 8)
        self.assertEqual(_default_eval_workers(n_gpus=8, requested=3), 3)

    def test_discover_checkpoints_sorts_numeric_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("checkpoint-200", "checkpoint-100", "global_step_500"):
                path = root / name
                path.mkdir()
                (path / "config.json").write_text("{}", encoding="utf-8")
                (path / "model.safetensors").write_text("", encoding="utf-8")
            found = discover_checkpoints(root)
            self.assertEqual([item["checkpoint_name"] for item in found], ["checkpoint-100", "checkpoint-200", "global_step_500"])

    def test_discover_checkpoints_uses_relative_name_for_nested_verl_exports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("global_step_50", "global_step_100"):
                path = root / name / "actor" / "huggingface"
                path.mkdir(parents=True)
                (path / "config.json").write_text("{}", encoding="utf-8")
                (path / "model-00001-of-00002.safetensors").write_text("", encoding="utf-8")
            found = discover_checkpoints(root)
            self.assertEqual(
                [item["checkpoint_name"] for item in found],
                [
                    "global_step_50__actor__huggingface",
                    "global_step_100__actor__huggingface",
                ],
            )

    def test_load_skill_sets_rebuilds_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "generated_skills" / "skillrl"
            root.mkdir(parents=True)
            path = root / "api_teacher.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "source_idx": 7,
                        "problem": "Solve x+2=5",
                        "method": "skillrl",
                        "teacher_backend": "api_teacher",
                        "status": "ok",
                        "skill_text": "use inverse operations",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            skill_sets = load_skill_sets(Path(td), ["skillrl"], ["api_teacher"])
            row = skill_sets[("skillrl", "api_teacher")][("source_idx", "7")]
            self.assertEqual(row["skill_text"], "use inverse operations")

    def test_run_eval_reuses_same_skills_across_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            skills_dir = tmp / "skills_run"
            for method in ("skillrl", "reasoningbank"):
                for teacher_backend in ("api_teacher", "server_teacher"):
                    path = skills_dir / "generated_skills" / method
                    path.mkdir(parents=True, exist_ok=True)
                    rows = [
                        {
                            "source_idx": 1,
                            "problem": "Solve x+2=5",
                            "method": method,
                            "teacher_backend": teacher_backend,
                            "status": "ok",
                            "skill_text": f"{method}-{teacher_backend}-skill",
                        },
                        {
                            "source_idx": 2,
                            "problem": "Solve x^2=-1",
                            "method": method,
                            "teacher_backend": teacher_backend,
                            "status": "missing_skill",
                            "skill_text": "",
                        },
                    ]
                    path.joinpath(f"{teacher_backend}.jsonl").write_text(
                        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8",
                    )

            traj = tmp / "traj.jsonl"
            traj_rows = [
                {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "a", "is_correct": True, "ground_truth": "3"},
                {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "b", "is_correct": False, "ground_truth": "3"},
                {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "c", "is_correct": False, "ground_truth": "3"},
                {"idx": 1, "problem": "Solve x+2=5", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "d", "is_correct": False, "ground_truth": "3"},
                {"idx": 2, "problem": "Solve x^2=-1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "e", "is_correct": False, "ground_truth": "no real solution"},
                {"idx": 2, "problem": "Solve x^2=-1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "f", "is_correct": False, "ground_truth": "no real solution"},
                {"idx": 2, "problem": "Solve x^2=-1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "g", "is_correct": False, "ground_truth": "no real solution"},
                {"idx": 2, "problem": "Solve x^2=-1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "h", "is_correct": False, "ground_truth": "no real solution"},
            ]
            traj.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in traj_rows) + "\n", encoding="utf-8")

            ckpt_root = tmp / "ckpts"
            for name in ("checkpoint-100", "checkpoint-200"):
                path = ckpt_root / name
                path.mkdir(parents=True)
                (path / "config.json").write_text("{}", encoding="utf-8")
                (path / "model.safetensors").write_text("", encoding="utf-8")

            class FakeManager:
                def __init__(self, args) -> None:
                    self.args = args

                def start(self, checkpoint):
                    return object()

                def stop(self, proc) -> None:
                    return None

            class FakeClient:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                def generate_sync(self, prompts, sampling, request_timeout=None):
                    prompt = prompts[0]
                    n = int(sampling["n"])
                    if "SKILL:" in prompt and "checkpoint-200" not in prompt:
                        text = "reasoning \\boxed{3}"
                    else:
                        text = "reasoning \\boxed{0}"
                    return [type("Resp", (), {"outputs": [type("Out", (), {"text": text})() for _ in range(n)]})()]

            def fake_evaluate_checkpoint(*, checkpoint, questions, skill_sets, args, solve_template):
                details = []
                attempts = []
                baseline_acc = 0.0 if checkpoint["checkpoint_name"] == "checkpoint-100" else 0.25
                for (method, teacher_backend), mapping in sorted(skill_sets.items()):
                    skill_row = mapping[("source_idx", "1")]
                    details.append(
                        {
                            "checkpoint_name": checkpoint["checkpoint_name"],
                            "checkpoint_path": checkpoint["checkpoint_path"],
                            "checkpoint_order": checkpoint["checkpoint_order"],
                            "source_idx": 1,
                            "problem": "Solve x+2=5",
                            "method": method,
                            "teacher_backend": teacher_backend,
                            "baseline_acc": baseline_acc,
                            "skill_acc": 1.0 if checkpoint["checkpoint_name"] == "checkpoint-200" else 0.5,
                            "delta": 0.75 if checkpoint["checkpoint_name"] == "checkpoint-200" else 0.5,
                            "skip_reason": "",
                        }
                    )
                    details.append(
                        {
                            "checkpoint_name": checkpoint["checkpoint_name"],
                            "checkpoint_path": checkpoint["checkpoint_path"],
                            "checkpoint_order": checkpoint["checkpoint_order"],
                            "source_idx": 2,
                            "problem": "Solve x^2=-1",
                            "method": method,
                            "teacher_backend": teacher_backend,
                            "baseline_acc": baseline_acc,
                            "skill_acc": None,
                            "delta": None,
                            "skip_reason": "missing_skill",
                        }
                    )
                    attempts.append(
                        {
                            "checkpoint_name": checkpoint["checkpoint_name"],
                            "checkpoint_path": checkpoint["checkpoint_path"],
                            "checkpoint_order": checkpoint["checkpoint_order"],
                            "source_idx": 1,
                            "problem": "Solve x+2=5",
                            "method": method,
                            "teacher_backend": teacher_backend,
                            "attempt_idx": 0,
                            "prompt": skill_row["skill_text"],
                            "ground_truth": "3",
                            "student_response": "reasoning \\boxed{3}",
                            "is_correct": True,
                        }
                    )
                return details, attempts

            out_dir = tmp / "out"
            args = mock.Mock(
                skills_run_dir=str(skills_dir),
                checkpoint_root=str(ckpt_root),
                output_dir=str(out_dir),
                trajectories=str(traj),
                sample_size=10,
                methods=None,
                teacher_backends=None,
                student_rollout_n=4,
                student_temperature=0.7,
                student_top_p=0.95,
                student_max_tokens=4096,
                student_timeout=60.0,
                student_max_retries=1,
                student_max_concurrent=0,
                served_model_name="",
                checkpoint_limit=0,
                repo_root=str(tmp),
                rollout_start_script=str(tmp / "start.sh"),
                rollout_log_root=str(tmp / "logs"),
                gpu_ids="0",
                n_gpus=1,
                rollout_host="127.0.0.1",
                rollout_base_port=8760,
                rollout_health_timeout=10,
            )

            with mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints.evaluate_checkpoint", side_effect=fake_evaluate_checkpoint):
                self.assertEqual(run_eval(args), 0)

            cross_summary = json.loads((out_dir / "cross_checkpoint_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(cross_summary), 8)
            cross_details = read_jsonl(out_dir / "cross_checkpoint_details.jsonl")
            self.assertEqual(len(cross_details), 16)
            checkpoint_names = {row["checkpoint_name"] for row in cross_details}
            self.assertEqual(checkpoint_names, {"checkpoint-100", "checkpoint-200"})

    def test_run_eval_writes_distinct_nested_checkpoint_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            skills_dir = tmp / "skills_run"
            skill_path = skills_dir / "generated_skills" / "skillrl"
            skill_path.mkdir(parents=True, exist_ok=True)
            skill_path.joinpath("server_teacher.jsonl").write_text(
                json.dumps(
                    {
                        "source_idx": 1,
                        "problem": "Solve x+2=5",
                        "method": "skillrl",
                        "teacher_backend": "server_teacher",
                        "status": "ok",
                        "skill_text": "skill",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            traj = tmp / "traj.jsonl"
            traj.write_text(
                json.dumps(
                    {
                        "idx": 1,
                        "problem": "Solve x+2=5",
                        "topic": "Math->Algebra",
                        "topic_key": "Math_Algebra",
                        "student_response": "a",
                        "is_correct": True,
                        "ground_truth": "3",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            ckpt_root = tmp / "ckpts"
            for name in ("global_step_50", "global_step_100"):
                path = ckpt_root / name / "actor" / "huggingface"
                path.mkdir(parents=True)
                (path / "config.json").write_text("{}", encoding="utf-8")
                (path / "model-00001-of-00002.safetensors").write_text("", encoding="utf-8")

            def fake_evaluate_checkpoint(*, checkpoint, questions, skill_sets, args, solve_template):
                return (
                    [
                        {
                            "checkpoint_name": checkpoint["checkpoint_name"],
                            "checkpoint_path": checkpoint["checkpoint_path"],
                            "checkpoint_order": checkpoint["checkpoint_order"],
                            "source_idx": 1,
                            "problem": "Solve x+2=5",
                            "method": "skillrl",
                            "teacher_backend": "server_teacher",
                            "baseline_acc": 0.0,
                            "skill_acc": 1.0,
                            "delta": 1.0,
                            "skip_reason": "",
                        }
                    ],
                    [],
                )

            out_dir = tmp / "out"
            args = mock.Mock(
                skills_run_dir=str(skills_dir),
                checkpoint_root=str(ckpt_root),
                output_dir=str(out_dir),
                trajectories=str(traj),
                sample_size=10,
                methods=["skillrl"],
                teacher_backends=["server_teacher"],
                student_rollout_n=4,
                student_temperature=0.7,
                student_top_p=0.95,
                student_max_tokens=4096,
                student_timeout=60.0,
                student_max_retries=1,
                student_max_concurrent=0,
                served_model_name="",
                checkpoint_limit=0,
                repo_root=str(tmp),
                rollout_start_script=str(tmp / "start.sh"),
                rollout_log_root=str(tmp / "logs"),
                gpu_ids="0",
                n_gpus=1,
                rollout_host="127.0.0.1",
                rollout_base_port=8760,
                rollout_health_timeout=10,
            )

            with mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints.evaluate_checkpoint", side_effect=fake_evaluate_checkpoint):
                self.assertEqual(run_eval(args), 0)

            self.assertTrue((out_dir / "per_checkpoint" / "global_step_50__actor__huggingface" / "summary.json").is_file())
            self.assertTrue((out_dir / "per_checkpoint" / "global_step_100__actor__huggingface" / "summary.json").is_file())

    def test_run_eval_resume_reuses_existing_nested_checkpoint_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            skills_dir = tmp / "skills_run"
            skill_path = skills_dir / "generated_skills" / "skillrl"
            skill_path.mkdir(parents=True, exist_ok=True)
            skill_path.joinpath("server_teacher.jsonl").write_text(
                json.dumps(
                    {
                        "source_idx": 1,
                        "problem": "Solve x+2=5",
                        "method": "skillrl",
                        "teacher_backend": "server_teacher",
                        "status": "ok",
                        "skill_text": "skill",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            traj = tmp / "traj.jsonl"
            traj.write_text(
                json.dumps(
                    {
                        "idx": 1,
                        "problem": "Solve x+2=5",
                        "topic": "Math->Algebra",
                        "topic_key": "Math_Algebra",
                        "student_response": "a",
                        "is_correct": True,
                        "ground_truth": "3",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            ckpt_root = tmp / "ckpts"
            step50 = ckpt_root / "global_step_50" / "actor" / "huggingface"
            step100 = ckpt_root / "global_step_100" / "actor" / "huggingface"
            for path in (step50, step100):
                path.mkdir(parents=True)
                (path / "config.json").write_text("{}", encoding="utf-8")
                (path / "model-00001-of-00002.safetensors").write_text("", encoding="utf-8")

            out_dir = tmp / "out"
            resumed_dir = out_dir / "per_checkpoint" / "huggingface"
            resumed_dir.mkdir(parents=True)
            resumed_summary = [
                {
                    "checkpoint_name": "huggingface",
                    "checkpoint_path": str(step100),
                    "checkpoint_order": 1,
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "evaluated_questions": 1,
                    "baseline_acc": 0.0,
                    "skill_acc": 1.0,
                    "abs_delta": 1.0,
                    "rel_delta": None,
                    "improved": 1,
                    "degraded": 0,
                    "unchanged": 0,
                }
            ]
            (resumed_dir / "summary.json").write_text(json.dumps(resumed_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            resumed_details = [
                {
                    "checkpoint_name": "huggingface",
                    "checkpoint_path": str(step100),
                    "checkpoint_order": 1,
                    "source_idx": 1,
                    "problem": "Solve x+2=5",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "baseline_acc": 0.0,
                    "skill_acc": 1.0,
                    "delta": 1.0,
                    "skip_reason": "",
                }
            ]
            (resumed_dir / "details.json").write_text(json.dumps(resumed_details, ensure_ascii=False, indent=2), encoding="utf-8")
            resumed_attempts = [
                {
                    "checkpoint_name": "huggingface",
                    "checkpoint_path": str(step100),
                    "checkpoint_order": 1,
                    "source_idx": 1,
                    "problem": "Solve x+2=5",
                    "method": "skillrl",
                    "teacher_backend": "server_teacher",
                    "attempt_idx": 0,
                    "prompt": "skill",
                    "ground_truth": "3",
                    "student_response": "reasoning \\boxed{3}",
                    "is_correct": True,
                }
            ]
            (resumed_dir / "attempts.json").write_text(json.dumps(resumed_attempts, ensure_ascii=False, indent=2), encoding="utf-8")

            seen = []

            def fake_evaluate_checkpoint(*, checkpoint, questions, skill_sets, args, solve_template):
                seen.append(checkpoint["checkpoint_name"])
                return (
                    [
                        {
                            "checkpoint_name": checkpoint["checkpoint_name"],
                            "checkpoint_path": checkpoint["checkpoint_path"],
                            "checkpoint_order": checkpoint["checkpoint_order"],
                            "source_idx": 1,
                            "problem": "Solve x+2=5",
                            "method": "skillrl",
                            "teacher_backend": "server_teacher",
                            "baseline_acc": 0.0,
                            "skill_acc": 0.5,
                            "delta": 0.5,
                            "skip_reason": "",
                        }
                    ],
                    [],
                )

            args = mock.Mock(
                skills_run_dir=str(skills_dir),
                checkpoint_root=str(ckpt_root),
                output_dir=str(out_dir),
                trajectories=str(traj),
                sample_size=10,
                methods=["skillrl"],
                teacher_backends=["server_teacher"],
                student_rollout_n=4,
                student_temperature=0.7,
                student_top_p=0.95,
                student_max_tokens=4096,
                student_timeout=60.0,
                student_max_retries=1,
                student_max_concurrent=0,
                served_model_name="",
                checkpoint_limit=0,
                repo_root=str(tmp),
                rollout_start_script=str(tmp / "start.sh"),
                rollout_log_root=str(tmp / "logs"),
                gpu_ids="0",
                n_gpus=1,
                rollout_host="127.0.0.1",
                rollout_base_port=8760,
                rollout_health_timeout=10,
                eval_max_workers=0,
                resume=True,
            )

            with mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints.evaluate_checkpoint", side_effect=fake_evaluate_checkpoint):
                self.assertEqual(run_eval(args), 0)

            self.assertEqual(seen, ["global_step_50__actor__huggingface"])
            cross_details = read_jsonl(out_dir / "cross_checkpoint_details.jsonl")
            checkpoint_names = {row["checkpoint_name"] for row in cross_details}
            self.assertEqual(checkpoint_names, {"global_step_50__actor__huggingface", "global_step_100__actor__huggingface"})
            self.assertTrue((out_dir / "per_checkpoint" / "global_step_100__actor__huggingface" / "summary.json").is_file())

    def test_evaluate_checkpoint_runs_rollouts_concurrently(self) -> None:
        checkpoint = {
            "checkpoint_name": "checkpoint-100",
            "checkpoint_path": "/tmp/checkpoint-100",
            "checkpoint_order": 0,
        }
        questions = [
            {
                "meta": {
                    "source_idx": 1,
                    "problem": "p1",
                    "ground_truth": "1",
                }
            },
            {
                "meta": {
                    "source_idx": 2,
                    "problem": "p2",
                    "ground_truth": "2",
                }
            },
        ]
        skill_sets = {
            ("skillrl", "api_teacher"): {
                ("source_idx", "1"): {"status": "ok", "skill_text": "s1"},
                ("source_idx", "2"): {"status": "ok", "skill_text": "s2"},
            }
        }
        args = mock.Mock(
            n_gpus=2,
            rollout_host="127.0.0.1",
            rollout_base_port=8760,
            student_rollout_n=1,
            student_temperature=0.7,
            student_top_p=0.95,
            student_top_k=50,
            student_max_tokens=4096,
            student_timeout=60.0,
            rollout_health_timeout=10,
            eval_max_workers=2,
            repo_root="/tmp",
            rollout_start_script="/tmp/start.sh",
            rollout_log_root="/tmp/logs",
            gpu_ids="0,1",
        )

        class FakeManager:
            def __init__(self, _args) -> None:
                pass

            def start(self, _checkpoint):
                return object()

            def stop(self, _proc) -> None:
                return None

        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_rollout(*, server_urls, prompt, question, ground_truth, args):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return [{"attempt_idx": 0, "student_response": f"reasoning \\boxed{{{ground_truth}}}", "is_correct": True}], 1.0, 1

        with mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints.RolloutServerManager", FakeManager), \
            mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints._run_prompt_rollout", side_effect=fake_rollout):
            details, attempts = evaluate_checkpoint(
                checkpoint=checkpoint,
                questions=questions,
                skill_sets=skill_sets,
                args=args,
                solve_template="SKILL: {skill}\nQuestion: {question}",
            )

        self.assertEqual(len(details), 2)
        self.assertEqual(len(attempts), 2)
        self.assertGreaterEqual(max_active, 2)

    def test_rollout_server_manager_stop_kills_process_group(self) -> None:
        args = mock.Mock(
            repo_root="/tmp",
            rollout_start_script="/tmp/start.sh",
            rollout_log_root="/tmp/logs",
            gpu_ids="0,1",
            n_gpus=2,
            rollout_host="127.0.0.1",
            rollout_base_port=8760,
            rollout_health_timeout=10,
        )
        manager = RolloutServerManager(args)
        proc = mock.Mock(pid=1234)
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="bash", timeout=20), None]

        with mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints.os.getpgid", return_value=4321), \
            mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints.os.killpg") as killpg, \
            mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints.subprocess.run") as run_cmd, \
            mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints._check_health", return_value=False), \
            mock.patch("baselines.preliminary.eval_skill_drift_across_checkpoints.time.sleep"):
            manager.stop(proc)

        self.assertEqual(killpg.call_args_list[0].args, (4321, signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args, (4321, signal.SIGKILL))
        self.assertEqual(run_cmd.call_args.args[0], ["pkill", "-f", "skill_src.solver_offline_rollout_server"])
