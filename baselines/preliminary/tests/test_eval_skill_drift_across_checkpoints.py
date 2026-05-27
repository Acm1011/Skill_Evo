from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.preliminary.eval_skill_drift_across_checkpoints import (
    discover_checkpoints,
    load_skill_sets,
    run_eval,
)


class EvalSkillDriftAcrossCheckpointsTests(unittest.TestCase):
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

