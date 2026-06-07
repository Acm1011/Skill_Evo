from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from baselines.preliminary.skill_utility_eval_server import (
    SkillUtilityEvalService,
    TriggerRequest,
    _checkpoint_name,
    select_questions,
)


class SkillUtilityEvalServerTests(unittest.TestCase):
    def test_select_questions_tail(self) -> None:
        rows = []
        for idx in range(5):
            for _ in range(2):
                rows.append(
                    {
                        "idx": idx,
                        "problem": f"p{idx}",
                        "topic": "Math->A",
                        "topic_key": "Math_A",
                        "student_response": "x",
                        "is_correct": idx % 2 == 0,
                        "ground_truth": str(idx),
                    }
                )
        selected = select_questions(rows, sample_size=2, selection="tail")
        self.assertEqual([item["meta"]["source_idx"] for item in selected], [3, 4])

    def test_checkpoint_name_prefers_global_step(self) -> None:
        self.assertEqual(_checkpoint_name(20, "/tmp/global_step_20/actor"), "global_step_20")
        self.assertEqual(_checkpoint_name(0, "/tmp/global_step_20/actor"), "global_step_20")

    def test_enqueue_deduplicates_existing_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            traj = root / "traj.jsonl"
            traj.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "idx": idx,
                            "problem": f"p{idx}",
                            "topic": "Math->A",
                            "topic_key": "Math_A",
                            "student_response": "x",
                            "is_correct": True,
                            "ground_truth": str(idx),
                        },
                        ensure_ascii=False,
                    )
                    for idx in range(3)
                )
                + "\n",
                encoding="utf-8",
            )
            skills_dir = root / "skills" / "generated_skills" / "skillrl"
            skills_dir.mkdir(parents=True)
            skills_dir.joinpath("api_teacher.jsonl").write_text(
                json.dumps(
                    {
                        "source_idx": 0,
                        "problem": "p0",
                        "method": "skillrl",
                        "teacher_backend": "api_teacher",
                        "status": "ok",
                        "skill_text": "skill",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            ckpt = root / "global_step_20" / "actor"
            ckpt.mkdir(parents=True)
            ckpt.joinpath("config.json").write_text("{}", encoding="utf-8")
            ckpt.joinpath("model.safetensors").write_text("", encoding="utf-8")

            args = type(
                "Args",
                (),
                {
                    "methods": ["skillrl"],
                    "teacher_backends": ["api_teacher"],
                    "trajectories": str(traj),
                    "sample_size": 2,
                    "question_selection": "tail",
                    "skills_run_dir": str(root / "skills"),
                    "output_dir": str(root / "out"),
                    "student_rollout_n": 4,
                    "student_temperature": 0.7,
                    "student_top_p": 0.95,
                    "student_top_k": 50,
                    "student_max_tokens": 4096,
                    "student_timeout": 600.0,
                    "student_max_retries": 3,
                    "repo_root": str(root),
                    "rollout_start_script": str(root / "dummy.sh"),
                    "rollout_log_root": str(root / "logs"),
                    "gpu_ids": "2,3",
                    "n_gpus": 2,
                    "rollout_host": "127.0.0.1",
                    "rollout_base_port": 8760,
                    "rollout_health_timeout": 1,
                    "eval_max_workers": 0,
                },
            )()

            with mock.patch(
                "baselines.preliminary.skill_utility_eval_server.evaluate_checkpoint",
                return_value=([], []),
            ):
                service = SkillUtilityEvalService(args)
                first = service.enqueue(TriggerRequest(checkpoint_path=str(ckpt), global_step=20))
                second = service.enqueue(TriggerRequest(checkpoint_path=str(ckpt), global_step=20))
                self.assertEqual(first.checkpoint_path, second.checkpoint_path)
                time.sleep(0.05)
                status = service.status()
                self.assertEqual(status["jobs"][0]["checkpoint_name"], "global_step_20")


if __name__ == "__main__":
    unittest.main()
