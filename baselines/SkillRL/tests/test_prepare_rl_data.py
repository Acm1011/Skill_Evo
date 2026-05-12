from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.SkillRL.prepare_rl_data import (
    build_general_candidates,
    build_mistake_candidates,
    build_task_candidates,
    format_skill_prompt,
    retrieve_bucket,
    run_prepare_rl_data,
)


class PrepareRLDataTests(unittest.TestCase):
    def test_candidate_builders(self) -> None:
        general = build_general_candidates(
            [{"skill_id": "gen_001", "title": "A", "principle": "B", "when_to_apply": "When X"}]
        )
        task = build_task_candidates(
            [{"skill_id": "dyn_001", "title": "C", "principle": "D", "when_to_apply": ""}]
        )
        mistakes = build_mistake_candidates(
            [{"description": "Bad", "how_to_avoid": "Do good"}]
        )
        self.assertEqual(general[0]["problem_type"], "When X")
        self.assertEqual(task[0]["problem_type"], "C D")
        self.assertEqual(mistakes[0]["id"], "cm_000000")
        self.assertEqual(mistakes[0]["problem_type"], "Bad Do good")

    def test_retrieve_bucket_restores_items(self) -> None:
        candidates = [
            {"id": "a", "problem_type": "x", "utility": 0.0, "_item": {"skill_id": "a"}},
            {"id": "b", "problem_type": "y", "utility": 0.0, "_item": {"skill_id": "b"}},
        ]
        with mock.patch("baselines.SkillRL.prepare_rl_data._post_rank", return_value=[1, 0]):
            out = retrieve_bucket(
                question="q",
                candidates=candidates,
                top_k=2,
                retriever_url="http://127.0.0.1:8766",
                mode="embedding",
                retrieve_lambda=0.5,
            )
        self.assertEqual([x["skill_id"] for x in out], ["b", "a"])

    def test_format_skill_prompt(self) -> None:
        txt = format_skill_prompt(
            topic_key="Mathematics_Calculus",
            general_skills=[{"title": "G", "principle": "gp"}],
            task_skills=[{"title": "T", "principle": "tp", "when_to_apply": "soon"}],
            mistakes=[{"description": "wrong", "how_to_avoid": "right"}],
        )
        self.assertIn("### General Principles", txt)
        self.assertIn("### Mathematics Calculus Skills", txt)
        self.assertIn("_Apply when: soon_", txt)
        self.assertIn("### Mistakes to Avoid", txt)

    def test_run_prepare_rl_data_outputs_jsonl_and_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            deepmath = tmp / "deepmath.jsonl"
            skills = tmp / "skills.json"
            out_jsonl = tmp / "out.jsonl"
            out_parquet = tmp / "out.parquet"

            row = {
                "prompt": [{"role": "user", "content": "legacy"}],
                "reward_model": {"ground_truth": "42"},
                "extra_info": {"problem": "What is 6*7?", "topic": "Math->Arithmetic"},
                "idx": 7,
            }
            deepmath.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            skills.write_text(
                json.dumps(
                    {
                        "general_skills": [{"skill_id": "gen_001", "title": "G", "principle": "gp"}],
                        "task_specific_skills": {
                            "Math_Arithmetic": [
                                {"skill_id": "dyn_001", "title": "T", "principle": "tp", "when_to_apply": "task"}
                            ]
                        },
                        "common_mistakes": [{"description": "wrong", "how_to_avoid": "right"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            args = argparse.Namespace(
                deepmath_jsonl=str(deepmath),
                skills_json=str(skills),
                retriever_url="http://127.0.0.1:8766",
                start=0,
                end=None,
                top_k_general=1,
                top_k_task=1,
                top_k_mistake=1,
                mode="embedding",
                retrieve_lambda=0.5,
                output_jsonl=str(out_jsonl),
                output_parquet=str(out_parquet),
                keep_raw_prompt=True,
                fail_on_retrieve_error=True,
            )

            def fake_post_rank(**kwargs):
                return list(range(min(kwargs["top_k"], len(kwargs["candidates"]))))

            with mock.patch("baselines.SkillRL.prepare_rl_data._post_rank", side_effect=fake_post_rank):
                rc = run_prepare_rl_data(args)
            self.assertEqual(rc, 0)
            lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["extra_info"]["retrieved_general_count"], 1)
            self.assertEqual(rec["extra_info"]["retrieved_task_count"], 1)
            self.assertEqual(rec["extra_info"]["retrieved_mistake_count"], 1)
            self.assertIn("What is 6*7?", rec["prompt"][0]["content"])
            self.assertTrue(out_parquet.is_file())


if __name__ == "__main__":
    unittest.main()
