from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skill_src.prepare_test_data import SkillEvoAdapter, SkillRLAdapter, _process_one_file


class _DummyController:
    @staticmethod
    def _sanitize_extra_info_for_parquet(extra_info):
        return None

    @staticmethod
    def _parquet_flatten_prompt_and_extra_info(rec):
        return None

    @staticmethod
    def _coerce_remaining_nested_to_json_strings(rec):
        return None


class _FakeSkill:
    def __init__(self, sid: str, name: str) -> None:
        self.id = sid
        self.skill_name = name
        self.problem_type = "ptype"
        self.key_insight = "insight"
        self.method = "method"


class _FakeManager:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, question: str, top_k: int):
        self.calls += 1
        return [_FakeSkill(f"mem_{top_k}", f"skill_for_{question}")]


class PrepareTestDataTests(unittest.TestCase):
    def test_skill_evo_adapter_writes_skill_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_path = tmp / "input.jsonl"
            output_dir = tmp / "out"
            row = {
                "prompt": [{"role": "user", "content": "legacy"}],
                "extra_info": {"problem": "What is 1+1?"},
            }
            input_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for _ in range(2)) + "\n",
                encoding="utf-8",
            )

            manager = _FakeManager()
            adapter = SkillEvoAdapter(manager, top_k=3)
            kept, seen = _process_one_file(
                input_path,
                output_dir,
                adapter=adapter,
                controller_cls=_DummyController,
                template_text="SKILL: {skill}\nQuestion: {question}",
                write_jsonl=True,
                write_parquet=False,
            )

            self.assertEqual((kept, seen), (2, 2))
            self.assertEqual(manager.calls, 2)
            lines = (output_dir / "input_skill.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            rec = json.loads(lines[0])
            self.assertEqual(rec["extra_info"]["skill_id"], ["mem_3"])
            self.assertIn("What is 1+1?", rec["prompt"][0]["content"])

    def test_skillrl_adapter_writes_retrieval_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_path = tmp / "input.jsonl"
            output_dir = tmp / "out"
            row = {
                "prompt": [{"role": "user", "content": "legacy"}],
                "extra_info": {"problem": "Compute x", "topic": "Math->Algebra"},
            }
            input_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for _ in range(2)) + "\n",
                encoding="utf-8",
            )

            calls = {"n": 0}

            def fake_prepare_candidates(bank, topic_key):
                self.assertEqual(topic_key, "Math_Algebra")
                return (
                    [{"id": "gen_001", "problem_type": "g", "utility": 0.0, "_item": {"skill_id": "gen_001", "title": "G", "principle": "gp"}}],
                    [{"id": "tsk_001", "problem_type": "t", "utility": 0.0, "_item": {"skill_id": "tsk_001", "title": "T", "principle": "tp", "when_to_apply": "when"}}],
                    [{"id": "cm_000000", "problem_type": "m", "utility": 0.0, "_item": {"_retrieval_id": "cm_000000", "description": "wrong", "how_to_avoid": "right"}}],
                )

            def fake_retrieve_bucket(**kwargs):
                calls["n"] += 1
                return [kwargs["candidates"][0]["_item"]]

            adapter = SkillRLAdapter(
                bank=object(),
                retriever_url="http://127.0.0.1:8766",
                mode="embedding",
                retrieve_lambda=0.5,
                top_k_general=1,
                top_k_task=1,
                top_k_mistake=1,
                prepare_candidates_fn=fake_prepare_candidates,
                retrieve_bucket_fn=fake_retrieve_bucket,
                topic_slug_fn=lambda topic: "Math_Algebra" if topic else "unknown",
            )

            kept, seen = _process_one_file(
                input_path,
                output_dir,
                adapter=adapter,
                controller_cls=_DummyController,
                template_text="SKILL: {skill}\nQuestion: {question}",
                write_jsonl=True,
                write_parquet=False,
            )

            self.assertEqual((kept, seen), (2, 2))
            self.assertEqual(calls["n"], 6)
            lines = (output_dir / "input_skill.jsonl").read_text(encoding="utf-8").strip().splitlines()
            rec = json.loads(lines[0])
            self.assertEqual(rec["extra_info"]["topic_key"], "Math_Algebra")
            self.assertEqual(rec["extra_info"]["retrieved_general_skill_ids"], ["gen_001"])
            self.assertEqual(rec["extra_info"]["retrieved_task_skill_ids"], ["tsk_001"])
            self.assertEqual(rec["extra_info"]["retrieved_common_mistake_ids"], ["cm_000000"])
            self.assertEqual(rec["extra_info"]["skill_id"], ["gen_001", "tsk_001", "cm_000000"])
            self.assertIn("### General Principles", rec["prompt"][0]["content"])


if __name__ == "__main__":
    unittest.main()
