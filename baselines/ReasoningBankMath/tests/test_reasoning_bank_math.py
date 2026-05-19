from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.ReasoningBankMath.build_embeddings import run_build_embeddings
from baselines.ReasoningBankMath.build_memory import run_build_memory
from baselines.ReasoningBankMath.evolve_memory import run_evolve_memory
from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.ReasoningBankMath.memory_bank import dedupe_records, load_trajectories
from baselines.ReasoningBankMath.memory_parser import parse_memory_items
from baselines.ReasoningBankMath.retrieve_memory import retrieve_records


def _teacher_output(title: str, desc: str, content: str) -> str:
    return (
        "# Memory Item 1\n"
        f"## Title {title}\n"
        f"## Description {desc}\n"
        f"## Content {content}\n"
    )


class ReasoningBankMathTests(unittest.TestCase):
    def test_parse_memory_items_standard_and_fallback(self) -> None:
        items = parse_memory_items(
            _teacher_output("Check constraints", "Short desc", "Verify domain before solving.")
        )
        self.assertEqual(items[0]["title"], "Check constraints")
        recovered = parse_memory_items("Use substitution, then verify the final answer.")
        self.assertEqual(recovered[0]["title"], "Recovered Memory")

    def test_load_trajectories_consumes_skillrl_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "traj.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "idx": 7,
                        "problem": "What is 2+2?",
                        "topic": "Math->Arithmetic",
                        "topic_key": "Math_Arithmetic",
                        "difficulty": "easy",
                        "student_response": "We compute 4.",
                        "is_correct": True,
                        "ground_truth": "4",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_trajectories(str(path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["topic_key"], "Math_Arithmetic")
        self.assertEqual(rows[0]["ground_truth"], "4")

    def test_dedupe_records_merges_duplicate_provenance(self) -> None:
        existing = [
            {
                "memory_id": "mem_old",
                "embedding_text": "Question: q\nsame",
                "memory_items": [
                    {
                        "title": "Keep equations consistent",
                        "description": "d",
                        "content": "Track the same variable through each step.",
                    }
                ],
                "provenance": [{"source_idx": 1}],
                "duplicate_count": 0,
            }
        ]
        incoming = [
            {
                "memory_id": "mem_new",
                "embedding_text": "Question: q\nsame",
                "memory_items": [
                    {
                        "title": "Keep equations consistent",
                        "description": "d2",
                        "content": "Track the same variable through each step.",
                    }
                ],
                "provenance": [{"source_idx": 2}],
                "duplicate_count": 0,
            }
        ]
        merged, dup_map = dedupe_records(existing, incoming, similarity_threshold=0.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(dup_map["mem_new"], "mem_old")
        self.assertEqual(merged[0]["duplicate_count"], 1)
        self.assertEqual(len(merged[0]["provenance"]), 2)

    def test_end_to_end_build_retrieve_evolve(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            traj1 = tmp / "traj1.jsonl"
            memory_bank = tmp / "memory_bank.jsonl"
            embeddings = tmp / "memory_embeddings.jsonl"
            traj2 = tmp / "traj2.jsonl"
            memory_bank_v2 = tmp / "memory_bank_v2.jsonl"
            embeddings_v2 = tmp / "memory_embeddings_v2.jsonl"

            traj1.write_text(
                json.dumps(
                    {
                        "idx": 1,
                        "problem": "Solve x + 2 = 5",
                        "topic": "Math->Algebra",
                        "topic_key": "Math_Algebra",
                        "difficulty": "easy",
                        "student_response": "Subtract 2 from both sides, so x=3. Check by substitution.",
                        "is_correct": True,
                        "ground_truth": "3",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_teacher(messages, **kwargs):
                user = messages[-1]["content"]
                if "x + 2 = 5" in user or "x + 7 = 10" in user:
                    return _teacher_output(
                        "Isolate the variable",
                        "Reverse operations carefully.",
                        "Undo additions or multiplications step by step and verify the result in the original equation.",
                    )
                if "x^2 = 9" in user:
                    return _teacher_output(
                        "Check both branches",
                        "Square roots can produce multiple candidates.",
                        "When reversing a square, consider both positive and negative roots and validate each one.",
                    )
                return _teacher_output(
                    "Review constraints",
                    "Keep domain restrictions visible.",
                    "Before finishing, confirm the candidate solution respects every stated constraint.",
                )

            build_args = argparse.Namespace(
                trajectories=str(traj1),
                output=str(memory_bank),
                teacher_base_url="",
                teacher_api_key="",
                teacher_model="",
                timeout=60.0,
                temperature=0.2,
                max_tokens=512,
                fail_on_error=True,
            )
            with mock.patch("baselines.ReasoningBankMath.build_memory.chat_complete", side_effect=fake_teacher):
                rc = run_build_memory(build_args)
            self.assertEqual(rc, 0)
            bank_rows = read_jsonl(memory_bank)
            self.assertEqual(len(bank_rows), 1)
            self.assertIn("embedding_text", bank_rows[0])
            self.assertEqual(bank_rows[0]["status"], "success")

            emb_args = argparse.Namespace(
                memory_bank=str(memory_bank),
                output=str(embeddings),
                existing_embeddings="",
                backend="hash",
                embed_base_url="",
                embed_api_key="",
                embed_model="",
                timeout=60.0,
                hash_dim=128,
            )
            rc = run_build_embeddings(emb_args)
            self.assertEqual(rc, 0)
            emb_rows = read_jsonl(embeddings)
            self.assertEqual(len(emb_rows), 1)

            scored = retrieve_records(
                question="Solve x + 10 = 13",
                memory_rows=bank_rows,
                embedding_rows=emb_rows,
                top_k=1,
                backend="hash",
                base_url="",
                api_key="",
                model="",
                timeout=60.0,
                hash_dim=128,
                query_topic="Math->Algebra",
                topic_bonus=0.05,
            )
            self.assertEqual(len(scored), 1)
            self.assertEqual(scored[0][0]["memory_id"], bank_rows[0]["memory_id"])

            traj2.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "idx": 2,
                                "problem": "Solve x + 7 = 10",
                                "topic": "Math->Algebra",
                                "topic_key": "Math_Algebra",
                                "difficulty": "easy",
                                "student_response": "Subtract 7 from both sides, so x=3. Verify.",
                                "is_correct": True,
                                "ground_truth": "3",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "idx": 3,
                                "problem": "Solve x^2 = 9",
                                "topic": "Math->Algebra",
                                "topic_key": "Math_Algebra",
                                "difficulty": "easy",
                                "student_response": "x = 3 only.",
                                "is_correct": False,
                                "ground_truth": "3,-3",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            evolve_args = argparse.Namespace(
                memory_bank=str(memory_bank),
                trajectories=str(traj2),
                output_memory_bank=str(memory_bank_v2),
                existing_embeddings=str(embeddings),
                output_embeddings=str(embeddings_v2),
                teacher_base_url="",
                teacher_api_key="",
                teacher_model="",
                embed_backend="hash",
                embed_base_url="",
                embed_api_key="",
                embed_model="",
                timeout=60.0,
                temperature=0.2,
                max_tokens=512,
                hash_dim=128,
                similarity_threshold=0.98,
                fail_on_error=True,
            )
            with mock.patch("baselines.ReasoningBankMath.evolve_memory.chat_complete", side_effect=fake_teacher):
                rc = run_evolve_memory(evolve_args)
            self.assertEqual(rc, 0)

            merged_rows = read_jsonl(memory_bank_v2)
            self.assertEqual(len(merged_rows), 2)
            merged_first = [r for r in merged_rows if r["query"] == "Solve x + 2 = 5"][0]
            self.assertEqual(merged_first["duplicate_count"], 1)
            self.assertEqual(len(merged_first["provenance"]), 2)
            self.assertTrue(embeddings_v2.is_file())


if __name__ == "__main__":
    unittest.main()
