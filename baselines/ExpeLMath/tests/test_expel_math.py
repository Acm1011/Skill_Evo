from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.ExpeLMath.build_embeddings import run_build_embeddings
from baselines.ExpeLMath.build_memory import group_trajectories, run_build_memory
from baselines.ExpeLMath.eval_with_memory import run_eval
from baselines.ExpeLMath.evolve_memory import run_evolve_memory
from baselines.ExpeLMath.memory_bank import dedupe_records
from baselines.ExpeLMath.memory_parser import parse_teacher_output
from baselines.ExpeLMath.prepare_prompt_data import run_prepare_prompt_data
from baselines.ExpeLMath.retrieve_memory import retrieve_records
from baselines.ReasoningBankMath.io_utils import read_jsonl


def _teacher_output(raw_rule: str, memory_type: str, title: str, desc: str, content: str) -> str:
    return (
        "# Raw Rule\n"
        f"{raw_rule}\n\n"
        "# Memory Item 1\n"
        f"## Title {title}\n"
        f"## Description {desc}\n"
        f"## Content {content}\n"
        f"## Type {memory_type}\n"
    )


class ExpelMathTests(unittest.TestCase):
    def test_parse_teacher_output_standard_and_fallback(self) -> None:
        parsed = parse_teacher_output(
            _teacher_output(
                "Compare the successful method against the failed branch.",
                "compare_rule",
                "Contrast trajectories",
                "Look for the successful pivot.",
                "Identify which transformation fixes the error and reuse it.",
            ),
            default_memory_type="compare_rule",
        )
        self.assertIn("raw_rule", parsed)
        self.assertEqual(parsed["memory_items"][0]["memory_type"], "compare_rule")
        recovered = parse_teacher_output("Always verify the final candidate in the original equation.", default_memory_type="success_rule")
        self.assertTrue(recovered["raw_rule"])
        self.assertEqual(recovered["memory_items"][0]["memory_type"], "success_rule")

    def test_group_trajectories_buckets_compare_success_failure(self) -> None:
        rows = [
            {"idx": 1, "problem": "p1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "ok", "is_correct": True, "ground_truth": "1"},
            {"idx": 2, "problem": "p1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "bad", "is_correct": False, "ground_truth": "1"},
            {"idx": 3, "problem": "p1", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "bad2", "is_correct": False, "ground_truth": "1"},
            {"idx": 4, "problem": "p2", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "ok1", "is_correct": True, "ground_truth": "2"},
            {"idx": 5, "problem": "p2", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "ok2", "is_correct": True, "ground_truth": "2"},
            {"idx": 6, "problem": "p2", "topic": "Math->Algebra", "topic_key": "Math_Algebra", "student_response": "ok3", "is_correct": True, "ground_truth": "2"},
            {"idx": 7, "problem": "p3", "topic": "Math->Geometry", "topic_key": "Math_Geometry", "student_response": "bad1", "is_correct": False, "ground_truth": "3"},
            {"idx": 8, "problem": "p3", "topic": "Math->Geometry", "topic_key": "Math_Geometry", "student_response": "bad2", "is_correct": None, "ground_truth": "3"},
        ]
        groups = group_trajectories(rows, group_by="problem", max_success_group=4, max_failure_group=4)
        kinds = sorted(g["memory_type"] for g in groups)
        self.assertEqual(kinds, ["compare_rule", "failure_rule", "failure_rule", "success_rule"])

        compare_group = next(g for g in groups if g["memory_type"] == "compare_rule")
        self.assertEqual(len(compare_group["rows"]), 2)
        self.assertEqual(sum(1 for r in compare_group["rows"] if r["is_correct"] is True), 1)
        self.assertEqual(sum(1 for r in compare_group["rows"] if r["is_correct"] is not True), 1)

        success_group = next(g for g in groups if g["memory_type"] == "success_rule")
        self.assertEqual(len(success_group["rows"]), 2)
        self.assertTrue(all(r["is_correct"] is True for r in success_group["rows"]))

        failure_groups = [g for g in groups if g["memory_type"] == "failure_rule"]
        self.assertEqual(len(failure_groups), 2)
        self.assertTrue(all(len(g["rows"]) == 2 for g in failure_groups))
        self.assertTrue(all(all(r["is_correct"] is not True for r in g["rows"]) for g in failure_groups))

    def test_dedupe_respects_memory_type(self) -> None:
        existing = [
            {
                "memory_id": "mem_old",
                "memory_type": "success_rule",
                "embedding_text": "Question: q\nsame",
                "memory_items": [
                    {
                        "title": "Isolate variables",
                        "description": "d",
                        "content": "Undo inverse operations step by step.",
                        "memory_type": "success_rule",
                    }
                ],
                "provenance": [{"source_idx": 1}],
                "duplicate_count": 0,
            }
        ]
        incoming = [
            {
                "memory_id": "mem_new",
                "memory_type": "failure_rule",
                "embedding_text": "Question: q\nsame",
                "memory_items": [
                    {
                        "title": "Isolate variables",
                        "description": "d",
                        "content": "Undo inverse operations step by step.",
                        "memory_type": "failure_rule",
                    }
                ],
                "provenance": [{"source_idx": 2}],
                "duplicate_count": 0,
            }
        ]
        merged, dup_map = dedupe_records(existing, incoming, similarity_threshold=0.98)
        self.assertEqual(len(merged), 2)
        self.assertEqual(dup_map, {})

    def test_retrieve_prioritizes_topic_and_rule_type(self) -> None:
        memory_rows = [
            {
                "memory_id": "m1",
                "topic_key": "Math_Algebra",
                "memory_type": "compare_rule",
                "raw_rule": "Compare successful and failed transformations.",
                "embedding_text": "Solve x + 2 = 5 Compare successful and failed transformations.",
            },
            {
                "memory_id": "m2",
                "topic_key": "Math_Geometry",
                "memory_type": "failure_rule",
                "raw_rule": "Do not drop constraints.",
                "embedding_text": "Solve x + 2 = 5 Compare successful and failed transformations.",
            },
        ]
        embedding_rows = [
            {"memory_id": "m1", "embedding": [1.0, 0.0]},
            {"memory_id": "m2", "embedding": [1.0, 0.0]},
        ]
        with mock.patch(
            "baselines.ExpeLMath.retrieve_memory.embed_texts",
            return_value=[[1.0, 0.0]],
        ):
            scored = retrieve_records(
                question="Solve x + 7 = 10",
                memory_rows=memory_rows,
                embedding_rows=embedding_rows,
                top_k=2,
                backend="hash",
                base_url="",
                api_key="",
                model="",
                timeout=60.0,
                hash_dim=2,
                query_topic="Math->Algebra",
                topic_bonus=0.05,
            )
        self.assertEqual(scored[0][0]["memory_id"], "m1")

    def test_prepare_prompt_data_only_changes_prompt_content(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            memory_bank = tmp / "memory_bank.jsonl"
            input_jsonl = tmp / "input.jsonl"
            out_jsonl = tmp / "greedy_data_skills.jsonl"
            out_parquet = tmp / "greedy_data_skills.parquet"

            memory_bank.write_text(
                json.dumps(
                    {
                        "memory_id": "m1",
                        "topic": "Math->Algebra",
                        "topic_key": "Math_Algebra",
                        "status": "success",
                        "memory_type": "success_rule",
                        "raw_rule": "Use inverse operations consistently.",
                        "embedding_text": "Question: Solve x + 2 = 5\nTopic: Math->Algebra\nRaw Rule: Use inverse operations consistently.",
                        "memory_items": [],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            original_row = {
                "ability": "math",
                "data_source": "AMC23",
                "problem": "Solve x + 2 = 5",
                "prompt": [{"role": "user", "content": "Original prompt"}],
                "reward_model": {"ground_truth": "3", "style": "rule"},
                "extra_info": {"idx": 7, "problem": "Solve x + 2 = 5", "solution": "3", "split": "test"},
            }
            input_jsonl.write_text(json.dumps(original_row, ensure_ascii=False) + "\n", encoding="utf-8")

            with mock.patch(
                "baselines.ExpeLMath.prepare_prompt_data._post_rank",
                return_value=[0],
            ):
                args = argparse.Namespace(
                    input_jsonl=str(input_jsonl),
                    memory_bank=str(memory_bank),
                    retriever_url="http://127.0.0.1:8766",
                    start=0,
                    end=None,
                    top_k=5,
                    mode="embedding",
                    retrieve_lambda=0.5,
                    output_jsonl=str(out_jsonl),
                    output_parquet=str(out_parquet),
                    fail_on_retrieve_error=True,
                )
                self.assertEqual(run_prepare_prompt_data(args), 0)

            rows = read_jsonl(out_jsonl)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["ability"], original_row["ability"])
            self.assertEqual(row["data_source"], original_row["data_source"])
            self.assertEqual(row["reward_model"], original_row["reward_model"])
            self.assertEqual(row["extra_info"], original_row["extra_info"])
            self.assertEqual(row["problem"], original_row["problem"])
            self.assertEqual(row["prompt"][0]["role"], "user")
            self.assertNotEqual(row["prompt"][0]["content"], original_row["prompt"][0]["content"])
            self.assertIn("SKILL:", row["prompt"][0]["content"])
            self.assertIn("Question: Solve x + 2 = 5", row["prompt"][0]["content"])

    def test_end_to_end_build_retrieve_eval_evolve(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            traj1 = tmp / "traj1.jsonl"
            bank = tmp / "memory_bank.jsonl"
            emb = tmp / "emb.jsonl"
            eval_data = tmp / "eval.jsonl"
            traj2 = tmp / "traj2.jsonl"
            bank_v2 = tmp / "memory_bank_v2.jsonl"
            emb_v2 = tmp / "emb_v2.jsonl"

            traj1.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "idx": 1,
                                "problem": "Solve x + 2 = 5",
                                "topic": "Math->Algebra",
                                "topic_key": "Math_Algebra",
                                "student_response": "Subtract 2 from both sides, x=3.",
                                "is_correct": True,
                                "ground_truth": "3",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "idx": 2,
                                "problem": "Solve x + 2 = 5",
                                "topic": "Math->Algebra",
                                "topic_key": "Math_Algebra",
                                "student_response": "Add 2 to get x=7.",
                                "is_correct": False,
                                "ground_truth": "3",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "idx": 3,
                                "problem": "Solve x + 7 = 10",
                                "topic": "Math->Algebra",
                                "topic_key": "Math_Algebra",
                                "student_response": "Subtract 7 from both sides, x=3.",
                                "is_correct": True,
                                "ground_truth": "3",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "idx": 4,
                                "problem": "Solve x + 7 = 10",
                                "topic": "Math->Algebra",
                                "topic_key": "Math_Algebra",
                                "student_response": "Use inverse operations: x = 10 - 7 = 3.",
                                "is_correct": True,
                                "ground_truth": "3",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_teacher(messages, **kwargs):
                user = messages[-1]["content"]
                if "Failed trajectories" in user and "Successful trajectories" in user:
                    return _teacher_output(
                        "Compare each algebra step against the successful branch and keep inverse operations consistent.",
                        "compare_rule",
                        "Contrast algebra branches",
                        "Successful solutions preserve inverse operations.",
                        "When two trajectories diverge, identify the first incorrect transformation and replace it with the successful inverse operation.",
                    )
                return _teacher_output(
                    "Isolate the variable using inverse operations and verify the result.",
                    "success_rule",
                    "Isolate the variable",
                    "Use inverse operations carefully.",
                    "Undo additions or multiplications step by step and verify the candidate in the original equation.",
                )

            build_args = argparse.Namespace(
                trajectories=str(traj1),
                output=str(bank),
                teacher_base_url="http://127.0.0.1:8000/v1",
                teacher_api_key="",
                teacher_model="mock-model",
                teacher_backend="chat",
                rollout_server_urls="",
                rollout_host="",
                rollout_base_port="",
                rollout_n_servers="",
                group_by="problem",
                max_success_group=4,
                max_failure_group=4,
                timeout=60.0,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                max_tokens=512,
                fail_on_error=True,
            )
            with mock.patch("baselines.ExpeLMath.build_memory.chat_complete", side_effect=fake_teacher):
                self.assertEqual(run_build_memory(build_args), 0)
            bank_rows = read_jsonl(bank)
            self.assertEqual(len(bank_rows), 2)
            self.assertEqual(sorted(row["memory_type"] for row in bank_rows), ["compare_rule", "success_rule"])

            emb_args = argparse.Namespace(
                memory_bank=str(bank),
                output=str(emb),
                existing_embeddings="",
                backend="hash",
                embed_base_url="",
                embed_api_key="",
                embed_model="",
                timeout=60.0,
                hash_dim=128,
            )
            self.assertEqual(run_build_embeddings(emb_args), 0)
            emb_rows = read_jsonl(emb)
            self.assertEqual(len(emb_rows), 2)

            eval_data.write_text(
                json.dumps(
                    {
                        "extra_info": {"problem": "Solve x + 10 = 13", "topic": "Math->Algebra"},
                        "reward_model": {"ground_truth": ["3"]},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            class _MockOut:
                def __init__(self, text: str) -> None:
                    self.outputs = [type("Item", (), {"text": text})()]

            with mock.patch(
                "baselines.ExpeLMath.eval_with_memory.VLLMHTTPClient.generate_sync",
                return_value=[_MockOut("Subtract 10 from both sides, so the answer is \\boxed{3}.")],
            ):
                eval_args = argparse.Namespace(
                    deepmath_jsonl=str(eval_data),
                    memory_bank=str(bank),
                    embeddings=str(emb),
                    output=str(tmp / "eval_out.jsonl"),
                    start=0,
                    end=1,
                    top_k=2,
                    retrieval_mode="rules",
                    backend="hash",
                    embed_base_url="",
                    embed_api_key="",
                    embed_model="",
                    topic_bonus=0.05,
                    hash_dim=128,
                    server_urls=["http://127.0.0.1:8765"],
                    served_model_name="mock-student",
                    max_tokens=128,
                    temperature=0.0,
                    top_p=1.0,
                    timeout=60.0,
                    max_retries=1,
                    max_concurrent=0,
                )
                self.assertEqual(run_eval(eval_args), 0)
            eval_rows = read_jsonl(tmp / "eval_out.jsonl")
            self.assertEqual(len(eval_rows), 1)
            self.assertTrue(eval_rows[0]["retrieved_memory_ids"])
            self.assertTrue(eval_rows[0]["is_correct"])

            traj2.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "idx": 5,
                                "problem": "Solve x + 20 = 23",
                                "topic": "Math->Algebra",
                                "topic_key": "Math_Algebra",
                                "student_response": "Subtract 20 from both sides, x=3.",
                                "is_correct": True,
                                "ground_truth": "3",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "idx": 6,
                                "problem": "Solve x + 20 = 23",
                                "topic": "Math->Algebra",
                                "topic_key": "Math_Algebra",
                                "student_response": "Apply the inverse operation and get x = 23 - 20 = 3.",
                                "is_correct": True,
                                "ground_truth": "3",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            evolve_args = argparse.Namespace(
                memory_bank=str(bank),
                trajectories=str(traj2),
                output_memory_bank=str(bank_v2),
                existing_embeddings=str(emb),
                output_embeddings=str(emb_v2),
                teacher_base_url="http://127.0.0.1:8000/v1",
                teacher_api_key="",
                teacher_model="mock-model",
                teacher_backend="chat",
                rollout_server_urls="",
                rollout_host="",
                rollout_base_port="",
                rollout_n_servers="",
                group_by="problem",
                max_success_group=4,
                max_failure_group=4,
                embed_backend="hash",
                embed_base_url="",
                embed_api_key="",
                embed_model="",
                timeout=60.0,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                max_tokens=512,
                hash_dim=128,
                similarity_threshold=0.98,
                fail_on_error=True,
            )
            with mock.patch("baselines.ExpeLMath.evolve_memory.chat_complete", side_effect=fake_teacher):
                self.assertEqual(run_evolve_memory(evolve_args), 0)
            merged_rows = read_jsonl(bank_v2)
            self.assertEqual(len(merged_rows), 2)
            success_row = next(row for row in merged_rows if row["memory_type"] == "success_rule")
            self.assertGreaterEqual(success_row["duplicate_count"], 1)
            self.assertGreaterEqual(len(success_row["provenance"]), 2)


if __name__ == "__main__":
    unittest.main()
