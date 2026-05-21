from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from baselines.ReasoningBankMath.build_embeddings import run_build_embeddings
from baselines.ReasoningBankMath.build_memory import run_build_memory
from baselines.ReasoningBankMath.compact_memory import run_compact_memory
from baselines.ReasoningBankMath.evolve_memory import run_evolve_memory
from baselines.ReasoningBankMath.prepare_prompt_data import run_prepare_prompt_data
from baselines.ReasoningBankMath.refine_memory import run_refine_memory
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

    def test_compact_memory_dedupes_existing_bank(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            memory_bank = tmp / "memory_bank.jsonl"
            compact_bank = tmp / "memory_bank_compact.jsonl"
            compact_emb = tmp / "memory_embeddings_compact.jsonl"
            rows = [
                {
                    "memory_id": "mem_a",
                    "query": "Solve x+2=5",
                    "topic": "Math->Algebra",
                    "topic_key": "Math_Algebra",
                    "status": "success",
                    "memory_items": [
                        {
                            "title": "Isolate the variable",
                            "description": "Reverse operations carefully.",
                            "content": "Undo additions step by step and verify in the original equation.",
                        }
                    ],
                    "provenance": [{"source_idx": 1}],
                },
                {
                    "memory_id": "mem_b",
                    "query": "Solve x+7=10",
                    "topic": "Math->Algebra",
                    "topic_key": "Math_Algebra",
                    "status": "success",
                    "memory_items": [
                        {
                            "title": "Isolate the variable",
                            "description": "Reverse operations carefully.",
                            "content": "Undo additions step by step and verify in the original equation.",
                        }
                    ],
                    "provenance": [{"source_idx": 2}],
                },
            ]
            memory_bank.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                memory_bank=str(memory_bank),
                output_memory_bank=str(compact_bank),
                existing_embeddings="",
                output_embeddings=str(compact_emb),
                embed_backend="hash",
                embed_base_url="",
                embed_api_key="",
                embed_model="",
                timeout=60.0,
                hash_dim=128,
                similarity_threshold=0.98,
            )
            rc = run_compact_memory(args)
            self.assertEqual(rc, 0)
            merged = read_jsonl(compact_bank)
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["duplicate_count"], 1)
            self.assertEqual(len(merged[0]["provenance"]), 2)
            self.assertTrue(compact_emb.is_file())

    def test_refine_memory_rewrites_cluster_with_llm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            memory_bank = tmp / "memory_bank.jsonl"
            refined_bank = tmp / "memory_bank_refined.jsonl"
            refined_emb = tmp / "memory_embeddings_refined.jsonl"
            rows = [
                {
                    "memory_id": "mem_a",
                    "query": "Solve x+2=5",
                    "topic": "Math->Algebra",
                    "topic_key": "Math_Algebra",
                    "status": "success",
                    "embedding_text": "Question: Solve x+2=5\nIsolate the variable",
                    "memory_items": [
                        {
                            "title": "Isolate the variable",
                            "description": "Reverse operations carefully.",
                            "content": "Undo additions step by step and verify in the original equation.",
                        }
                    ],
                    "provenance": [{"source_idx": 1}],
                },
                {
                    "memory_id": "mem_b",
                    "query": "Solve x+7=10",
                    "topic": "Math->Algebra",
                    "topic_key": "Math_Algebra",
                    "status": "success",
                    "embedding_text": "Question: Solve x+7=10\nIsolate the variable",
                    "memory_items": [
                        {
                            "title": "Undo inverse operations",
                            "description": "Peel operations off carefully.",
                            "content": "Reverse the outer operation, then verify the candidate by substitution.",
                        }
                    ],
                    "provenance": [{"source_idx": 2}],
                },
            ]
            memory_bank.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                memory_bank=str(memory_bank),
                output_memory_bank=str(refined_bank),
                existing_embeddings="",
                output_embeddings=str(refined_emb),
                teacher_base_url="http://127.0.0.1:8000/v1",
                teacher_api_key="",
                teacher_model="mock-model",
                teacher_backend="chat",
                rollout_server_urls="",
                rollout_host="",
                rollout_base_port="",
                rollout_n_servers="",
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
                cluster_similarity_threshold=0.0,
                refine_singletons=False,
                fail_on_error=True,
            )

            def fake_teacher(messages, **kwargs):
                return _teacher_output(
                    "Reverse operations in order",
                    "When solving simple equations, remove outer operations one at a time.",
                    "Undo the operations in reverse order and always substitute the result back to verify it satisfies the original equation.",
                )

            with mock.patch("baselines.ReasoningBankMath.refine_memory.chat_complete", side_effect=fake_teacher):
                rc = run_refine_memory(args)
            self.assertEqual(rc, 0)
            merged = read_jsonl(refined_bank)
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["memory_items"][0]["title"], "Reverse operations in order")
            self.assertEqual(len(merged[0]["provenance"]), 2)
            self.assertTrue(refined_emb.is_file())

    def test_prepare_prompt_data_outputs_jsonl_and_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_jsonl = tmp / "temp_data.jsonl"
            memory_bank = tmp / "memory_bank.jsonl"
            embeddings = tmp / "memory_embeddings.jsonl"
            out_jsonl = tmp / "out.jsonl"
            out_parquet = tmp / "out.parquet"

            input_jsonl.write_text(
                json.dumps(
                    {
                        "question": "Solve x + 2 = 5",
                        "gt": "3",
                        "extra_info": {"topic": "Math->Algebra", "difficulty": 1},
                        "idx": 10,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            memory_bank.write_text(
                json.dumps(
                    {
                        "memory_id": "mem_001",
                        "query": "Solve x + 7 = 10",
                        "topic": "Math->Algebra",
                        "topic_key": "Math_Algebra",
                        "status": "success",
                        "embedding_text": "Question: Solve x + 7 = 10\nIsolate the variable",
                        "memory_items": [
                            {
                                "title": "Isolate the variable",
                                "description": "Undo operations in reverse order.",
                                "content": "Reverse the outer operation and substitute the result back into the original equation.",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            embeddings.write_text(
                json.dumps(
                    {
                        "memory_id": "mem_001",
                        "text": "Question: Solve x + 7 = 10\nIsolate the variable",
                        "embedding": [1.0, 0.0, 0.0, 0.0],
                        "backend": "hash",
                        "model": "hash-4",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            args = argparse.Namespace(
                input_jsonl=str(input_jsonl),
                memory_bank=str(memory_bank),
                start=0,
                end=None,
                top_k=1,
                retriever_url="http://127.0.0.1:8766",
                mode="embedding",
                retrieve_lambda=0.5,
                data_source="temp_data",
                output_jsonl=str(out_jsonl),
                output_parquet=str(out_parquet),
                keep_raw_prompt=False,
                keep_raw_row=False,
                fail_on_retrieve_error=True,
            )

            with mock.patch(
                "baselines.ReasoningBankMath.prepare_prompt_data._post_rank",
                return_value=[0],
            ):
                rc = run_prepare_prompt_data(args)
            self.assertEqual(rc, 0)
            lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["problem"], "Solve x + 2 = 5")
            self.assertEqual(rec["ground_truth"], "3")
            self.assertEqual(rec["data_source"], "temp_data")
            self.assertEqual(rec["extra_info"]["retrieved_memory_count"], 1)
            self.assertEqual(rec["extra_info"]["retrieved_memory_ids"], ["mem_001"])
            self.assertIn("Solve x + 2 = 5", rec["prompt"][0]["content"])
            self.assertIn("Isolate the variable", rec["prompt"][0]["content"])
            self.assertTrue(out_parquet.is_file())

    def test_fix_eval_data_source_and_problem_repairs_existing_eval_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        script = repo_root / "baselines" / "ReasoningBankMath" / "scripts" / "fix_eval_data_source_and_problem.sh"

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = tmp / "eval" / "step_200"
            root.mkdir(parents=True)

            temp_input = tmp / "temp_data.jsonl"
            temp_problem = "Solve x + 2 = 5"
            temp_input.write_text(
                json.dumps(
                    {
                        "data_source": "AIME24",
                        "problem": temp_problem,
                        "reward_model": {"ground_truth": ["3"]},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            greedy_input = tmp / "greedy_data.jsonl"
            greedy_problem = "What is 6 * 7?"
            greedy_input.write_text(
                json.dumps(
                    {
                        "data_source": "AMC23",
                        "problem": greedy_problem,
                        "reward_model": {"ground_truth": ["42"]},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            pd.DataFrame(
                [
                    {
                        "data_source": "temp_data",
                        "problem": temp_problem,
                        "extra_info": {"problem": temp_problem},
                        "reward_model": {"ground_truth": ["3"]},
                        "responses": ["boxed 3"],
                        "rule_scores": [1.0],
                        "checked_scores": [1.0],
                    }
                ]
            ).to_parquet(root / "temp_data_responses.parquet", index=False)

            pd.DataFrame(
                [
                    {
                        "data_source": "greedy_data",
                        "problem": greedy_problem,
                        "extra_info": {"problem": greedy_problem},
                        "reward_model": {"ground_truth": ["42"]},
                        "responses": ["boxed 42"],
                        "rule_scores": [1.0],
                        "checked_scores": [1.0],
                    }
                ]
            ).to_parquet(root / "greedy_data_responses.parquet", index=False)

            (root / "temp_data_meta.json").write_text(
                json.dumps({"model_name": "mock-model", "n_samples": 1, "temperature": 0.7}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "greedy_data_meta.json").write_text(
                json.dumps({"model_name": "mock-model", "n_samples": 1, "temperature": 0.0}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "temp_data_Overall_results.jsonl").write_text(
                json.dumps({"data_source": "temp_data", "model": "mock-model", "checked_mean@1": "0.00"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (root / "greedy_data_Overall_results.jsonl").write_text(
                json.dumps({"data_source": "greedy_data", "model": "mock-model", "checked_mean@1": "0.00"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    "bash",
                    str(script),
                    "--root",
                    str(root),
                    "--temp-input",
                    str(temp_input),
                    "--greedy-input",
                    str(greedy_input),
                ],
                cwd=repo_root,
                check=True,
            )

            temp_df = pd.read_parquet(root / "temp_data_responses.parquet")
            greedy_df = pd.read_parquet(root / "greedy_data_responses.parquet")
            self.assertEqual(temp_df.iloc[0]["data_source"], "AIME24")
            self.assertEqual(greedy_df.iloc[0]["data_source"], "AMC23")
            self.assertEqual(temp_df.iloc[0]["extra_info"]["data_source"], "AIME24")
            self.assertEqual(greedy_df.iloc[0]["extra_info"]["data_source"], "AMC23")

            temp_overall = [
                json.loads(line)
                for line in (root / "temp_data_Overall_results.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            greedy_overall = [
                json.loads(line)
                for line in (root / "greedy_data_Overall_results.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(temp_overall[0]["data_source"], "AIME24")
            self.assertEqual(greedy_overall[0]["data_source"], "AMC23")
            self.assertEqual(temp_overall[0]["checked_mean@1"], "100.00")
            self.assertEqual(greedy_overall[0]["checked_mean@1"], "100.00")

            aggregate = json.loads((root / "aggregated_eval_results.json").read_text(encoding="utf-8"))
            self.assertIn("AIME24", aggregate["math_datasets"])
            self.assertIn("AMC23", aggregate["math_datasets"])

            all_steps = json.loads((root.parent / "all_steps_aggregated_results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(all_steps), 1)
            self.assertEqual(all_steps[0]["step"], 200)

            table_csv = root.parent / "eval_results_table.csv"
            table_md = root.parent / "eval_results_table.md"
            self.assertTrue(table_csv.is_file())
            self.assertTrue(table_md.is_file())

            table_df = pd.read_csv(table_csv)
            self.assertIn("AIME24", table_df.columns)
            self.assertIn("AMC23", table_df.columns)

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
                teacher_base_url="http://127.0.0.1:8000/v1",
                teacher_api_key="",
                teacher_model="mock-model",
                teacher_backend="chat",
                rollout_server_urls="",
                rollout_host="",
                rollout_base_port="",
                rollout_n_servers="",
                timeout=60.0,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
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
                teacher_base_url="http://127.0.0.1:8000/v1",
                teacher_api_key="",
                teacher_model="mock-model",
                teacher_backend="chat",
                rollout_server_urls="",
                rollout_host="",
                rollout_base_port="",
                rollout_n_servers="",
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
