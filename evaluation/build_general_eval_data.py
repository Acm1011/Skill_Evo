#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
from typing import Iterable

import pandas as pd


PROMPT_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}. You will be given some reusable problem-solving skills, and you can refer to them during the reasoning.
SKILL: {skill}
Question: {question}
""".strip()


def make_prompt(question: str, skill: str = "") -> list[dict]:
    return [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(skill=skill, question=question),
        }
    ]


def format_options(options: Iterable[str]) -> str:
    option_lines = []
    for idx, option in enumerate(options):
        option_lines.append(f"{chr(65 + idx)}. {option}")
    return "\n".join(option_lines)


def build_bbeh_records(data_dir: str, start_idx: int) -> list[dict]:
    path = os.path.join(data_dir, "bbeh-eval", "train.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for offset, line in enumerate(f):
            item = json.loads(line)
            question = item["question"]
            answer = str(item["answer"])
            task = item.get("task", "unknown")
            records.append(
                {
                    "data_source": "bbeh",
                    "problem": question,
                    "prompt": make_prompt(question),
                    "ability": "general",
                    "reward_model": {
                        "ground_truth": [answer],
                        "style": "text",
                    },
                    "extra_info": {
                        "idx": start_idx + offset,
                        "problem": question,
                        "question": question,
                        "answer": answer,
                        "task": task,
                        "benchmark": "bbeh",
                        "split": "train",
                        "skill_id": [],
                    },
                }
            )
    return records


def build_mmlupro_records(data_dir: str, start_idx: int) -> list[dict]:
    path = os.path.join(data_dir, "MMLU-Pro", "data", "test-00000-of-00001.parquet")
    df = pd.read_parquet(path)
    records = []
    for offset, row in enumerate(df.to_dict(orient="records")):
        options = [str(x) for x in row["options"]]
        question = f'{row["question"]}\n\nOptions:\n{format_options(options)}'
        answer = str(row["answer"])
        records.append(
            {
                "data_source": "mmlupro",
                "problem": question,
                "prompt": make_prompt(question),
                "ability": "general",
                "reward_model": {
                    "ground_truth": [answer],
                    "style": "multichoice",
                },
                "extra_info": {
                    "idx": start_idx + offset,
                    "problem": question,
                    "question": row["question"],
                    "options": options,
                    "answer": answer,
                    "answer_index": int(row["answer_index"]),
                    "category": row.get("category"),
                    "src": row.get("src"),
                    "benchmark": "mmlupro",
                    "split": "test",
                    "skill_id": [],
                },
            }
        )
    return records


def build_supergpqa_records(data_dir: str, start_idx: int) -> list[dict]:
    path = os.path.join(data_dir, "SuperGPQA", "SuperGPQA-all.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for offset, line in enumerate(f):
            row = json.loads(line)
            options = [str(x) for x in row["options"]]
            question = f'{row["question"]}\n\nOptions:\n{format_options(options)}'
            answer = str(row["answer_letter"])
            records.append(
                {
                    "data_source": "supergpqa",
                    "problem": question,
                    "prompt": make_prompt(question),
                    "ability": "general",
                    "reward_model": {
                        "ground_truth": [answer],
                        "style": "multichoice",
                    },
                    "extra_info": {
                        "idx": start_idx + offset,
                        "problem": question,
                        "question": row["question"],
                        "options": options,
                        "answer": str(row["answer"]),
                        "answer_letter": answer,
                        "discipline": row.get("discipline"),
                        "field": row.get("field"),
                        "subfield": row.get("subfield"),
                        "difficulty": row.get("difficulty"),
                        "benchmark": "supergpqa",
                        "split": "train",
                        "skill_id": [],
                    },
                }
            )
    return records


def build_records(data_dir: str) -> list[dict]:
    records = []
    builders = [
        build_bbeh_records,
        build_mmlupro_records,
        build_supergpqa_records,
    ]
    next_idx = 0
    for builder in builders:
        chunk = builder(data_dir, next_idx)
        records.extend(chunk)
        next_idx += len(chunk)
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged general benchmark parquet/jsonl for evaluation")
    parser.add_argument("--data_dir", type=str, default="/home/ycy/sdi/data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ycy/sdi/Skill_Evo/evaluation/.eval_custom_data/general_benchmarks",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=["greedy_general", "temp_general"],
        help="Output dataset basenames without extension",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = build_records(args.data_dir)
    df = pd.DataFrame(records)

    for dataset_name in args.dataset_names:
        parquet_path = os.path.join(args.output_dir, f"{dataset_name}.parquet")
        jsonl_path = os.path.join(args.output_dir, f"{dataset_name}.jsonl")
        df.to_parquet(parquet_path, index=False)
        write_jsonl(records, jsonl_path)
        print(f"Saved {len(records)} records to:")
        print(f"  {parquet_path}")
        print(f"  {jsonl_path}")

    counts = df["data_source"].value_counts().to_dict()
    print("Per-source counts:", counts)


if __name__ == "__main__":
    main()
