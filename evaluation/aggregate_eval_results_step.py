#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聚合按 step 组织的评测结果。

支持两类结果：
1. 数学评测：temp_data_Overall_results.jsonl / greedy_data_Overall_results.jsonl
2. 额外评测：aggregated_eval_results.json（由 bbeh/mmlupro/supergpqa 聚合得到）
"""
import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


def get_all_steps(save_path_dir: str, base_model_dir: Optional[str] = None) -> List[Tuple[int, str]]:
    steps = []

    if base_model_dir and os.path.exists(base_model_dir):
        steps.append((0, base_model_dir))

    if os.path.exists(save_path_dir):
        for name in os.listdir(save_path_dir):
            path = os.path.join(save_path_dir, name)
            if not os.path.isdir(path):
                continue
            match = re.match(r"^step_(\d+)$", name)
            if match:
                steps.append((int(match.group(1)), path))

    steps.sort(key=lambda x: x[0])
    return steps


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_records(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    try:
        return pd.read_json(path, lines=True).to_dict(orient="records")
    except ValueError:
        return []


def to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate_math_results(step_dir: str) -> Dict[str, dict]:
    datasets = {}

    math_files = {
        "greedy_data": os.path.join(step_dir, "greedy_data_Overall_results.jsonl"),
        "temp_data": os.path.join(step_dir, "temp_data_Overall_results.jsonl"),
    }

    for eval_name, path in math_files.items():
        records = load_jsonl_records(path)
        if not records:
            continue

        for item in records:
            data_source = item.get("data_source", eval_name)
            datasets[data_source] = {
                "eval_name": eval_name,
                "data_source": data_source,
                "n_samples": item.get("n_samples"),
                "temperature": item.get("temperature"),
                "rule@first": to_float(item.get("rule@first")),
                "checked@first": to_float(item.get("checked@first")),
                "rule_mean": to_float(item.get(f"rule_mean@{item.get('n_samples')}")),
                "checked_mean": to_float(item.get(f"checked_mean@{item.get('n_samples')}")),
                "checked_sample_mean": to_float(item.get(f"checked_sample_mean@{item.get('n_samples')}")),
            }

    return datasets


def aggregate_additional_results(step_dir: str) -> Dict[str, dict]:
    aggregated_path = os.path.join(step_dir, "aggregated_eval_results.json")
    data = load_json(aggregated_path)
    if not data:
        return {}
    return data.get("datasets", {})


def aggregate_step_results(step_dir: str, step_num: int) -> Optional[dict]:
    math_datasets = aggregate_math_results(step_dir)
    additional_datasets = aggregate_additional_results(step_dir)

    if not math_datasets and not additional_datasets:
        return None

    math_checked_means = [
        item["checked_mean"] for item in math_datasets.values()
        if item.get("checked_mean") is not None
    ]
    math_avg = round(sum(math_checked_means) / len(math_checked_means), 2) if math_checked_means else None

    general_accuracies = []
    for item in additional_datasets.values():
        accuracy = item.get("accuracy")
        if accuracy is None:
            accuracy = item.get("micro_accuracy")
        if accuracy is not None:
            accuracy = to_float(accuracy)
            if accuracy <= 1.0:
                accuracy *= 100.0
            general_accuracies.append(round(accuracy, 2))

    general_avg = round(sum(general_accuracies) / len(general_accuracies), 2) if general_accuracies else None

    overall_values = []
    if math_avg is not None:
        overall_values.append(math_avg)
    if general_avg is not None:
        overall_values.append(general_avg)
    overall_avg = round(sum(overall_values) / len(overall_values), 2) if overall_values else None

    return {
        "step": step_num,
        "step_dir": step_dir,
        "math_datasets": math_datasets,
        "additional_datasets": additional_datasets,
        "summary": {
            "math_avg": math_avg,
            "general_avg": general_avg,
            "overall_avg": overall_avg,
        },
    }


def aggregate_all_steps(save_path_dir: str, base_model_dir: Optional[str] = None, output_file: Optional[str] = None) -> None:
    steps = get_all_steps(save_path_dir, base_model_dir)

    if not steps:
        print(f"Warning: No step directories found in {save_path_dir}")
        return

    print(f"Found {len(steps)} steps: {[s[0] for s in steps]}")

    all_results = []
    for step_num, step_dir in steps:
        print(f"Processing step {step_num}: {step_dir}")
        result = aggregate_step_results(step_dir, step_num)

        if result is None:
            print(f"  Warning: No results found for step {step_num}")
            continue

        step_output_file = os.path.join(step_dir, "aggregated_eval_results.json")
        with open(step_output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Saved to: {step_output_file}")

        all_results.append(result)

    if output_file is None:
        output_file = os.path.join(save_path_dir, "all_steps_aggregated_results.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll steps aggregated results saved to: {output_file}")
    print(f"Total steps processed: {len(all_results)}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for result in all_results:
        step = result["step"]
        summary = result["summary"]
        print(
            f"Step {step}: "
            f"Math_AVG={summary['math_avg']}, "
            f"General_AVG={summary['general_avg']}, "
            f"Overall_AVG={summary['overall_avg']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate evaluation results organized by step")
    parser.add_argument("--save_path_dir", type=str, required=True,
                        help="Directory containing step_* subdirectories with evaluation results")
    parser.add_argument("--base_model_dir", type=str, default=None,
                        help="Directory containing base model evaluation results (as step 0)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path for aggregated results (optional)")

    args = parser.parse_args()
    aggregate_all_steps(args.save_path_dir, args.base_model_dir, args.output_file)
