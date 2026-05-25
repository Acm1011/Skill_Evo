#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from collections import OrderedDict

import pandas as pd


GENERAL_DATASETS = ["bbeh", "mmlupro", "supergpqa", "gpqa"]


def get_all_steps(save_path_dir, base_model_dir=None):
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


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_accuracy(value):
    if value is None:
        return None
    value = float(value)
    if value <= 1.0:
        value *= 100.0
    return round(value, 2)


def generate_results_table(exp_name, save_path_dir, base_model_dir=None, output_file=None):
    steps = get_all_steps(save_path_dir, base_model_dir)
    if not steps:
        print(f"Warning: No step directories found in {save_path_dir}")
        return None

    rows = []
    summary = {
        "exp_name": exp_name,
        "steps": [],
    }

    for step_num, step_dir in steps:
        aggregated_path = os.path.join(step_dir, "aggregated_eval_results.json")
        data = load_json(aggregated_path)
        if not data:
            print(f"Warning: aggregated result not found: {aggregated_path}")
            continue

        datasets = data.get("datasets", {})
        row = OrderedDict()
        row["Step"] = step_num

        values = []
        step_info = {
            "step": step_num,
            "datasets": {},
        }
        for dataset_name in GENERAL_DATASETS:
            dataset_result = datasets.get(dataset_name, {})
            accuracy = dataset_result.get("accuracy")
            if accuracy is None:
                accuracy = dataset_result.get("micro_accuracy")
            accuracy = normalize_accuracy(accuracy)
            row[dataset_name] = accuracy
            step_info["datasets"][dataset_name] = accuracy
            if accuracy is not None:
                values.append(accuracy)

        general_avg = round(sum(values) / len(values), 2) if values else None
        overall = data.get("overall", {})
        overall_accuracy = normalize_accuracy(overall.get("overall_accuracy"))
        row["General_AVG"] = general_avg
        row["Overall_Accuracy"] = overall_accuracy
        rows.append(row)

        step_info["general_avg"] = general_avg
        step_info["overall_accuracy"] = overall_accuracy
        summary["steps"].append(step_info)

    if not rows:
        print("Warning: no valid aggregated results found")
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("Step").reset_index(drop=True)

    if output_file is None:
        output_file = os.path.join(save_path_dir, f"{exp_name}_general_results_table.csv")

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Results table saved to: {output_file}")

    md_file = output_file.replace(".csv", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# General Evaluation Results by Step\n\n")
        f.write(f"Experiment: {exp_name}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Results table (Markdown) saved to: {md_file}")

    json_file = output_file.replace(".csv", ".json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary JSON saved to: {json_file}")

    print("\nGeneral Results Table:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate general evaluation results table for multi-step checkpoints")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_path_dir", type=str, required=True)
    parser.add_argument("--base_model_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    generate_results_table(args.exp_name, args.save_path_dir, args.base_model_dir, args.output_file)
