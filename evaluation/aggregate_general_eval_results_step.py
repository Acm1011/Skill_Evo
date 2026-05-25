#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os


DATASETS = ["bbeh", "mmlupro", "supergpqa", "gpqa"]


def load_result(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_accuracy(result):
    accuracy = result.get("accuracy")
    if accuracy is None:
        accuracy = result.get("micro_accuracy")
    if accuracy is None:
        return None
    accuracy = float(accuracy)
    if accuracy <= 1.0:
        accuracy *= 100.0
    return round(accuracy, 2)


def aggregate_step_dir(step_dir, output_file=None):
    if output_file is None:
        output_file = os.path.join(step_dir, "aggregated_eval_results.json")

    datasets = {}
    total_success = 0
    total_fail = 0
    total_samples = 0
    model_name = None
    model_path = None

    for dataset_name in DATASETS:
        result_path = os.path.join(step_dir, f"{dataset_name}_final_results.json")
        result = load_result(result_path)
        if not result:
            continue

        if model_name is None:
            model_name = result.get("model")
        if model_path is None:
            model_path = result.get("model_path")

        datasets[dataset_name] = {
            "dataset": result.get("dataset", dataset_name),
            "accuracy": normalize_accuracy(result),
            "macro_accuracy": result.get("macro_accuracy"),
            "success": result.get("success", 0),
            "fail": result.get("fail", 0),
            "total": result.get("total", 0),
            "per_category_accuracy": result.get("per_category_accuracy", {}),
        }
        total_success += result.get("success", 0)
        total_fail += result.get("fail", 0)
        total_samples += result.get("total", 0)

    if not datasets:
        print(f"Warning: no general eval results found in {step_dir}")
        return 1

    accuracies = [item["accuracy"] for item in datasets.values() if item.get("accuracy") is not None]

    aggregated_result = {
        "model_name": model_name,
        "model_path": model_path,
        "datasets": datasets,
        "overall": {
            "total_success": total_success,
            "total_fail": total_fail,
            "total_samples": total_samples,
            "overall_accuracy": round(total_success / total_samples * 100, 2) if total_samples > 0 else 0.0,
        },
        "summary": {
            "general_avg": round(sum(accuracies) / len(accuracies), 2) if accuracies else None,
        }
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(aggregated_result, f, ensure_ascii=False, indent=2)

    print(f"Aggregated general eval results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate general evaluation results for a single step directory")
    parser.add_argument("--step_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    raise SystemExit(aggregate_step_dir(args.step_dir, args.output_file))
