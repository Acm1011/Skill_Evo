#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<'USAGE'
Usage:
  bash baselines/ReasoningBankMath/scripts/fix_prepared_prompt_data_schema.sh \
    --prepared-dir /path/to/baselines/ReasoningBankMath/outputs/prepared

What it does:
  Convert old prepared prompt data files:
    temp_data_memory_prompt.{jsonl,parquet}
    greedy_data_memory_prompt.{jsonl,parquet}
  into evaluation-compatible:
    temp_data.{jsonl,parquet}
    greedy_data.{jsonl,parquet}
USAGE
}

PREPARED_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prepared-dir)
            PREPARED_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[fix-prepared-prompt] unknown arg: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${PREPARED_DIR}" ]]; then
    echo "[fix-prepared-prompt] --prepared-dir is required" >&2
    exit 2
fi

python - <<'PY' "${PREPARED_DIR}"
import json
import os
import sys
import pandas as pd

prepared_dir = sys.argv[1]

def norm_rec(rec, dataset_name):
    extra = rec.get("extra_info") if isinstance(rec.get("extra_info"), dict) else {}
    problem = rec.get("problem") or extra.get("problem")
    gt = rec.get("ground_truth")
    if gt is None:
        rm = rec.get("reward_model")
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")
    rec["problem"] = problem
    rec["ground_truth"] = gt
    rec["data_source"] = dataset_name
    if "reward_model" not in rec or not isinstance(rec["reward_model"], dict):
        rec["reward_model"] = {"ground_truth": gt}
    return rec

for src, dst in [
    ("temp_data_memory_prompt", "temp_data"),
    ("greedy_data_memory_prompt", "greedy_data"),
]:
    jsonl_in = os.path.join(prepared_dir, f"{src}.jsonl")
    parquet_in = os.path.join(prepared_dir, f"{src}.parquet")
    jsonl_out = os.path.join(prepared_dir, f"{dst}.jsonl")
    parquet_out = os.path.join(prepared_dir, f"{dst}.parquet")

    if not os.path.exists(jsonl_in):
        raise SystemExit(f"missing input jsonl: {jsonl_in}")
    if not os.path.exists(parquet_in):
        raise SystemExit(f"missing input parquet: {parquet_in}")

    rows = []
    with open(jsonl_in, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(norm_rec(json.loads(line), dst))

    with open(jsonl_out, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pd.DataFrame(rows).to_parquet(parquet_out, index=False)
    print(f"converted: {src} -> {dst}")
PY
