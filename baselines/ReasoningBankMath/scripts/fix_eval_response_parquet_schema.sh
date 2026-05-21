#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<'USAGE'
Usage:
  bash baselines/ReasoningBankMath/scripts/fix_eval_response_parquet_schema.sh \
    --root /path/to/eval_saved_dir

What it does:
  Patch existing step_* response parquet files in-place so they match the
  evaluation pipeline schema expected by post_eval_step.py.

Targets:
  step_*/temp_data_responses.parquet
  step_*/greedy_data_responses.parquet
USAGE
}

ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[fix-eval-response] unknown arg: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${ROOT}" ]]; then
    echo "[fix-eval-response] --root is required" >&2
    exit 2
fi

python - <<'PY' "${ROOT}"
import json
import os
import sys
import pandas as pd

root = sys.argv[1]

def maybe_parse_json(x):
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x

def ensure_response_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [value]

def normalize_df(df, dataset_name):
    for col in ("prompt", "formatted_prompt", "extra_info", "reward_model", "responses", "response"):
        if col in df.columns:
            df[col] = df[col].map(maybe_parse_json)

    if "problem" not in df.columns and "extra_info" in df.columns:
        df["problem"] = df["extra_info"].map(
            lambda x: x.get("problem") if isinstance(x, dict) else None
        )

    if "ground_truth" not in df.columns:
        if "reward_model" in df.columns:
            df["ground_truth"] = df["reward_model"].map(
                lambda x: x.get("ground_truth") if isinstance(x, dict) else x
            )

    df["data_source"] = dataset_name

    if "responses" not in df.columns and "response" in df.columns:
        df["responses"] = df["response"]

    if "responses" in df.columns:
        df["responses"] = df["responses"].map(ensure_response_list)

    if "formatted_prompt" not in df.columns and "prompt" in df.columns:
        df["formatted_prompt"] = df["prompt"]

    return df

patched = 0
for dirpath, _, filenames in os.walk(root):
    for dataset_name in ("temp_data", "greedy_data"):
        fn = f"{dataset_name}_responses.parquet"
        if fn not in filenames:
            continue
        path = os.path.join(dirpath, fn)
        df = pd.read_parquet(path)
        df = normalize_df(df, dataset_name)
        tmp = path + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        print(f"patched: {path}")
        patched += 1

print(f"total patched: {patched}")
PY
