#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<'USAGE'
Usage:
  bash baselines/ReasoningBankMath/scripts/fix_eval_data_source_and_problem.sh \
    --root /path/to/eval_dir_or_step_dir \
    [--temp-input /path/to/data/temp_data.jsonl] \
    [--greedy-input /path/to/data/greedy_data.jsonl]

What it does:
  1. Build a problem -> data_source mapping from the original input jsonl files.
  2. Patch existing eval artifacts in-place:
     - temp_data_responses.parquet
     - greedy_data_responses.parquet
     - temp_data_Overall_results.jsonl
     - greedy_data_Overall_results.jsonl
     - aggregated_eval_results.json
  3. Recompute per-data_source Overall results from the patched parquet files.

Notes:
  - This is intended for already-generated results whose data_source was
    incorrectly overwritten to temp_data / greedy_data.
  - It does not rerun model inference.
USAGE
}

ROOT=""
TEMP_INPUT="${REPO_ROOT}/data/temp_data.jsonl"
GREEDY_INPUT="${REPO_ROOT}/data/greedy_data.jsonl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --temp-input)
            TEMP_INPUT="$2"
            shift 2
            ;;
        --greedy-input)
            GREEDY_INPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[fix-eval-data-source] unknown arg: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${ROOT}" ]]; then
    echo "[fix-eval-data-source] --root is required" >&2
    exit 2
fi

python - <<'PY' "${ROOT}" "${TEMP_INPUT}" "${GREEDY_INPUT}"
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

root = sys.argv[1]
temp_input = sys.argv[2]
greedy_input = sys.argv[3]


def maybe_parse_json(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            out = value.tolist()
            if isinstance(out, list):
                return out
        except Exception:
            pass
    return [value]


def as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = maybe_parse_json(value)
    return parsed if isinstance(parsed, dict) else {}


def normalize_problem(problem: Any) -> str:
    return str(problem or "").strip()


def load_problem_map(path: str, dataset_name: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise SystemExit(f"missing input jsonl: {path}")
    mapping: Dict[str, str] = {}
    conflicts = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
            problem = normalize_problem(row.get("problem") or extra.get("problem"))
            data_source = str(row.get("data_source") or extra.get("data_source") or dataset_name).strip()
            if not problem:
                continue
            prev = mapping.get(problem)
            if prev is not None and prev != data_source:
                conflicts += 1
                continue
            mapping[problem] = data_source
    if not mapping:
        raise SystemExit(f"no usable problems found in: {path}")
    if conflicts:
        print(f"[fix-eval-data-source] warning: {conflicts} conflicting problem->data_source rows ignored from {path}")
    print(f"[fix-eval-data-source] loaded {len(mapping)} problem mappings from {path}")
    return mapping


PROBLEM_MAPS = {
    "temp_data": load_problem_map(temp_input, "temp_data"),
    "greedy_data": load_problem_map(greedy_input, "greedy_data"),
}


def infer_data_source(problem: str, dataset_name: str, extra: Optional[Dict[str, Any]] = None, current: Any = None) -> str:
    problem = normalize_problem(problem)
    if problem:
        mapped = PROBLEM_MAPS[dataset_name].get(problem)
        if mapped:
            return mapped
    if isinstance(extra, dict):
        val = extra.get("data_source")
        if val and val not in ("temp_data", "greedy_data"):
            return str(val)
    if current and current not in ("temp_data", "greedy_data"):
        return str(current)
    return dataset_name


def get_ground_truth(row: Dict[str, Any]) -> Any:
    gt = row.get("ground_truth")
    if gt is not None:
        return gt
    rm = as_dict(row.get("reward_model"))
    return rm.get("ground_truth")


def patch_responses_parquet(path: str, dataset_name: str) -> Optional[pd.DataFrame]:
    df = pd.read_parquet(path)
    for col in ("prompt", "formatted_prompt", "extra_info", "reward_model", "responses", "response"):
        if col in df.columns:
            df[col] = df[col].map(maybe_parse_json)

    if "responses" not in df.columns and "response" in df.columns:
        df["responses"] = df["response"]
    if "responses" in df.columns:
        df["responses"] = df["responses"].map(ensure_list)

    if "formatted_prompt" not in df.columns and "prompt" in df.columns:
        df["formatted_prompt"] = df["prompt"]

    problems: List[str] = []
    data_sources: List[str] = []
    ground_truths: List[Any] = []
    extra_infos: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        extra = as_dict(row.get("extra_info"))
        problem = normalize_problem(row.get("problem") or extra.get("problem"))
        data_source = infer_data_source(problem, dataset_name, extra=extra, current=row.get("data_source"))
        gt = get_ground_truth(row.to_dict())
        if not isinstance(extra, dict):
            extra = {}
        extra["problem"] = problem
        extra["data_source"] = data_source
        problems.append(problem)
        data_sources.append(data_source)
        ground_truths.append(gt)
        extra_infos.append(extra)

    df["problem"] = problems
    df["data_source"] = data_sources
    df["extra_info"] = extra_infos
    if "ground_truth" not in df.columns:
        df["ground_truth"] = ground_truths
    if "reward_model" not in df.columns:
        df["reward_model"] = [{"ground_truth": gt} for gt in ground_truths]
    else:
        def fix_rm(value: Any, gt: Any) -> Dict[str, Any]:
            rm = as_dict(value)
            if rm.get("ground_truth") is None:
                rm["ground_truth"] = gt
            return rm
        df["reward_model"] = [fix_rm(value, gt) for value, gt in zip(df["reward_model"], ground_truths)]

    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    print(f"[fix-eval-data-source] patched parquet: {path}")
    return df


def to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes, dict)):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return to_python_scalar(obj)


def build_overall_rows(df: pd.DataFrame, dataset_name: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_samples_meta = int(meta.get("n_samples") or 0)
    model_name = meta.get("model_name") or meta.get("model") or ""

    for data_source, group in df.groupby("data_source"):
        records = group.to_dict(orient="records")
        if not records:
            continue

        sample_counts = []
        for rec in records:
            raw_scores = ensure_list(rec.get("raw_scores"))
            rule_scores = ensure_list(rec.get("rule_scores"))
            checked_scores = ensure_list(rec.get("checked_scores"))
            sample_counts.append(max(len(raw_scores), len(rule_scores), len(checked_scores), 1))
        n_samples = max(sample_counts) if sample_counts else max(n_samples_meta, 1)

        def mean_at(key: str, idx: int) -> float:
            vals = []
            for rec in records:
                seq = ensure_list(rec.get(key))
                if idx < len(seq):
                    vals.append(float(seq[idx]))
            return (sum(vals) / len(vals)) if vals else 0.0

        rule_first = mean_at("rule_scores", 0)
        checked_first = mean_at("checked_scores", 0)
        rule_mean = sum(mean_at("rule_scores", i) for i in range(n_samples)) / n_samples
        checked_mean = sum(mean_at("checked_scores", i) for i in range(n_samples)) / n_samples

        row: Dict[str, Any] = {
            "data_source": data_source,
            "model": model_name,
            "rule@first": f"{rule_first * 100:.2f}",
            f"rule_mean@{n_samples}": f"{rule_mean * 100:.2f}",
            "checked@first": f"{checked_first * 100:.2f}",
            f"checked_mean@{n_samples}": f"{checked_mean * 100:.2f}",
        }

        eval_params = {
            "max_gen_len": meta.get("max_gen_len"),
            "n_samples": n_samples,
            "temperature": meta.get("temperature"),
            "top_p": meta.get("top_p"),
            "top_k": meta.get("top_k"),
        }
        if any(v is not None for v in eval_params.values()):
            row["eval_params"] = eval_params
        rows.append(row)

    rows.sort(key=lambda x: x["data_source"])
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_step_dir(step_dir: str) -> None:
    datasets: Dict[str, Dict[str, Any]] = {}

    for eval_name in ("greedy_data", "temp_data"):
        overall_path = os.path.join(step_dir, f"{eval_name}_Overall_results.jsonl")
        if not os.path.exists(overall_path):
            continue
        try:
            records = pd.read_json(overall_path, lines=True).to_dict(orient="records")
        except ValueError:
            records = []
        for item in records:
            data_source = item.get("data_source", eval_name)
            n_samples = item.get("eval_params", {}).get("n_samples")
            if not n_samples:
                for key in item.keys():
                    if key.startswith("checked_mean@"):
                        try:
                            n_samples = int(key.split("@", 1)[1])
                        except Exception:
                            n_samples = None
                        break
            datasets[data_source] = {
                "eval_name": eval_name,
                "data_source": data_source,
                "n_samples": n_samples,
                "temperature": item.get("eval_params", {}).get("temperature"),
                "rule@first": float(item.get("rule@first", 0) or 0),
                "checked@first": float(item.get("checked@first", 0) or 0),
                "rule_mean": float(item.get(f"rule_mean@{n_samples}", 0) or 0),
                "checked_mean": float(item.get(f"checked_mean@{n_samples}", 0) or 0),
                "checked_sample_mean": float(item.get(f"checked_sample_mean@{n_samples}", 0) or 0),
            }

    math_vals = [item["checked_mean"] for item in datasets.values()]
    math_avg = round(sum(math_vals) / len(math_vals), 2) if math_vals else None
    payload = {
        "step": None,
        "step_dir": step_dir,
        "math_datasets": datasets,
        "additional_datasets": {},
        "summary": {
            "math_avg": math_avg,
            "general_avg": None,
            "overall_avg": math_avg,
        },
    }
    out_path = os.path.join(step_dir, "aggregated_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[fix-eval-data-source] rebuilt aggregate: {out_path}")


def maybe_step_dirs(root_dir: str) -> List[str]:
    if os.path.basename(root_dir).startswith("step_"):
        return [root_dir]
    dirs = []
    for dirpath, _, filenames in os.walk(root_dir):
        has_eval_file = any(
            name in filenames
            for name in (
                "temp_data_responses.parquet",
                "greedy_data_responses.parquet",
                "temp_data_Overall_results.jsonl",
                "greedy_data_Overall_results.jsonl",
            )
        )
        if has_eval_file:
            dirs.append(dirpath)
    return sorted(set(dirs))


step_dirs = maybe_step_dirs(root)
if not step_dirs:
    raise SystemExit(f"no eval result directories found under: {root}")

for step_dir in step_dirs:
    print(f"[fix-eval-data-source] processing: {step_dir}")
    for dataset_name in ("temp_data", "greedy_data"):
        parquet_path = os.path.join(step_dir, f"{dataset_name}_responses.parquet")
        meta_path = os.path.join(step_dir, f"{dataset_name}_meta.json")
        overall_path = os.path.join(step_dir, f"{dataset_name}_Overall_results.jsonl")

        if not os.path.exists(parquet_path):
            continue

        df = patch_responses_parquet(parquet_path, dataset_name)
        meta = load_json(meta_path)
        rows = build_overall_rows(df, dataset_name, meta)
        write_jsonl(overall_path, rows)
        print(f"[fix-eval-data-source] rebuilt overall: {overall_path}")

    aggregate_step_dir(step_dir)

print(f"[fix-eval-data-source] done, processed {len(step_dirs)} directories")
PY
