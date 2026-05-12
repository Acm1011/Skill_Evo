#!/usr/bin/env python3
"""
分析评测结果中的输出 token 长度。

默认会读取 ``/home/ycy/sdi/skill_saved/evaluation/evaluation`` 下指定实验的
``step_x`` 目录（或无 step 的根目录），统计：

1. 每种题型的正确/错误/总体平均输出 token 长度
2. 所有题型汇总后的正确/错误/总体平均输出 token 长度

token 计数优先使用 OpenAI 生态的 ``tiktoken`` 做近似统计；若环境未安装，
会降级到基于正则的近似切分，并在输出里记录实际使用的后端。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RESULTS_ROOT = Path("/home/ycy/sdi/skill_saved/evaluation/evaluation")
DEFAULT_OUTPUT_FILE = RESULTS_ROOT / "token_length_analysis.json"

# 在这里配置需要分析的实验与 step。
# None 表示实验目录本身就是结果目录，不再进入 step_x。
EXPERIMENTS_TO_ANALYZE: Dict[str, List[Optional[int]]] = {
    "baseline_grpo_qwen3_4b_temperature0.7": [90],
    "data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v3_temperature0.7": [45],
    "data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v3_with_skills_temperature0.7": [45],
}


def _default_encoding() -> str:
    return "cl100k_base"


class TokenCounter:
    def __init__(self, encoding_name: str) -> None:
        self.encoding_name = encoding_name
        self.backend = "regex_fallback"
        self._encoder = None
        try:
            import tiktoken  # type: ignore

            self._encoder = tiktoken.get_encoding(encoding_name)
            self.backend = f"tiktoken:{encoding_name}"
        except Exception:
            self._encoder = None

    def count(self, text: Any) -> int:
        if text is None:
            return 0
        if not isinstance(text, str):
            text = str(text)
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


@dataclass
class ResponseRecord:
    dataset: str
    is_correct: bool
    token_length: int


def _safe_mean(values: List[int]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _make_bucket(records: List[ResponseRecord]) -> Dict[str, Any]:
    correct = [r.token_length for r in records if r.is_correct]
    wrong = [r.token_length for r in records if not r.is_correct]
    total = [r.token_length for r in records]
    return {
        "correct_count": len(correct),
        "wrong_count": len(wrong),
        "total_count": len(total),
        "correct_avg_tokens": _safe_mean(correct),
        "wrong_avg_tokens": _safe_mean(wrong),
        "overall_avg_tokens": _safe_mean(total),
    }


def _iter_result_files(step_dir: Path) -> Iterable[Path]:
    for path in sorted(step_dir.glob("*_eval_results.jsonl")):
        if path.name.endswith("_Overall_results.jsonl"):
            continue
        yield path


def _extract_response_text(rsp_info_item: Dict[str, Any]) -> str:
    return str(
        rsp_info_item.get("response_str")
        or rsp_info_item.get("response")
        or rsp_info_item.get("text")
        or ""
    )


def _extract_response_correct(rsp_info_item: Dict[str, Any]) -> bool:
    if "is_check_correct" in rsp_info_item:
        return bool(rsp_info_item["is_check_correct"])
    if "is_rule_correct" in rsp_info_item:
        return bool(rsp_info_item["is_rule_correct"])
    return False


def _load_records_from_jsonl(jsonl_path: Path, token_counter: TokenCounter) -> List[ResponseRecord]:
    records: List[ResponseRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no} JSON 解析失败: {exc}") from exc

            dataset = str(row.get("data_source") or jsonl_path.name.replace("_eval_results.jsonl", ""))
            rsp_info = row.get("rsp_info") or []

            if rsp_info:
                for item in rsp_info:
                    if not isinstance(item, dict):
                        continue
                    text = _extract_response_text(item)
                    records.append(
                        ResponseRecord(
                            dataset=dataset,
                            is_correct=_extract_response_correct(item),
                            token_length=token_counter.count(text),
                        )
                    )
                continue

            responses = row.get("responses") or []
            checked_scores = row.get("checked_scores") or row.get("rule_scores") or []
            for idx, text in enumerate(responses):
                is_correct = False
                if idx < len(checked_scores):
                    is_correct = bool(checked_scores[idx])
                records.append(
                    ResponseRecord(
                        dataset=dataset,
                        is_correct=is_correct,
                        token_length=token_counter.count(text),
                    )
                )
    return records


def _analyze_step_dir(
    experiment_name: str,
    step: Optional[int],
    step_dir: Path,
    token_counter: TokenCounter,
) -> Dict[str, Any]:
    result_files = list(_iter_result_files(step_dir))
    if not result_files:
        raise FileNotFoundError(f"未找到评测结果文件: {step_dir}")

    records: List[ResponseRecord] = []
    for file_path in result_files:
        records.extend(_load_records_from_jsonl(file_path, token_counter))

    by_dataset: Dict[str, List[ResponseRecord]] = defaultdict(list)
    for record in records:
        by_dataset[record.dataset].append(record)

    per_dataset = {
        dataset: _make_bucket(dataset_records)
        for dataset, dataset_records in sorted(by_dataset.items())
    }

    parquet_files = sorted(path.name for path in step_dir.glob("*_responses.parquet"))

    return {
        "experiment_name": experiment_name,
        "step": step,
        "step_label": "root" if step is None else f"step_{step}",
        "step_dir": str(step_dir),
        "jsonl_files": [path.name for path in result_files],
        "parquet_files": parquet_files,
        "dataset_names": sorted(per_dataset.keys()),
        "per_dataset": per_dataset,
        "overall": _make_bucket(records),
    }


def analyze_experiments(
    results_root: Path,
    experiments_to_analyze: Dict[str, List[Optional[int]]],
    encoding_name: str,
) -> Dict[str, Any]:
    token_counter = TokenCounter(encoding_name=encoding_name)
    analysis_results: List[Dict[str, Any]] = []
    missing_targets: List[Dict[str, Any]] = []

    for experiment_name, steps in experiments_to_analyze.items():
        experiment_dir = results_root / experiment_name
        for step in steps:
            step_dir = experiment_dir if step is None else experiment_dir / f"step_{step}"
            if not step_dir.exists():
                missing_targets.append(
                    {
                        "experiment_name": experiment_name,
                        "step": step,
                        "expected_dir": str(step_dir),
                        "reason": "directory_not_found",
                    }
                )
                continue
            try:
                analysis_results.append(
                    _analyze_step_dir(
                        experiment_name=experiment_name,
                        step=step,
                        step_dir=step_dir,
                        token_counter=token_counter,
                    )
                )
            except FileNotFoundError:
                missing_targets.append(
                    {
                        "experiment_name": experiment_name,
                        "step": step,
                        "expected_dir": str(step_dir),
                        "reason": "no_eval_jsonl_found",
                    }
                )

    nested_results: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for item in analysis_results:
        nested_results[item["experiment_name"]][item["step_label"]] = item

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "tokenizer_backend": token_counter.backend,
        "encoding_name": encoding_name,
        "experiments_to_analyze": experiments_to_analyze,
        "analysis_results": analysis_results,
        "analysis_by_experiment": dict(nested_results),
        "missing_targets": missing_targets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析评测输出 token 长度")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="评测结果根目录",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="整合输出文件路径",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=_default_encoding(),
        help="tiktoken 编码名；未安装 tiktoken 时自动降级",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = analyze_experiments(
        results_root=args.results_root,
        experiments_to_analyze=EXPERIMENTS_TO_ANALYZE,
        encoding_name=args.encoding,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved analysis to: {args.output_file}")
    print(f"Tokenizer backend: {output['tokenizer_backend']}")
    print(f"Analyzed targets: {len(output['analysis_results'])}")
    print(f"Missing targets: {len(output['missing_targets'])}")


if __name__ == "__main__":
    main()
