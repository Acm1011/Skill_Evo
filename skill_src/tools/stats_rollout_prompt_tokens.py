#!/usr/bin/env python3
"""
统计 rollout jsonl 中每条样本 ``prompt[0].content`` 的 token 长度（快速近似）。

使用 OpenAI 的 ``tiktoken`` 编码计数，**不加载 HF 模型**，整文件扫描也很快。
与 Qwen 等训练用 tokenizer 的绝对值可能略有偏差，适合看均值与超长占比。

用法::

    conda activate se
    pip install tiktoken   # 若尚未安装
    python -m skill_src.tools.stats_rollout_prompt_tokens \\
        --jsonl /data4/ycy/skill_saved/deepmath_n12800_bs128_with_random.jsonl \\
        --encoding cl100k_base \\
        --threshold 4096

常用 ``--encoding``: ``cl100k_base``（默认）, ``o200k_base``, ``p50k_base``。
也可用环境变量 ``TIKTOKEN_ENCODING``。

jq 抽查::

    jq -r '.prompt[0].content | length' file.jsonl | head
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple


def _default_encoding() -> str:
    v = (os.environ.get("TIKTOKEN_ENCODING") or "").strip()
    return v if v else "cl100k_base"


def _extract_content(row: Dict[str, Any]) -> Optional[str]:
    p = row.get("prompt")
    if not isinstance(p, list) or len(p) == 0:
        return None
    first = p[0]
    if not isinstance(first, dict):
        return None
    c = first.get("content")
    if c is None:
        return None
    return c if isinstance(c, str) else str(c)


def run_stats(
    jsonl_path: str,
    encoding_name: str,
    threshold: int,
) -> Tuple[int, float, int, int]:
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)

    n = 0
    n_bad = 0
    n_over = 0
    sum_tok = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            content = _extract_content(row)
            if content is None:
                n_bad += 1
                continue
            nt = len(enc.encode(content))
            sum_tok += nt
            n += 1
            if nt > threshold:
                n_over += 1

    mean = (sum_tok / n) if n else 0.0
    return n, mean, n_over, n_bad


def main() -> None:
    p = argparse.ArgumentParser(
        description="用 tiktoken 快速统计 jsonl 中 prompt[0].content 的 token 均值与超阈值占比"
    )
    p.add_argument(
        "--jsonl",
        type=str,
        default="/data4/ycy/skill_saved/deepmath_n12800_bs128_with_random.jsonl",
        help="rollout jsonl 路径",
    )
    p.add_argument(
        "--encoding",
        type=str,
        default=_default_encoding(),
        help="tiktoken 编码名（默认 cl100k_base；可用环境变量 TIKTOKEN_ENCODING）",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=8192,
        help="超过该 token 数计为超长样本（默认 4096）",
    )
    args = p.parse_args()

    path = os.path.abspath(os.path.expanduser(args.jsonl))
    if not os.path.isfile(path):
        print(f"错误: 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    enc_name = (args.encoding or "").strip()
    if not enc_name:
        print("错误: 请指定 --encoding 或设置 TIKTOKEN_ENCODING", file=sys.stderr)
        sys.exit(2)

    try:
        n, mean, n_over, n_bad = run_stats(path, enc_name, args.threshold)
    except ValueError as e:
        print(f"错误: 无效的 tiktoken encoding「{enc_name}」: {e}", file=sys.stderr)
        sys.exit(3)
    except ImportError:
        print("错误: 未安装 tiktoken。请执行: pip install tiktoken", file=sys.stderr)
        sys.exit(4)

    over_ratio = (100.0 * n_over / n) if n else 0.0

    print(f"jsonl: {path}")
    print(f"tiktoken_encoding: {enc_name} (近似，非训练 tokenizer)")
    print(f"threshold_tokens: {args.threshold}")
    print(f"valid_rows: {n}")
    print(f"skipped_malformed_or_no_prompt: {n_bad}")
    print(f"mean_content_tokens: {mean:.2f}")
    print(f"over_threshold_count: {n_over}")
    print(f"over_threshold_ratio_percent: {over_ratio:.4f}")


if __name__ == "__main__":
    main()
