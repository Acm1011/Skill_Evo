import argparse
import collections
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from se_code_ttrl.Challenger_dataset import get_prompts


DEFAULT_ICL_PATH = "/home/ycy/data1/data/ttrl/ttrl_math_mix.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Test weakness_icl sampling distribution")
    parser.add_argument("--num_querys", type=int, default=1000, help="total number of samples to draw")
    parser.add_argument(
        "--icl_files",
        type=str,
        default=DEFAULT_ICL_PATH,
        help="path to ICL data jsonl/parquet/json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="directory to save plots and csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ICL data from: {args.icl_files}")
    print(f"Sampling num_querys={args.num_querys} with weakness_icl strategy ...")

    samples = get_prompts(
        num_querys=args.num_querys,
        get_prompts_func="weakness_icl",
        icl_files=args.icl_files,
    )

    if not samples:
        print("No samples returned.")
        return

    df = pd.DataFrame(samples)

    # 统计每个 data_source 的频数
    counts = df["data_source"].value_counts().sort_index()
    total = counts.sum()

    print("\nSampled distribution by data_source:")
    for src, cnt in counts.items():
        ratio = cnt / total if total > 0 else 0.0
        print(f"  {src}: {cnt} ({ratio:.4f})")

    # 保存分布到 CSV
    dist_df = counts.rename("count").reset_index().rename(columns={"index": "data_source"})
    dist_df["ratio"] = dist_df["count"] / total
    csv_path = out_dir / "icl_sampling_distribution.csv"
    dist_df.to_csv(csv_path, index=False)
    print(f"\nSaved distribution CSV to: {csv_path}")

    # 画柱状图
    plt.figure(figsize=(10, 5))
    plt.bar(dist_df["data_source"], dist_df["ratio"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Sample ratio")
    plt.title("weakness_icl sampling distribution by data_source")
    plt.tight_layout()

    png_path = out_dir / "icl_sampling_distribution.png"
    plt.savefig(png_path)
    print(f"Saved distribution plot to: {png_path}")


if __name__ == "__main__":
    main()
