#!/usr/bin/env python3
"""
一次性：对已存在的 DeepMath rollout JSONL 调用 build_merged_rollout_records，
用 embedding cache（可选）或随机对照，补全 random_q_* / reward_model.raw_random_q_acc 等，与 solver_offline_driver 一致。

示例::

    conda activate se
    python -m skill_src.tools.backfill_deepmath_random_q \\
        --input /home/ycy/sdi/Skill_Evo/skill_src/Zero/rollout_results/deepmath_n12800_bs128.jsonl \\
        --embedding-cache-path /home/ycy/sdi/skill_saved/embedding_cache \\
        --num-random-questions 4 \\
        --random-q-training-step 5 --random-q-interval-batch-size 128

    shuf -n 1 out.jsonl | jq '.extra_info.random_q_info | keys, (.questions|length), (.acc|length)'
    shuf -n 1 out.jsonl | jq '.reward_model.raw_random_q_acc | length'
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from skill_src.rollout_deepmath import (
    load_merged_rows_from_jsonl,
    save_rollout_results,
)
from skill_src.solver_offline_driver import (
    _env_bool_default,
    _env_int_min1,
    _load_embedding_cache_from_dir,
    build_merged_rollout_records,
)


def main() -> None:
    default_in = "/home/ycy/sdi/Skill_Evo/skill_src/Zero/rollout_results/deepmath_n12800_bs128.jsonl"
    default_out = "/home/ycy/sdi/Skill_Evo/skill_src/Zero/rollout_results/deepmath_n12800_bs128_with_random_4_rollout4_low_acc.jsonl"
    p = argparse.ArgumentParser(description="Backfill random_q_* on DeepMath rollout jsonl")
    p.add_argument("--input", type=str, default=default_in, help="输入 jsonl")
    p.add_argument(
        "--output",
        type=str,
        default=default_out,
        help="输出 jsonl（默认 <input_stem>_with_random.jsonl；与 --in-place 互斥）",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="覆盖输入文件；会先复制一份 *.bak_backfill",
    )
    p.add_argument(
        "--embedding-cache-path",
        type=str,
        default=None,
        help="含 *.meta.json + *.npz 的目录；与 driver --embedding-cache-path 相同",
    )
    p.add_argument(
        "--num-random-questions",
        type=int,
        default=int(os.environ.get("SE_NUM_RANDOM_QUESTIONS", "10")),
        help="对照题数量 K（默认 10 或环境变量 SE_NUM_RANDOM_QUESTIONS）",
    )
    p.add_argument("--skill-type", type=str, default="skill_generation_v1")
    p.add_argument(
        "--expected-n",
        type=int,
        default=None,
        help="若指定则校验 idx 恰为 0..N-1 且共 N 条",
    )
    _ts_env = (os.environ.get("SE_RANDOM_Q_TRAINING_STEP") or "").strip()
    _rq_ts_def = int(_ts_env) if _ts_env else None
    p.add_argument(
        "--random-q-training-step",
        type=int,
        default=_rq_ts_def,
        help="training 区间 = 本值 × --random-q-interval-batch-size；不设或 ≤0 则对照候选为全表",
    )
    _ibs_env = (os.environ.get("SE_RANDOM_Q_INTERVAL_BATCH_SIZE") or "").strip()
    _ibs_def = int(_ibs_env) if _ibs_env else None
    p.add_argument(
        "--random-q-interval-batch-size",
        type=int,
        default=_ibs_def,
        help="与 rollout 的 batch_size 一致（如 128）；与 --random-q-training-step 同时 >0 时启用区间截断",
    )
    p.add_argument(
        "--random-q-prefer-low-acc",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_default("SE_RANDOM_Q_PREFER_LOW_ACC", True),
        help="约一半高相似/均匀、一半低 acc；--no-random-q-prefer-low-acc 关闭",
    )
    p.add_argument(
        "--random-q-sim-pool-factor",
        type=int,
        default=_env_int_min1("SE_RANDOM_Q_SIM_POOL_FACTOR", 15),
        help="embedding：低 acc 半侧从 top-(K×因子) 短名单挑选",
    )
    args = p.parse_args()

    inp = Path(args.input).expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"输入不存在: {inp}")

    if args.in_place and args.output:
        raise SystemExit("不可同时使用 --in-place 与 --output")

    expected_n: Optional[int] = args.expected_n
    if expected_n is None:
        merged = load_merged_rows_from_jsonl(str(inp), expected_n=None)
        expected_n = len(merged)
    else:
        merged = load_merged_rows_from_jsonl(str(inp), expected_n=expected_n)

    emb = (args.embedding_cache_path or "").strip() or None
    if emb and args.num_random_questions > 0:
        doc, query = _load_embedding_cache_from_dir(emb)
        max_idx = expected_n - 1
        if doc.shape[0] <= max_idx or query.shape[0] <= max_idx:
            raise ValueError(
                f"embedding 行数须严格大于 max_idx={max_idx}，"
                f"当前 doc={doc.shape[0]}, query={query.shape[0]}"
            )

    tr_step: Optional[int] = None
    tr_bs: Optional[int] = None
    rq_ts = args.random_q_training_step
    rq_bs = args.random_q_interval_batch_size
    if rq_ts is not None and int(rq_ts) > 0:
        if rq_bs is None or int(rq_bs) <= 0:
            raise ValueError(
                "已设置 --random-q-training-step>0 时，必须同时提供正整数 "
                "--random-q-interval-batch-size（或与之一致的 batch_size），"
                "或设置环境变量 SE_RANDOM_Q_INTERVAL_BATCH_SIZE"
            )
        tr_step, tr_bs = int(rq_ts), int(rq_bs)

    build_merged_rollout_records(
        merged,
        args.num_random_questions,
        args.skill_type,
        emb,
        training_step=tr_step,
        batch_size_for_interval=tr_bs,
        prefer_low_acc=bool(args.random_q_prefer_low_acc),
        sim_pool_factor=max(1, int(args.random_q_sim_pool_factor)),
    )

    if args.in_place:
        bak = inp.with_suffix(inp.suffix + ".bak_backfill")
        shutil.copy2(inp, bak)
        print(f"[backfill] 已备份: {bak}")
        tmp = str(inp) + ".tmp_backfill"
        save_rollout_results(merged, tmp, append=False)
        os.replace(tmp, str(inp))
        print(f"[backfill] 已原地覆盖: {inp}")
    else:
        out_path = (
            str(Path(args.output).expanduser().resolve())
            if args.output
            else str(inp.with_name(f"{inp.stem}_with_random{inp.suffix}"))
        )
        save_rollout_results(merged, out_path, append=False)
        print(f"[backfill] 已写入: {out_path}")


if __name__ == "__main__":
    main()
