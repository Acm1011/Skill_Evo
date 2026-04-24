"""python -m baselines.SkillRL <subcommand> ..."""
from __future__ import annotations

import argparse
import sys

from .build_rl_parquet import build_build_rl_parquet_parser
from .build_sft_data import build_build_sft_parser
from .student_rollout import build_gen_traj_parser
from .teacher_distill import build_distill_parser, build_inspect_parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepMath + SkillBank pipeline (baselines/SkillRL)")
    sub = parser.add_subparsers(dest="command", required=True)
    build_gen_traj_parser(sub)
    build_distill_parser(sub)
    build_inspect_parser(sub)
    build_build_sft_parser(sub)
    build_build_rl_parquet_parser(sub)
    args = parser.parse_args(argv)
    fn = getattr(args, "_run", None)
    if fn is None:
        parser.print_help()
        return 2
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
