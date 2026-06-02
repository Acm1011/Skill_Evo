"""python -m baselines.ARISE <subcommand> ..."""
from __future__ import annotations

import argparse

from .prepare_prompt_data import build_prepare_prompt_data_parser
from .prepare_rl_data import build_prepare_rl_data_parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepMath + ARISE library pipeline (baselines/ARISE)")
    sub = parser.add_subparsers(dest="command", required=True)
    build_prepare_rl_data_parser(sub)
    build_prepare_prompt_data_parser(sub)
    args = parser.parse_args(argv)
    fn = getattr(args, "_run", None)
    if fn is None:
        parser.print_help()
        return 2
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
