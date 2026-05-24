"""python -m baselines.ExpeLMath <subcommand> ..."""
from __future__ import annotations

import argparse

from baselines.SkillRL.student_rollout import build_gen_traj_parser

from .build_embeddings import build_build_embeddings_parser
from .build_memory import build_build_memory_parser
from .eval_with_memory import build_eval_parser
from .evolve_memory import build_evolve_memory_parser
from .prepare_prompt_data import build_prepare_prompt_data_parser
from .retrieve_memory import build_retrieve_parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Math ExpeL pipeline (baselines/ExpeLMath)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build_gen_traj_parser(sub)
    build_build_memory_parser(sub)
    build_build_embeddings_parser(sub)
    build_retrieve_parser(sub)
    build_prepare_prompt_data_parser(sub)
    build_eval_parser(sub)
    build_evolve_memory_parser(sub)
    args = parser.parse_args(argv)
    fn = getattr(args, "_run", None)
    if fn is None:
        parser.print_help()
        return 2
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
