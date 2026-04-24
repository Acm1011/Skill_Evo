"""Build Alpaca JSON for LLaMA-Factory SFT (skill-augmented DeepMath)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .layered_skill_bank import LayeredSkillBank
from .student_rollout import extract_ground_truth, extract_meta, extract_problem
from .text_utils import topic_slug


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_skill_use_template() -> str:
    return (_prompt_dir() / "skill_use_math.txt").read_text(encoding="utf-8")


def load_trajectories_index(path: Optional[str]) -> Dict[Any, Dict[str, Any]]:
    if not path:
        return {}
    out: Dict[Any, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = row.get("idx")
            if idx is not None:
                out[idx] = row
            li = row.get("line_idx")
            if li is not None:
                out[li] = row
    return out


def skills_block(bank: LayeredSkillBank, problem: str, topic: Optional[str], top_k: int) -> str:
    mem = bank.retrieve(task_description=problem or "", topic=topic, top_k=top_k)
    return bank.format_for_prompt(mem)


def run_build_sft(args: argparse.Namespace) -> int:
    bank = LayeredSkillBank.from_path(args.skills_json)
    template = load_skill_use_template()
    traj_idx = load_trajectories_index(args.trajectories or None)

    rows_out: List[Dict[str, str]] = []
    skipped = 0

    with open(args.deepmath_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < args.start:
                continue
            if line_no >= args.end:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem = extract_problem(row)
            if not problem:
                skipped += 1
                continue
            meta = extract_meta(row, line_no)
            topic = meta.get("topic")
            gt = extract_ground_truth(row)
            skill_text = skills_block(bank, problem, topic, args.top_k)
            try:
                instruction = template.format(skill=skill_text, question=problem)
            except Exception as e:
                print(f"[build-sft] skip line {line_no}: template: {e}", file=sys.stderr)
                skipped += 1
                continue

            trow = traj_idx.get(meta.get("idx"), traj_idx.get(line_no))
            output: Optional[str] = None
            if trow:
                if args.only_correct and trow.get("is_correct") is not True:
                    skipped += 1
                    continue
                output = (trow.get("student_response") or "").strip() or None

            if not output and args.fallback_boxed_gt and gt:
                output = (
                    "Brief reasoning omitted for SFT warm-up.\n\n"
                    f"Final answer: \\boxed{{{gt}}}"
                )

            if not output:
                skipped += 1
                continue

            rows_out.append(
                {
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "topic_key": topic_slug(topic),
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)

    print(
        f"[build-sft] wrote {len(rows_out)} samples -> {out_path} (skipped {skipped})",
        file=sys.stderr,
    )
    return 0


def build_build_sft_parser(sub: Any) -> None:
    p = sub.add_parser("build-sft", help="DeepMath + skills -> Alpaca JSON for LLaMA-Factory")
    p.add_argument("--deepmath-jsonl", required=True)
    p.add_argument("--skills-json", required=True)
    p.add_argument("--trajectories", default="", help="Optional trajectories.jsonl for response text")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--only-correct", action="store_true")
    p.add_argument("--fallback-boxed-gt", action="store_true")
    p.add_argument("--output", required=True)
    p.set_defaults(_run=run_build_sft)
