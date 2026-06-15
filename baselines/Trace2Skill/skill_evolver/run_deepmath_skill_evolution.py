#!/usr/bin/env python3
"""DeepMath runner for the original Trace2Skill evolution pipeline.

This script does not replace Trace2Skill with a simple memory summarizer. It
adapts DeepMath/SkillRL rollout trajectories into Trace2Skill's native error
record format, then calls the original ParallelSkillEvolver MAP -> REDUCE ->
TRANSLATION -> APPLY pipeline to evolve a skill folder. After evolution it can
inject the evolved skill into DeepMath temp/greedy prompt data.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL line {line_no} in {path}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_problem(row: dict) -> str:
    for key in ("problem", "question", "raw_question"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    for key in ("problem", "question"):
        val = extra.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return ""


def extract_ground_truth(row: dict) -> str:
    for key in ("ground_truth", "gt", "answer"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    reward = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
    val = reward.get("ground_truth")
    if isinstance(val, list) and val:
        return str(val[0]).strip()
    if isinstance(val, str) and val.strip():
        return val.strip()
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    for key in ("answer", "solution"):
        val = extra.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def extract_response(row: dict) -> str:
    for key in ("student_response", "response", "completion", "model_response", "solution"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def extract_answer(text: str) -> str:
    tag = _ANSWER_TAG_RE.search(text or "")
    if tag:
        return tag.group(1).strip()
    boxed = [m.group(1).strip() for m in _BOXED_RE.finditer(text or "")]
    if boxed:
        return boxed[-1]
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1].rstrip(".") if lines else ""


def is_correct(row: dict, response: str, gt: str) -> bool | None:
    flag = row.get("is_correct")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, (int, float)):
        return bool(flag)
    pred = extract_answer(response)
    if not pred or not gt:
        return None
    try:
        from mathruler.grader import grade_answer
        return bool(grade_answer(pred, gt))
    except Exception:
        return pred.strip().lower().rstrip(".") == gt.strip().lower().rstrip(".")


def topic_of(row: dict) -> str:
    for key in ("topic", "topic_key"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    for key in ("topic", "topic_key"):
        val = extra.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "general_math"


def rollout_to_error_records(rows: list[dict], include_success: bool = True) -> list[dict]:
    records: list[dict] = []
    for idx, row in enumerate(rows):
        problem = extract_problem(row)
        response = extract_response(row)
        gt = extract_ground_truth(row)
        if not problem or not response:
            continue
        ok = is_correct(row, response, gt)
        if ok is True and not include_success:
            continue
        pred = extract_answer(response)
        topic = topic_of(row)
        instance_id = str(row.get("idx") or row.get("id") or row.get("line_idx") or idx)
        if ok is True:
            items = [{
                "type": "failure_memory",
                "title": f"Reusable successful strategy for {topic}",
                "description": "A successful DeepMath trajectory that should be distilled into general skill guidance.",
                "content": (
                    f"Problem:\n{problem}\n\n"
                    f"Successful response:\n{response}\n\n"
                    f"Ground truth: {gt}\nPredicted answer: {pred}"
                ),
            }]
        else:
            items = [
                {
                    "type": "failure_cause",
                    "title": f"Incorrect DeepMath answer in {topic}",
                    "description": "The model's final answer did not match the ground truth or could not be verified.",
                    "content": (
                        f"Problem:\n{problem}\n\n"
                        f"Model response:\n{response}\n\n"
                        f"Predicted answer: {pred}\nGround truth: {gt}\n"
                        "Analyze the reasoning gap and convert it into reusable mathematical skill guidance."
                    ),
                },
                {
                    "type": "failure_memory",
                    "title": f"Lesson to remember for {topic}",
                    "description": "A general lesson derived from the failed trajectory.",
                    "content": (
                        "Future attempts should explicitly verify algebraic transformations, case splits, "
                        "domain constraints, and final answer formatting against the original problem."
                    ),
                },
            ]
        records.append({"instance_id": instance_id, "task_id": instance_id, "items": items})
    return records


def read_skill_text(skill_dir: Path) -> str:
    parts: list[str] = []
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        parts.append(skill_md.read_text(encoding="utf-8"))
    refs = skill_dir / "references"
    if refs.is_dir():
        for ref in sorted(refs.glob("*.md")):
            parts.append(f"\n\n## Reference: {ref.name}\n" + ref.read_text(encoding="utf-8"))
    return "\n".join(parts).strip()


def build_prompt(skill_text: str, problem: str) -> str:
    return (
        "You are solving a math problem. Use the following evolved Trace2Skill memory when relevant.\n\n"
        f"## Skill Memory\n{skill_text}\n\n"
        f"## Problem\n{problem}\n\n"
        "Think step by step, and put the final answer inside <answer>...</answer>."
    )


def prepare_prompt_data(input_jsonl: Path, skill_dir: Path, output_jsonl: Path, output_parquet: Path | None, data_source: str, start: int, end: int | None) -> int:
    rows = read_jsonl(input_jsonl)
    skill_text = read_skill_text(skill_dir)
    output_rows: list[dict] = []
    for line_no, row in enumerate(rows):
        if line_no < start:
            continue
        if end is not None and line_no >= end:
            break
        problem = extract_problem(row)
        if not problem:
            continue
        gt = extract_ground_truth(row)
        extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        ex = dict(extra)
        ex.update({
            "problem": problem,
            "idx": row.get("idx", ex.get("idx", line_no)),
            "trace2skill_memory_dir": str(skill_dir),
            "trace2skill_memory_chars": len(skill_text),
        })
        rec = {
            "problem": problem,
            "ground_truth": gt,
            "prompt": [{"role": "user", "content": build_prompt(skill_text, problem)}],
            "reward_model": {"ground_truth": gt},
            "data_source": data_source,
            "extra_info": ex,
        }
        if isinstance(row.get("prompt"), list):
            rec["original_prompt"] = copy.deepcopy(row["prompt"])
        output_rows.append(rec)
    write_jsonl(output_jsonl, output_rows)
    if output_parquet is not None:
        try:
            import pandas as pd
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(output_rows).to_parquet(output_parquet, index=False)
        except ImportError as exc:
            raise SystemExit("Writing parquet requires pandas and pyarrow") from exc
    return len(output_rows)


def build_client(args: argparse.Namespace):
    from src.react_agent.models import ApiChatClient, OpenAIClient

    kwargs: dict[str, Any] = {"model": args.model}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.cache_path:
        kwargs["cache_path"] = str(args.cache_path)
    if args.llm_client == "api_chat":
        kwargs.pop("base_url", None)
        kwargs.pop("api_key", None)
        kwargs["config_path"] = args.api_chat_config
        return ApiChatClient(**kwargs)
    return OpenAIClient(**kwargs)


def run_build_skill(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.trajectories)
    records = rollout_to_error_records(rows, include_success=args.include_success)
    if args.max_records and args.max_records > 0:
        records = records[:args.max_records]
    if not records:
        raise SystemExit("No Trace2Skill error records were produced from trajectories.")
    if args.records_out:
        args.records_out.parent.mkdir(parents=True, exist_ok=True)
        args.records_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        from skill_evolver.parallel_evolving_agent import ParallelSkillEvolver
    except ImportError as exc:
        raise SystemExit(
            "Trace2Skill build-skill requires the original Trace2Skill dependencies. "
            "Please install them first, e.g. `pip install -r requirements.txt`, "
            "then rerun this command."
        ) from exc

    client = build_client(args)
    evolver = ParallelSkillEvolver(
        client=client,
        skill_dir=args.skill_dir,
        batch_size=args.batch_size,
        merge_batch_size=args.merge_batch_size,
        max_workers=args.max_workers,
        max_merge_levels=args.max_merge_levels,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        dry_run=args.dry_run,
        prompt_variant=args.prompt,
        output_dir=args.intermediates_dir,
        parse_failure_dir=args.parse_failure_dir,
        max_skill_lines=args.max_skill_lines,
        skip_translation=args.skip_translation,
        patch_pipeline=args.patch_pipeline,
        semantic_item_marker_format=args.semantic_item_marker_format,
    )
    result = evolver.run(records, input_mode="records")
    print(json.dumps({
        "records": len(records),
        "patches": len(result.get("patches", [])),
        "edits": len(result.get("edits", [])),
        "skill_dir": str(args.skill_dir),
    }, ensure_ascii=False, indent=2))
    return 0


def run_prepare(args: argparse.Namespace) -> int:
    n = prepare_prompt_data(
        input_jsonl=args.input_jsonl,
        skill_dir=args.skill_dir,
        output_jsonl=args.output_jsonl,
        output_parquet=args.output_parquet,
        data_source=args.data_source,
        start=args.start,
        end=args.end,
    )
    print(f"[trace2skill-deepmath] wrote {n} rows to {args.output_jsonl}", flush=True)
    return 0


def run_all(args: argparse.Namespace) -> int:
    run_build_skill(args)
    if args.temp_input_jsonl and args.temp_output_jsonl:
        prepare_prompt_data(args.temp_input_jsonl, args.skill_dir, args.temp_output_jsonl, args.temp_output_parquet, args.data_source, args.start, args.end)
    if args.greedy_input_jsonl and args.greedy_output_jsonl:
        prepare_prompt_data(args.greedy_input_jsonl, args.skill_dir, args.greedy_output_jsonl, args.greedy_output_parquet, args.data_source, args.start, args.end)
    return 0


def add_common_evolve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trajectories", type=Path, default=Path("../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl"))
    p.add_argument("--skill-dir", type=Path, default=Path("released_skills/deepmath"))
    p.add_argument("--model", required=True)
    p.add_argument("--llm-client", choices=["openai", "api_chat"], default="openai")
    p.add_argument("--api-chat-config", default="config/llm_api.json")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--cache-path", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--merge-batch-size", type=int, default=5)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--max-merge-levels", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-skill-lines", type=int, default=500)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--include-success", action="store_true")
    p.add_argument("--records-out", type=Path, default=Path("outputs/deepmath_error_records.json"))
    p.add_argument("--intermediates-dir", type=Path, default=Path("outputs/deepmath_parallel_output"))
    p.add_argument("--parse-failure-dir", type=Path, default=Path("parse_failures_deepmath"))
    p.add_argument("--prompt", default="generic", choices=["skill", "generic", "patterns", "patterns_generic"])
    p.add_argument("--patch-pipeline", choices=["json", "markdown"], default="json")
    p.add_argument("--semantic-item-marker-format", choices=["bracket", "heading"], default="bracket")
    p.add_argument("--skip-translation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace2Skill DeepMath adapter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-skill")
    add_common_evolve_args(p_build)
    p_build.set_defaults(func=run_build_skill)

    p_prepare = sub.add_parser("prepare-prompt-data")
    p_prepare.add_argument("--input-jsonl", type=Path, required=True)
    p_prepare.add_argument("--skill-dir", type=Path, default=Path("released_skills/deepmath"))
    p_prepare.add_argument("--output-jsonl", type=Path, required=True)
    p_prepare.add_argument("--output-parquet", type=Path, default=None)
    p_prepare.add_argument("--data-source", default="Trace2SkillMath")
    p_prepare.add_argument("--start", type=int, default=0)
    p_prepare.add_argument("--end", type=int, default=None)
    p_prepare.set_defaults(func=run_prepare)

    p_all = sub.add_parser("run-all")
    add_common_evolve_args(p_all)
    p_all.add_argument("--temp-input-jsonl", type=Path, default=None)
    p_all.add_argument("--temp-output-jsonl", type=Path, default=None)
    p_all.add_argument("--temp-output-parquet", type=Path, default=None)
    p_all.add_argument("--greedy-input-jsonl", type=Path, default=None)
    p_all.add_argument("--greedy-output-jsonl", type=Path, default=None)
    p_all.add_argument("--greedy-output-parquet", type=Path, default=None)
    p_all.add_argument("--data-source", default="Trace2SkillMath")
    p_all.add_argument("--start", type=int, default=0)
    p_all.add_argument("--end", type=int, default=None)
    p_all.set_defaults(func=run_all)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
