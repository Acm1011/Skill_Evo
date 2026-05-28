from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from baselines.ReasoningBankMath.io_utils import read_jsonl, write_jsonl
from baselines.SkillRL.teacher_distill import chat_complete
from baselines.SkillRL.text_utils import parse_json_object
from baselines.preliminary.eval_skill_drift_across_checkpoints import (
    _parse_choice_filter,
    _question_lookup_key,
    _skill_row_keys,
)
from baselines.preliminary.eval_source_linked_skills import group_questions

DEFAULT_TRAJECTORIES = "Skill_Evo/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl"
METHODS = ("skillrl", "reasoningbank", "expelmath")
TEACHER_BACKENDS = ("api_teacher", "server_teacher")

SYSTEM_PROMPT = (
    "You are an expert evaluator for math problem-solving skills. "
    "Judge how helpful a provided skill would be for solving the given math question. "
    "Return only a valid JSON object."
)

USER_PROMPT_TEMPLATE = """Evaluate whether the following skill would help solve the math question.

Scoring rubric:
5 = Highly helpful. The skill is directly applicable and would materially improve the solution.
4 = Helpful. The skill is relevant and likely useful, though not sufficient by itself.
3 = Somewhat helpful. The skill has partial relevance but limited direct utility.
2 = Slightly helpful or weakly relevant. The connection is tenuous or generic.
1 = Not helpful or misleading. The skill is irrelevant, too vague, or likely distracts from solving the question.

Question:
{question}

Topic:
{topic}

Skill:
{skill}

Return JSON with exactly these fields:
- score: integer from 1 to 5
- label: short string
- rationale: brief explanation in 1-3 sentences
"""


def _load_skill_sets(
    skills_run_dir: str | Path,
    methods: Sequence[str],
    teacher_backends: Sequence[str],
) -> Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, Any]]]:
    root = Path(skills_run_dir) / "generated_skills"
    skill_sets: Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, Any]]] = {}
    for method in methods:
        for teacher_backend in teacher_backends:
            path = root / method / f"{teacher_backend}.jsonl"
            if not path.is_file():
                continue
            rows = read_jsonl(path)
            mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for row in rows:
                for key in _skill_row_keys(row):
                    mapping[key] = row
            skill_sets[(method, teacher_backend)] = mapping
    if not skill_sets:
        raise SystemExit(f"no generated skills found under {root}")
    return skill_sets


def _default_eval_workers(requested: int) -> int:
    if requested > 0:
        return requested
    return 8


def _call_expert_api(messages: List[Dict[str, str]], args: argparse.Namespace) -> str:
    return chat_complete(
        messages,
        base_url=args.expert_api_base_url,
        api_key=args.expert_api_key,
        model=args.expert_api_model,
        timeout=args.expert_timeout,
        temperature=args.expert_temperature,
        max_tokens=args.expert_max_tokens,
    )


def _judge_skill(
    *,
    question: Dict[str, Any],
    skill_row: Dict[str, Any],
    method: str,
    teacher_backend: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    meta = question["meta"]
    out: Dict[str, Any] = {
        "source_idx": meta["source_idx"],
        "problem": meta["problem"],
        "topic": meta["topic"],
        "method": method,
        "teacher_backend": teacher_backend,
        "skill_status": skill_row.get("status"),
        "skill_text": skill_row.get("skill_text") or "",
    }
    if skill_row.get("status") != "ok" or not str(skill_row.get("skill_text") or "").strip():
        out.update(
            {
                "expert_score": None,
                "expert_label": "",
                "expert_rationale": "",
                "raw_expert_output": "",
                "judge_status": "skipped_invalid_skill",
            }
        )
        return out

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=meta["problem"],
        topic=meta["topic"] or "unknown",
        skill=str(skill_row.get("skill_text") or "").strip(),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = _call_expert_api(messages, args)
    except Exception as e:
        out.update(
            {
                "expert_score": None,
                "expert_label": "",
                "expert_rationale": "",
                "raw_expert_output": "",
                "judge_status": f"expert_error:{type(e).__name__}",
            }
        )
        return out

    parsed = parse_json_object(raw or "")
    score = parsed.get("score") if isinstance(parsed, dict) else None
    label = str(parsed.get("label") or "").strip() if isinstance(parsed, dict) else ""
    rationale = str(parsed.get("rationale") or "").strip() if isinstance(parsed, dict) else ""
    if not isinstance(score, int) or score < 1 or score > 5:
        out.update(
            {
                "expert_score": None,
                "expert_label": label,
                "expert_rationale": rationale,
                "raw_expert_output": raw,
                "judge_status": "parse_error",
            }
        )
        return out

    out.update(
        {
            "expert_score": score,
            "expert_label": label,
            "expert_rationale": rationale,
            "raw_expert_output": raw,
            "judge_status": "ok",
        }
    )
    return out


def _summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["method"], row["teacher_backend"])].append(row)
    out: List[Dict[str, Any]] = []
    for (method, teacher_backend), items in sorted(buckets.items()):
        scored = [row for row in items if row.get("expert_score") is not None]
        scores = [int(row["expert_score"]) for row in scored]
        out.append(
            {
                "method": method,
                "teacher_backend": teacher_backend,
                "n_questions": len(items),
                "scored_questions": len(scored),
                "skipped_questions": len(items) - len(scored),
                "mean_score": (sum(scores) / len(scores)) if scores else None,
                "score_5": sum(1 for value in scores if value == 5),
                "score_4": sum(1 for value in scores if value == 4),
                "score_3": sum(1 for value in scores if value == 3),
                "score_2": sum(1 for value in scores if value == 2),
                "score_1": sum(1 for value in scores if value == 1),
                "skipped_invalid_skill": sum(1 for row in items if row.get("judge_status") == "skipped_invalid_skill"),
                "parse_error": sum(1 for row in items if row.get("judge_status") == "parse_error"),
                "expert_error": sum(1 for row in items if str(row.get("judge_status") or "").startswith("expert_error:")),
            }
        )
    return out


def _write_summary(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def run_eval(args: argparse.Namespace) -> int:
    methods = _parse_choice_filter(args.methods, METHODS)
    teacher_backends = _parse_choice_filter(args.teacher_backends, TEACHER_BACKENDS)
    trajectories = read_jsonl(args.trajectories)
    questions = group_questions(trajectories, args.sample_size)
    skill_sets = _load_skill_sets(args.skills_run_dir, methods, teacher_backends)
    eval_workers = _default_eval_workers(int(getattr(args, "eval_max_workers", 0) or 0))
    output_dir = Path(args.output_dir)
    all_rows: List[Dict[str, Any]] = []

    for (method, teacher_backend), mapping in sorted(skill_sets.items()):
        ordered_results: List[Tuple[int, Dict[str, Any]]] = []

        def _task(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
            question_idx, question = item
            qkey = _question_lookup_key(question)
            skill_row = mapping.get(qkey)
            if skill_row is None and qkey[0] == "source_idx":
                skill_row = mapping.get(("problem", question["meta"]["problem"]))
            if skill_row is None:
                return question_idx, {
                    "source_idx": question["meta"]["source_idx"],
                    "problem": question["meta"]["problem"],
                    "topic": question["meta"]["topic"],
                    "method": method,
                    "teacher_backend": teacher_backend,
                    "skill_status": "missing_skill",
                    "skill_text": "",
                    "expert_score": None,
                    "expert_label": "",
                    "expert_rationale": "",
                    "raw_expert_output": "",
                    "judge_status": "missing_skill",
                }
            return question_idx, _judge_skill(
                question=question,
                skill_row=skill_row,
                method=method,
                teacher_backend=teacher_backend,
                args=args,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as executor:
            futures = [executor.submit(_task, item) for item in enumerate(questions)]
            for future in concurrent.futures.as_completed(futures):
                ordered_results.append(future.result())

        ordered_results.sort(key=lambda item: item[0])
        rows = [row for _, row in ordered_results]
        write_jsonl(output_dir / "per_skill_set" / method / f"{teacher_backend}.jsonl", rows)
        all_rows.extend(rows)

    write_jsonl(output_dir / "expert_scores.jsonl", all_rows)
    _write_summary(output_dir / "summary.json", _summary(all_rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use an external expert API to score skill helpfulness per question")
    parser.add_argument("--skills-run-dir", required=True, help="output dir from eval_source_linked_skills")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trajectories", default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--teacher-backends", nargs="*", default=None)
    parser.add_argument("--expert-api-base-url", default=os.environ.get("EVAL_TEACHER_API_BASE_URL", "").strip().rstrip("/"))
    parser.add_argument("--expert-api-model", default=os.environ.get("EVAL_TEACHER_API_MODEL", "").strip())
    parser.add_argument("--expert-api-key", default=os.environ.get("EVAL_TEACHER_API_KEY", "").strip())
    parser.add_argument("--expert-temperature", type=float, default=0.0)
    parser.add_argument("--expert-max-tokens", type=int, default=512)
    parser.add_argument("--expert-timeout", type=float, default=600.0)
    parser.add_argument("--eval-max-workers", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.expert_api_base_url or not args.expert_api_model:
        raise SystemExit("expert api config is required: --expert-api-base-url and --expert-api-model")
    return int(run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
