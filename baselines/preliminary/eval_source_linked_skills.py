from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from baselines.ExpeLMath.memory_parser import parse_teacher_output as parse_expel_output
from baselines.ExpeLMath.retrieve_memory import format_retrieved_prompt as format_expel_prompt
from baselines.ReasoningBankMath.io_utils import read_jsonl, write_jsonl
from baselines.ReasoningBankMath.memory_parser import parse_memory_items
from baselines.ReasoningBankMath.retrieve_memory import (
    format_retrieved_prompt as format_rbm_prompt,
)
from baselines.ReasoningBankMath.teacher import (
    chat_complete as rbm_chat_complete,
    rollout_complete as rbm_rollout_complete,
)
from baselines.SkillRL.student_rollout import grade_if_possible
from baselines.SkillRL.teacher_distill import (
    build_success_failure_block,
    chat_complete as skillrl_chat_complete,
    rollout_complete as skillrl_rollout_complete,
)
from baselines.SkillRL.text_utils import parse_json_object, topic_slug

TEACHER_BACKENDS = ("api_teacher", "server_teacher")
METHODS = ("skillrl", "reasoningbank", "expelmath")


class _RoundRobinUrls:
    def __init__(self, urls: Sequence[str]) -> None:
        if not urls:
            raise ValueError("urls is empty")
        self._urls = [str(url).rstrip("/") for url in urls]
        self._lock = threading.Lock()
        self._index = 0

    def next_url(self) -> str:
        with self._lock:
            url = self._urls[self._index]
            self._index = (self._index + 1) % len(self._urls)
            return url


def _prompt_dir(method: str) -> Path:
    base = Path(__file__).resolve().parent.parent
    mapping = {
        "skillrl": base / "SkillRL" / "prompts" / "teacher_layered_skills.txt",
        "reasoningbank": base / "ReasoningBankMath" / "prompts" / "success_memory.txt",
        "expelmath": base / "ExpeLMath" / "prompts" / "success_memory.txt",
        "solve": base / "ReasoningBankMath" / "prompts" / "skill_use_math.txt",
    }
    return mapping[method]


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _group_key(row: Dict[str, Any]) -> str:
    idx = row.get("idx")
    if idx is not None:
        return f"idx::{idx}"
    problem = str(row.get("problem") or "").strip()
    return f"problem::{problem}"


def _shorten(text: Any, limit: int = 280) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + "..."


def _extract_meta(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rep = rows[0]
    topic = rep.get("topic")
    return {
        "source_idx": rep.get("idx", rep.get("line_idx")),
        "problem": str(rep.get("problem") or "").strip(),
        "topic": topic,
        "topic_key": str(rep.get("topic_key") or topic_slug(topic)),
        "ground_truth": rep.get("ground_truth"),
    }


def group_questions(rows: Sequence[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for row in rows:
        key = _group_key(row)
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)
    out: List[Dict[str, Any]] = []
    for key in order[: max(0, sample_size)]:
        q_rows = grouped[key]
        meta = _extract_meta(q_rows)
        baseline_count = len(q_rows)
        baseline_correct = sum(1 for r in q_rows if r.get("is_correct") is True)
        out.append(
            {
                "question_id": key,
                "rows": q_rows,
                "meta": meta,
                "baseline_rollout_count": baseline_count,
                "baseline_correct_count": baseline_correct,
                "baseline_acc": (baseline_correct / baseline_count) if baseline_count else 0.0,
            }
        )
    return out


def _success_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("is_correct") is True]


def _failure_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("is_correct") is not True]


def _pattern_json(rows: Sequence[Dict[str, Any]], limit: int = 8) -> str:
    payload: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        payload.append(
            {
                "problem": _shorten(row.get("problem"), 240),
                "response_excerpt": _shorten(row.get("student_response"), 360),
                "topic": row.get("topic") or row.get("topic_key") or "unknown",
                "is_correct": row.get("is_correct"),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _skillrl_prompt(question: Dict[str, Any], template: str) -> str:
    rows = question["rows"]
    success_rows = _success_rows(rows)
    failure_rows = _failure_rows(rows)
    blocks = build_success_failure_block(success_rows, failure_rows, max_chars=12000)
    return template.format(
        topic_bucket=question["meta"]["topic_key"],
        success_trajectories_block=blocks["success_trajectories_block"],
        failure_trajectories_block=blocks["failure_trajectories_block"],
        success_patterns=blocks["success_patterns"] or _pattern_json(success_rows),
        failure_patterns=blocks["failure_patterns"] or _pattern_json(failure_rows),
    )


def _format_single_traj(row: Dict[str, Any]) -> str:
    correctness = row.get("is_correct")
    correctness_s = "true" if correctness is True else ("false" if correctness is False else "unknown")
    return (
        f"Problem:\n{row.get('problem') or ''}\n\n"
        f"Topic: {row.get('topic') or 'unknown'}\n"
        f"Correct: {correctness_s}\n"
        f"Ground truth: {row.get('ground_truth') or ''}\n\n"
        f"Student solution:\n{row.get('student_response') or ''}\n"
    )


def _rbm_prompt(question: Dict[str, Any], template: str) -> Optional[str]:
    _ = template
    success_rows = _success_rows(question["rows"])
    if not success_rows:
        return None
    return _format_single_traj(success_rows[0])


def _expel_prompt(question: Dict[str, Any], template: str) -> Optional[str]:
    _ = template
    success_rows = _success_rows(question["rows"])
    if not success_rows:
        return None
    return (
        f"Successful trajectories:\n"
        f"Success 1\n{_format_single_traj(success_rows[0]).strip()}\n"
    )


def _call_api_teacher(method: str, messages: List[Dict[str, str]], args: argparse.Namespace) -> str:
    if method == "skillrl":
        return skillrl_chat_complete(
            messages,
            base_url=args.teacher_api_base_url,
            api_key=args.teacher_api_key,
            model=args.teacher_api_model,
            timeout=args.teacher_timeout,
            temperature=args.teacher_temperature,
            max_tokens=args.teacher_max_tokens,
        )
    return rbm_chat_complete(
        messages,
        base_url=args.teacher_api_base_url,
        api_key=args.teacher_api_key,
        model=args.teacher_api_model,
        timeout=args.teacher_timeout,
        temperature=args.teacher_temperature,
        max_tokens=args.teacher_max_tokens,
    )


def _call_server_teacher(method: str, messages: List[Dict[str, str]], args: argparse.Namespace, teacher_urls: List[str]) -> str:
    if method == "skillrl":
        return skillrl_rollout_complete(
            messages,
            server_urls=teacher_urls,
            timeout=args.teacher_timeout,
            temperature=args.teacher_temperature,
            max_tokens=args.teacher_max_tokens,
            top_p=args.teacher_top_p,
            top_k=args.teacher_top_k,
        )
    return rbm_rollout_complete(
        messages,
        server_urls=teacher_urls,
        timeout=args.teacher_timeout,
        temperature=args.teacher_temperature,
        max_tokens=args.teacher_max_tokens,
        top_p=args.teacher_top_p,
        top_k=args.teacher_top_k,
    )


def _render_skillrl_partial(partial: Dict[str, Any], topic_key: str) -> str:
    sections: List[str] = []
    general = partial.get("general_skills") or []
    if general:
        lines = ["### General Principles"]
        for item in general:
            lines.append(f"- **{item.get('title', '')}**: {item.get('principle', '')}")
        sections.append("\n".join(lines))
    task_specific = partial.get("task_specific_skills") or []
    if task_specific:
        lines = [f"### {topic_key.replace('_', ' ').title()} Skills"]
        for item in task_specific:
            lines.append(f"- **{item.get('title', '')}**: {item.get('principle', '')}")
            when = str(item.get("when_to_apply") or "").strip()
            if when:
                lines.append(f"  _Apply when: {when}_")
        sections.append("\n".join(lines))
    mistakes = partial.get("common_mistakes") or []
    if mistakes:
        lines = ["### Mistakes to Avoid"]
        for item in mistakes:
            lines.append(f"- **Don't**: {item.get('description', '')}")
            how = str(item.get("how_to_avoid") or "").strip()
            if how:
                lines.append(f"  **Instead**: {how}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections).strip() or "No relevant skills found for this task."


def _render_rbm_items(items: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    row = {
        "memory_id": f"rbm_{meta['source_idx']}",
        "topic_key": meta["topic_key"],
        "status": "success",
        "memory_items": items,
    }
    return format_rbm_prompt([row]) or "No relevant memories found."


def _render_expel(parsed: Dict[str, Any], meta: Dict[str, Any]) -> str:
    row = {
        "memory_id": f"expel_{meta['source_idx']}",
        "topic_key": meta["topic_key"],
        "memory_type": "success_rule",
        "raw_rule": parsed.get("raw_rule") or "",
    }
    return format_expel_prompt([row]) or "No retrieved insights."


def generate_skill_for_question(
    *,
    question: Dict[str, Any],
    method: str,
    teacher_backend: str,
    args: argparse.Namespace,
    teacher_urls: List[str],
    templates: Dict[str, str],
) -> Dict[str, Any]:
    meta = question["meta"]
    if method == "skillrl":
        user_prompt = _skillrl_prompt(question, templates["skillrl_teacher"])
        messages = [
            {"role": "system", "content": "You output only valid JSON objects for skill distillation."},
            {"role": "user", "content": user_prompt},
        ]
    elif method == "reasoningbank":
        traj = _rbm_prompt(question, templates["rbm_teacher"])
        if traj is None:
            return {
                "source_idx": meta["source_idx"],
                "problem": meta["problem"],
                "topic": meta["topic"],
                "method": method,
                "teacher_backend": teacher_backend,
                "prompt_used": "",
                "raw_teacher_output": "",
                "parsed_skill": None,
                "status": "skipped_no_success",
                "skill_text": "",
                "skip_reason": "no_success_trajectory",
            }
        messages = [
            {"role": "system", "content": templates["rbm_teacher"]},
            {"role": "user", "content": traj},
        ]
        user_prompt = traj
    else:
        traj = _expel_prompt(question, templates["expel_teacher"])
        if traj is None:
            return {
                "source_idx": meta["source_idx"],
                "problem": meta["problem"],
                "topic": meta["topic"],
                "method": method,
                "teacher_backend": teacher_backend,
                "prompt_used": "",
                "raw_teacher_output": "",
                "parsed_skill": None,
                "status": "skipped_no_success",
                "skill_text": "",
                "skip_reason": "no_success_trajectory",
            }
        messages = [
            {"role": "system", "content": templates["expel_teacher"]},
            {"role": "user", "content": traj},
        ]
        user_prompt = traj

    try:
        raw = (
            _call_api_teacher(method, messages, args)
            if teacher_backend == "api_teacher"
            else _call_server_teacher(method, messages, args, teacher_urls)
        )
    except Exception as e:
        return {
            "source_idx": meta["source_idx"],
            "problem": meta["problem"],
            "topic": meta["topic"],
            "method": method,
            "teacher_backend": teacher_backend,
            "prompt_used": user_prompt,
            "raw_teacher_output": "",
            "parsed_skill": None,
            "status": "teacher_error",
            "skill_text": "",
            "skip_reason": f"teacher_error:{type(e).__name__}",
        }

    if method == "skillrl":
        parsed = parse_json_object(raw)
        status = "ok" if parsed else "parse_error"
        skill_text = _render_skillrl_partial(parsed or {}, meta["topic_key"]) if parsed else ""
    elif method == "reasoningbank":
        items = parse_memory_items(raw)
        parsed = {"memory_items": items}
        status = "ok" if items else "parse_error"
        skill_text = _render_rbm_items(items, meta) if items else ""
    else:
        parsed = parse_expel_output(raw, default_memory_type="success_rule")
        items = list(parsed.get("memory_items") or [])
        status = "ok" if items else "parse_error"
        skill_text = _render_expel(parsed, meta) if items else ""

    return {
        "source_idx": meta["source_idx"],
        "problem": meta["problem"],
        "topic": meta["topic"],
        "method": method,
        "teacher_backend": teacher_backend,
        "prompt_used": user_prompt,
        "raw_teacher_output": raw,
        "parsed_skill": parsed,
        "status": status,
        "skill_text": skill_text,
        "skip_reason": "" if status == "ok" else status,
    }


def run_student_rollout(
    *,
    question: Dict[str, Any],
    skill_row: Dict[str, Any],
    solve_template: str,
    server_urls: List[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = question["meta"]
    if skill_row["status"] != "ok":
        detail = {
            "source_idx": meta["source_idx"],
            "problem": meta["problem"],
            "method": skill_row["method"],
            "teacher_backend": skill_row["teacher_backend"],
            "baseline_correct_count": question["baseline_correct_count"],
            "baseline_rollout_count": question["baseline_rollout_count"],
            "baseline_acc": question["baseline_acc"],
            "skill_correct_count": None,
            "skill_rollout_count": 0,
            "skill_acc": None,
            "delta": None,
            "skip_reason": skill_row.get("skip_reason") or skill_row["status"],
        }
        return [], detail

    prompt = solve_template.format(
        skill=skill_row["skill_text"],
        retrieved_context=skill_row["skill_text"],
        question=meta["problem"],
    )
    try:
        responses = _rollout_prompt(
            server_urls=server_urls,
            prompt=prompt,
            question=meta["problem"],
            ground_truth=meta["ground_truth"],
            rollout_n=args.student_rollout_n,
            args=args,
        )
    except Exception as e:
        detail = {
            "source_idx": meta["source_idx"],
            "problem": meta["problem"],
            "method": skill_row["method"],
            "teacher_backend": skill_row["teacher_backend"],
            "baseline_correct_count": question["baseline_correct_count"],
            "baseline_rollout_count": question["baseline_rollout_count"],
            "baseline_acc": question["baseline_acc"],
            "skill_correct_count": None,
            "skill_rollout_count": 0,
            "skill_acc": None,
            "delta": None,
            "skip_reason": f"student_error:{type(e).__name__}",
        }
        return [], detail
    rollout_rows: List[Dict[str, Any]] = []
    correct = 0
    for attempt_idx, text in enumerate(responses):
        is_correct = grade_if_possible(text, meta["ground_truth"])
        if is_correct is True:
            correct += 1
        rollout_rows.append(
            {
                "source_idx": meta["source_idx"],
                "problem": meta["problem"],
                "method": skill_row["method"],
                "teacher_backend": skill_row["teacher_backend"],
                "attempt_idx": attempt_idx,
                "prompt": prompt,
                "student_response": text,
                "is_correct": is_correct,
                "ground_truth": meta["ground_truth"],
            }
        )
    skill_acc = correct / len(responses) if responses else 0.0
    detail = {
        "source_idx": meta["source_idx"],
        "problem": meta["problem"],
        "method": skill_row["method"],
        "teacher_backend": skill_row["teacher_backend"],
        "baseline_correct_count": question["baseline_correct_count"],
        "baseline_rollout_count": question["baseline_rollout_count"],
        "baseline_acc": question["baseline_acc"],
        "skill_correct_count": correct,
        "skill_rollout_count": len(responses),
        "skill_acc": skill_acc,
        "delta": skill_acc - question["baseline_acc"],
        "skip_reason": "",
    }
    return rollout_rows, detail


def _summary(details: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in details:
        buckets[(row["method"], row["teacher_backend"])].append(row)
    out: List[Dict[str, Any]] = []
    for (method, teacher_backend), rows in sorted(buckets.items()):
        evaluated = [r for r in rows if r["skill_acc"] is not None]
        baseline_acc = (
            sum(float(r["baseline_acc"]) for r in evaluated) / len(evaluated)
            if evaluated
            else None
        )
        skill_acc = (
            sum(float(r["skill_acc"]) for r in evaluated) / len(evaluated)
            if evaluated
            else None
        )
        abs_delta = (skill_acc - baseline_acc) if (skill_acc is not None and baseline_acc is not None) else None
        rel_delta = (abs_delta / baseline_acc) if (abs_delta is not None and baseline_acc not in (None, 0.0)) else None
        out.append(
            {
                "method": method,
                "teacher_backend": teacher_backend,
                "n_questions": len(rows),
                "evaluated_questions": len(evaluated),
                "skipped_questions": len(rows) - len(evaluated),
                "baseline_acc": baseline_acc,
                "skill_acc": skill_acc,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
                "improved": sum(1 for r in evaluated if float(r["delta"]) > 0),
                "degraded": sum(1 for r in evaluated if float(r["delta"]) < 0),
                "unchanged": sum(1 for r in evaluated if float(r["delta"]) == 0),
                "skipped_no_success_skill": sum(1 for r in rows if r["skip_reason"] == "no_success_trajectory"),
            }
        )
    return out


def _write_summary(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _resume_key(source_idx: Any) -> str:
    return str(source_idx)


def _index_rows_by_source(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if "source_idx" not in row:
            continue
        out[_resume_key(row["source_idx"])] = row
    return out


def _index_rollout_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "source_idx" not in row:
            continue
        out[_resume_key(row["source_idx"])].append(row)
    return dict(out)


def _expected_source_keys(questions: Sequence[Dict[str, Any]]) -> List[str]:
    return [_resume_key(question["meta"]["source_idx"]) for question in questions]


def resume_status(args: argparse.Namespace) -> Dict[str, Any]:
    methods = _parse_methods(args.method)
    trajectories = read_jsonl(args.trajectories)
    questions = group_questions(trajectories, args.sample_size)
    expected_keys = _expected_source_keys(questions)
    expected_set = set(expected_keys)
    output_dir = Path(args.output_dir)
    details_rows = _read_jsonl_if_exists(output_dir / "details.jsonl")

    statuses: List[Dict[str, Any]] = []
    complete = True
    for method in methods:
        for teacher_backend in TEACHER_BACKENDS:
            skill_rows = _read_jsonl_if_exists(output_dir / "generated_skills" / method / f"{teacher_backend}.jsonl")
            rollout_rows = _read_jsonl_if_exists(output_dir / "student_rollout" / method / f"{teacher_backend}.jsonl")
            method_details = [
                row for row in details_rows
                if row.get("method") == method and row.get("teacher_backend") == teacher_backend
            ]
            skill_by_source = _index_rows_by_source(skill_rows)
            rollout_by_source = _index_rollout_rows(rollout_rows)
            detail_by_source = _index_rows_by_source(method_details)

            missing_skill = sorted(expected_set - set(skill_by_source.keys()))
            missing_detail = sorted(expected_set - set(detail_by_source.keys()))
            missing_rollout: List[str] = []
            for source_key in expected_keys:
                detail = detail_by_source.get(source_key)
                if detail is None:
                    continue
                rollout_count = int(detail.get("skill_rollout_count") or 0)
                if rollout_count > 0 and not rollout_by_source.get(source_key):
                    missing_rollout.append(source_key)

            backend_complete = not (missing_skill or missing_detail or missing_rollout)
            complete = complete and backend_complete
            statuses.append(
                {
                    "method": method,
                    "teacher_backend": teacher_backend,
                    "complete": backend_complete,
                    "expected_questions": len(expected_keys),
                    "missing_skill": missing_skill,
                    "missing_detail": missing_detail,
                    "missing_rollout": missing_rollout,
                }
            )

    return {
        "complete": complete,
        "methods": methods,
        "n_questions": len(expected_keys),
        "statuses": statuses,
    }


def _parse_methods(raw: str) -> List[str]:
    value = (raw or "all").strip().lower()
    if value == "all":
        return list(METHODS)
    if value not in METHODS:
        raise SystemExit(f"invalid --method: {raw}")
    return [value]


def _resolve_urls(cli_urls: Optional[Sequence[str]]) -> List[str]:
    from baselines.SkillRL.deepmath_io import resolve_rollout_server_urls

    return resolve_rollout_server_urls(cli_urls)


def _default_eval_workers(
    *,
    student_urls: Sequence[str],
    teacher_urls: Sequence[str],
    requested: int,
) -> int:
    if requested > 0:
        return requested
    return max(1, len(student_urls), len(teacher_urls))


def _rollout_prompt(
    *,
    server_urls: List[str],
    prompt: str,
    question: str,
    ground_truth: Any,
    rollout_n: int,
    args: argparse.Namespace,
) -> List[str]:
    if not server_urls:
        raise ValueError("server_urls is empty")
    base_url = random.choice(server_urls).rstrip("/")
    payload: Dict[str, Any] = {
        "data_records": [{"prompt": prompt, "question": question, "gt": ground_truth or ""}],
        "num_questions": 1,
        "suffix": "preliminary_source_linked_eval",
        "rollout_n": rollout_n,
        "max_tokens": args.student_max_tokens,
        "temperature": args.student_temperature,
        "top_p": args.student_top_p,
        "top_k": getattr(args, "student_top_k", 50),
    }
    with httpx.Client(timeout=args.student_timeout) as client:
        resp = client.post(f"{base_url}/rollout", json=payload)
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"No results in rollout response: {data.keys()}")
    responses = (results[0] or {}).get("responses") or []
    return [str(x) for x in responses]


def run_eval(args: argparse.Namespace) -> int:
    methods = _parse_methods(args.method)
    trajectories = read_jsonl(args.trajectories)
    questions = group_questions(trajectories, args.sample_size)
    student_urls = _resolve_urls(args.server_urls)
    teacher_urls = _resolve_urls(args.teacher_server_urls or args.server_urls)
    student_pool = _RoundRobinUrls(student_urls)
    teacher_pool = _RoundRobinUrls(teacher_urls)
    eval_workers = _default_eval_workers(
        student_urls=student_urls,
        teacher_urls=teacher_urls,
        requested=int(getattr(args, "eval_max_workers", 0) or 0),
    )
    templates = {
        "skillrl_teacher": _load_text(_prompt_dir("skillrl")),
        "rbm_teacher": _load_text(_prompt_dir("reasoningbank")),
        "expel_teacher": _load_text(_prompt_dir("expelmath")),
        "solve": _load_text(_prompt_dir("solve")),
    }
    output_dir = Path(args.output_dir)
    existing_details = _read_jsonl_if_exists(output_dir / "details.jsonl") if getattr(args, "resume", False) else []
    details_rows: List[Dict[str, Any]] = [row for row in existing_details if row.get("method") not in set(methods)]

    for method in methods:
        for teacher_backend in TEACHER_BACKENDS:
            skill_path = output_dir / "generated_skills" / method / f"{teacher_backend}.jsonl"
            rollout_path = output_dir / "student_rollout" / method / f"{teacher_backend}.jsonl"
            existing_skill_rows = _read_jsonl_if_exists(skill_path) if getattr(args, "resume", False) else []
            existing_rollout_rows = _read_jsonl_if_exists(rollout_path) if getattr(args, "resume", False) else []
            existing_method_details = [
                row for row in existing_details
                if row.get("method") == method and row.get("teacher_backend") == teacher_backend
            ]
            skill_by_source = _index_rows_by_source(existing_skill_rows)
            rollout_by_source = _index_rollout_rows(existing_rollout_rows)
            detail_by_source = _index_rows_by_source(existing_method_details)

            skill_rows: List[Dict[str, Any]] = []
            rollout_rows: List[Dict[str, Any]] = []
            ordered_results: List[Tuple[int, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]] = []
            pending_items: List[Tuple[int, Dict[str, Any]]] = []

            def _process_question(
                item: Tuple[int, Dict[str, Any]],
                preloaded_skill_row: Optional[Dict[str, Any]] = None,
            ) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
                question_idx, question = item
                skill_row = preloaded_skill_row
                if skill_row is None:
                    bound_teacher_urls = [teacher_pool.next_url()] if teacher_backend == "server_teacher" else teacher_urls
                    skill_row = generate_skill_for_question(
                        question=question,
                        method=method,
                        teacher_backend=teacher_backend,
                        args=args,
                        teacher_urls=bound_teacher_urls,
                        templates=templates,
                    )
                bound_student_urls = [student_pool.next_url()]
                per_attempt_rows, detail = run_student_rollout(
                    question=question,
                    skill_row=skill_row,
                    solve_template=templates["solve"],
                    server_urls=bound_student_urls,
                    args=args,
                )
                return question_idx, skill_row, per_attempt_rows, detail

            for item in enumerate(questions):
                question_idx, question = item
                source_key = _resume_key(question["meta"]["source_idx"])
                existing_skill = skill_by_source.get(source_key)
                existing_detail = detail_by_source.get(source_key)
                existing_rollouts = rollout_by_source.get(source_key, [])
                if existing_skill is None:
                    pending_items.append(item)
                    continue
                if existing_detail is not None and (
                    existing_rollouts or int(existing_detail.get("skill_rollout_count") or 0) == 0
                ):
                    ordered_results.append((question_idx, existing_skill, existing_rollouts, existing_detail))
                    continue
                pending_items.append(item)

            with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as executor:
                futures = []
                for item in pending_items:
                    source_key = _resume_key(item[1]["meta"]["source_idx"])
                    futures.append(executor.submit(_process_question, item, skill_by_source.get(source_key)))
                for future in concurrent.futures.as_completed(futures):
                    ordered_results.append(future.result())

            ordered_results.sort(key=lambda row: row[0])
            for _, skill_row, per_attempt_rows, detail in ordered_results:
                skill_rows.append(skill_row)
                rollout_rows.extend(per_attempt_rows)
                details_rows.append(detail)

            write_jsonl(skill_path, skill_rows)
            write_jsonl(rollout_path, rollout_rows)

    write_jsonl(output_dir / "details.jsonl", details_rows)
    _write_summary(output_dir / "summary.json", _summary(details_rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate source-linked skills with dual teacher backends")
    p.add_argument("--trajectories", default="Skill_Evo/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", default="all", choices=["skillrl", "reasoningbank", "expelmath", "all"])
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--server-urls", nargs="*", default=None)
    p.add_argument("--teacher-server-urls", nargs="*", default=None)
    p.add_argument("--served-model-name", default="")
    p.add_argument("--teacher-api-base-url", default=os.environ.get("EVAL_TEACHER_API_BASE_URL", "").strip().rstrip("/"))
    p.add_argument("--teacher-api-model", default=os.environ.get("EVAL_TEACHER_API_MODEL", "").strip())
    p.add_argument("--teacher-api-key", default=os.environ.get("EVAL_TEACHER_API_KEY", "").strip())
    p.add_argument("--student-rollout-n", type=int, default=4)
    p.add_argument("--teacher-temperature", type=float, default=0.2)
    p.add_argument("--teacher-top-p", type=float, default=0.95)
    p.add_argument("--teacher-top-k", type=int, default=50)
    p.add_argument("--teacher-max-tokens", type=int, default=4096)
    p.add_argument("--teacher-timeout", type=float, default=600.0)
    p.add_argument("--student-temperature", type=float, default=0.7)
    p.add_argument("--student-top-p", type=float, default=0.95)
    p.add_argument("--student-top-k", type=int, default=50)
    p.add_argument("--student-max-tokens", type=int, default=4096)
    p.add_argument("--student-timeout", type=float, default=600.0)
    p.add_argument("--student-max-retries", type=int, default=3)
    p.add_argument("--student-max-concurrent", type=int, default=0)
    p.add_argument("--eval-max-workers", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="reuse existing generated_skills/student_rollout/details outputs when present")
    p.add_argument("--resume-check-only", action="store_true", help="exit 0 if requested outputs are already complete; exit 10 otherwise")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume_check_only:
        status = resume_status(args)
        print(json.dumps(status, ensure_ascii=False))
        return 0 if status["complete"] else 10
    if not args.teacher_api_base_url or not args.teacher_api_model:
        raise SystemExit("teacher api config is required: --teacher-api-base-url and --teacher-api-model")
    if args.student_rollout_n <= 0:
        raise SystemExit("--student-rollout-n must be > 0")
    return int(run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
