"""Build RL jsonl/parquet from DeepMath using retriever-ranked skills."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from .layered_skill_bank import LayeredSkillBank
from .student_rollout import extract_ground_truth, extract_meta, extract_problem
from .text_utils import topic_slug


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_skill_use_template() -> str:
    return (_prompt_dir() / "skill_use_math.txt").read_text(encoding="utf-8")


def _normalize_ground_truth(gt: Optional[str]) -> str:
    if gt is None:
        return ""
    return gt.strip()


def _validate_prepare_args(args: argparse.Namespace) -> None:
    for name in ("top_k_general", "top_k_task", "top_k_mistake"):
        value = int(getattr(args, name))
        if value < 0:
            raise SystemExit(f"prepare-rl-data: {name} must be >= 0")
    if args.top_k_general == 0 and args.top_k_task == 0 and args.top_k_mistake == 0:
        raise SystemExit("prepare-rl-data: at least one top_k must be > 0")
    if args.mode not in {"embedding", "hybrid"}:
        raise SystemExit("prepare-rl-data: mode must be 'embedding' or 'hybrid'")


def _skill_candidate_text(item: Dict[str, Any]) -> str:
    when = str(item.get("when_to_apply") or "").strip()
    if when:
        return when
    title = str(item.get("title") or "").strip()
    principle = str(item.get("principle") or "").strip()
    return " ".join(part for part in (title, principle) if part).strip()


def _mistake_candidate_text(item: Dict[str, Any]) -> str:
    desc = str(item.get("description") or "").strip()
    fix = str(item.get("how_to_avoid") or "").strip()
    return " ".join(part for part in (desc, fix) if part).strip()


def build_general_candidates(skills: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in skills:
        text = _skill_candidate_text(item)
        sid = str(item.get("skill_id") or "").strip()
        if not text or not sid:
            continue
        payload = dict(item)
        payload["_retrieval_id"] = sid
        out.append({"id": sid, "problem_type": text, "utility": 0.0, "_item": payload})
    return out


def build_task_candidates(skills: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return build_general_candidates(skills)


def build_mistake_candidates(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        text = _mistake_candidate_text(item)
        if not text:
            continue
        rid = f"cm_{idx:06d}"
        payload = dict(item)
        payload["_retrieval_id"] = rid
        out.append(
            {
                "id": rid,
                "problem_type": text,
                "utility": 0.0,
                "_item": payload,
            }
        )
    return out


def _post_rank(
    *,
    retriever_url: str,
    question: str,
    candidates: Sequence[Dict[str, Any]],
    mode: str,
    retrieve_lambda: float,
    top_k: int,
) -> List[int]:
    payload = {
        "question": question,
        "candidates": [
            {
                "id": c["id"],
                "problem_type": c["problem_type"],
                "utility": c["utility"],
            }
            for c in candidates
        ],
        "mode": mode,
        "retrieve_lambda": retrieve_lambda,
        "top_k": top_k,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{retriever_url.rstrip('/')}/rank",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        raise RuntimeError(f"retriever request failed: {e}") from e
    if not data.get("ok"):
        raise RuntimeError(f"retriever error: {data.get('error')}")
    ranked = data.get("ranked_indices")
    if not isinstance(ranked, list):
        raise RuntimeError("retriever error: missing ranked_indices")
    out: List[int] = []
    for x in ranked:
        if not isinstance(x, int):
            raise RuntimeError("retriever error: ranked_indices must contain ints")
        out.append(x)
    return out


def retrieve_bucket(
    *,
    question: str,
    candidates: Sequence[Dict[str, Any]],
    top_k: int,
    retriever_url: str,
    mode: str,
    retrieve_lambda: float,
) -> List[Dict[str, Any]]:
    if top_k <= 0 or not candidates:
        return []
    ranked_indices = _post_rank(
        retriever_url=retriever_url,
        question=question,
        candidates=candidates,
        mode=mode,
        retrieve_lambda=retrieve_lambda,
        top_k=top_k,
    )
    out: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        if idx < 0 or idx >= len(candidates):
            raise RuntimeError(f"retriever returned out-of-range index: {idx}")
        out.append(candidates[idx]["_item"])
    return out


def _task_section_title(topic_key: str) -> str:
    if not topic_key or topic_key == "unknown":
        return "### Unknown Skills"
    return f"### {str(topic_key).replace('_', ' ').title()} Skills"


def format_skill_prompt(
    *,
    topic_key: str,
    general_skills: Sequence[Dict[str, Any]],
    task_skills: Sequence[Dict[str, Any]],
    mistakes: Sequence[Dict[str, Any]],
) -> str:
    sections: List[str] = []

    if general_skills:
        lines = ["### General Principles"]
        for skill in general_skills:
            title = str(skill.get("title") or "").strip()
            principle = str(skill.get("principle") or "").strip()
            lines.append(f"- **{title}**: {principle}")
        sections.append("\n".join(lines))

    if task_skills:
        lines = [_task_section_title(topic_key)]
        for skill in task_skills:
            title = str(skill.get("title") or "").strip()
            principle = str(skill.get("principle") or "").strip()
            when = str(skill.get("when_to_apply") or "").strip()
            lines.append(f"- **{title}**: {principle}")
            if when:
                lines.append(f"  _Apply when: {when}_")
        sections.append("\n".join(lines))

    if mistakes:
        lines = ["### Mistakes to Avoid"]
        for item in mistakes:
            desc = str(item.get("description") or "").strip()
            fix = str(item.get("how_to_avoid") or "").strip()
            if desc:
                lines.append(f"- **Don't**: {desc}")
                if fix:
                    lines.append(f"  **Instead**: {fix}")
        sections.append("\n".join(lines))

    return "\n\n".join(sec for sec in sections if sec.strip())


def _prepare_candidates(bank: LayeredSkillBank, topic_key: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    general = build_general_candidates(bank.skills.get("general_skills", []))
    task = build_task_candidates(bank.skills.get("task_specific_skills", {}).get(topic_key, []))
    mistakes = build_mistake_candidates(bank.skills.get("common_mistakes", []))
    return general, task, mistakes


def run_prepare_rl_data(args: argparse.Namespace) -> int:
    _validate_prepare_args(args)
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("prepare-rl-data 需要 pandas、pyarrow: pip install pandas pyarrow") from e

    bank = LayeredSkillBank.from_path(args.skills_json)
    template = load_skill_use_template()

    rows: List[Dict[str, Any]] = []
    skipped = 0

    with open(args.deepmath_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < args.start:
                continue
            if args.end is not None and line_no >= args.end:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[prepare-rl-data] skip line {line_no}: bad json: {e}", file=sys.stderr)
                skipped += 1
                continue
            if not isinstance(raw, dict):
                print(f"[prepare-rl-data] skip line {line_no}: not a JSON object", file=sys.stderr)
                skipped += 1
                continue

            problem = extract_problem(raw)
            if not problem:
                print(f"[prepare-rl-data] skip line {line_no}: no problem", file=sys.stderr)
                skipped += 1
                continue

            meta = extract_meta(raw, line_no)
            topic = meta.get("topic")
            topic_key = topic_slug(topic)
            gt = extract_ground_truth(raw)
            general_candidates, task_candidates, mistake_candidates = _prepare_candidates(bank, topic_key)

            try:
                retrieved_general = retrieve_bucket(
                    question=problem,
                    candidates=general_candidates,
                    top_k=args.top_k_general,
                    retriever_url=args.retriever_url,
                    mode=args.mode,
                    retrieve_lambda=args.retrieve_lambda,
                )
                retrieved_task = retrieve_bucket(
                    question=problem,
                    candidates=task_candidates,
                    top_k=args.top_k_task,
                    retriever_url=args.retriever_url,
                    mode=args.mode,
                    retrieve_lambda=args.retrieve_lambda,
                )
                retrieved_mistakes = retrieve_bucket(
                    question=problem,
                    candidates=mistake_candidates,
                    top_k=args.top_k_mistake,
                    retriever_url=args.retriever_url,
                    mode=args.mode,
                    retrieve_lambda=args.retrieve_lambda,
                )
            except Exception as e:
                if args.fail_on_retrieve_error:
                    raise SystemExit(f"prepare-rl-data: retrieve failed at line {line_no}: {e}") from e
                print(f"[prepare-rl-data] skip line {line_no}: retrieve failed: {e}", file=sys.stderr)
                skipped += 1
                continue

            skill_text = format_skill_prompt(
                topic_key=topic_key,
                general_skills=retrieved_general,
                task_skills=retrieved_task,
                mistakes=retrieved_mistakes,
            )
            if not skill_text.strip():
                print(f"[prepare-rl-data] skip line {line_no}: no retrieved skills", file=sys.stderr)
                skipped += 1
                continue

            try:
                user_content = template.format(skill=skill_text, question=problem)
            except Exception as e:
                print(f"[prepare-rl-data] skip line {line_no}: template failed: {e}", file=sys.stderr)
                skipped += 1
                continue

            prompt = [{"role": "user", "content": user_content}]
            extra = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
            ex = dict(extra)
            ex["problem"] = problem
            ex["topic"] = topic
            ex["topic_key"] = topic_key
            ex["idx"] = meta.get("idx", line_no)
            ex["retrieval_mode"] = args.mode
            ex["retriever_url"] = args.retriever_url
            ex["top_k_general"] = args.top_k_general
            ex["top_k_task"] = args.top_k_task
            ex["top_k_mistake"] = args.top_k_mistake
            ex["retrieved_general_skill_ids"] = [str(s.get("_retrieval_id") or s.get("skill_id") or "") for s in retrieved_general]
            ex["retrieved_task_skill_ids"] = [str(s.get("_retrieval_id") or s.get("skill_id") or "") for s in retrieved_task]
            ex["retrieved_common_mistake_ids"] = [str(s.get("_retrieval_id") or "") for s in retrieved_mistakes]
            ex["retrieved_general_count"] = len(retrieved_general)
            ex["retrieved_task_count"] = len(retrieved_task)
            ex["retrieved_mistake_count"] = len(retrieved_mistakes)
            ex["general_candidates_count"] = len(general_candidates)
            ex["task_candidates_count"] = len(task_candidates)
            ex["mistake_candidates_count"] = len(mistake_candidates)

            rec: Dict[str, Any] = {
                "prompt": prompt,
                "reward_model": {"ground_truth": _normalize_ground_truth(gt)},
                "data_source": "DeepMath-103K",
                "extra_info": ex,
            }
            if args.keep_raw_prompt and isinstance(raw.get("prompt"), list):
                rec["original_prompt"] = copy.deepcopy(raw["prompt"])
            rows.append(rec)

    if not rows:
        print("[prepare-rl-data] no rows", file=sys.stderr)
        return 1

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[prepare-rl-data] jsonl={len(rows)} -> {out_jsonl}", file=sys.stderr)

    out_parquet = Path(args.output_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    print(f"[prepare-rl-data] parquet={len(rows)} -> {out_parquet}", file=sys.stderr)
    print(f"[prepare-rl-data] skipped={skipped}", file=sys.stderr)
    return 0


def build_prepare_rl_data_parser(sub: Any) -> None:
    p = sub.add_parser("prepare-rl-data", help="DeepMath + retriever-ranked skills -> RL jsonl/parquet")
    p.add_argument("--deepmath-jsonl", default="/home/ycy/sdi/data/DeepMath-103K.jsonl")
    p.add_argument("--skills-json", default="baselines/SkillRL/outputs/skills_from_rollout_teacher.json")
    p.add_argument("--retriever-url", default="http://127.0.0.1:8766")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--top-k-general", type=int, default=3)
    p.add_argument("--top-k-task", type=int, default=3)
    p.add_argument("--top-k-mistake", type=int, default=2)
    p.add_argument("--mode", default="embedding", choices=["embedding", "hybrid"])
    p.add_argument("--retrieve-lambda", type=float, default=0.5)
    p.add_argument("--output-jsonl", default="baselines/SkillRL/outputs/deepmath_skills_rl.jsonl")
    p.add_argument("--output-parquet", default="baselines/SkillRL/outputs/deepmath_skills_rl.parquet")
    p.add_argument("--keep-raw-prompt", action="store_true")
    p.add_argument("--fail-on-retrieve-error", action="store_true")
    p.set_defaults(_run=run_prepare_rl_data)
