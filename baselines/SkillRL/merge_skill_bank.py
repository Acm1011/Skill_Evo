"""Merge partial skill dicts into a single claude_style bank with deduplication."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .layered_skill_bank import LayeredSkillBank, assign_skill_ids


def _norm_key(title: str, principle: str) -> Tuple[str, str]:
    t = (title or "").strip().lower()
    p = (principle or "").strip().lower()[:160]
    return t, p


def merge_partial_into_bank(
    bank: LayeredSkillBank,
    partial: Dict[str, Any],
    topic_key: str,
    *,
    gen_prefix: str = "gen_",
    task_prefix: str = "dyn_",
) -> None:
    """
    Merge one teacher JSON object into bank.
    partial keys: general_skills, task_specific_skills (list), common_mistakes (list).
    Assigns skill_id with gen_* for general, dyn_* for task-specific.
    """
    existing = bank._all_skill_ids()

    gen_in = partial.get("general_skills") or []
    if isinstance(gen_in, list):
        deduped: List[Dict[str, Any]] = []
        seen = {
            _norm_key(s.get("title", ""), s.get("principle", ""))
            for s in bank.skills.get("general_skills", [])
        }
        for s in gen_in:
            if not isinstance(s, dict):
                continue
            k = _norm_key(s.get("title", ""), s.get("principle", ""))
            if k in seen or not k[0]:
                continue
            seen.add(k)
            deduped.append(s)
        for s in assign_skill_ids(deduped, gen_prefix, existing):
            existing.add(s["skill_id"])
            bank.skills.setdefault("general_skills", []).append(s)

    task_in = partial.get("task_specific_skills") or []
    if isinstance(task_in, list):
        bucket = bank.skills.setdefault("task_specific_skills", {}).setdefault(topic_key, [])
        seen_t = {
            _norm_key(s.get("title", ""), s.get("principle", "")) for s in bucket
        }
        deduped_t: List[Dict[str, Any]] = []
        for s in task_in:
            if not isinstance(s, dict):
                continue
            k = _norm_key(s.get("title", ""), s.get("principle", ""))
            if k in seen_t or not k[0]:
                continue
            seen_t.add(k)
            deduped_t.append(s)
        for s in assign_skill_ids(deduped_t, task_prefix, existing):
            existing.add(s["skill_id"])
            bucket.append(s)

    cm_in = partial.get("common_mistakes") or []
    if isinstance(cm_in, list):
        mistakes = bank.skills.setdefault("common_mistakes", [])
        seen_m = {
            _norm_key(m.get("description", ""), m.get("how_to_avoid", ""))
            for m in mistakes
        }
        for m in cm_in:
            if not isinstance(m, dict):
                continue
            k = _norm_key(m.get("description", ""), m.get("how_to_avoid", ""))
            if k in seen_m or not k[0]:
                continue
            seen_m.add(k)
            mistakes.append(
                {
                    "description": (m.get("description") or "").strip(),
                    "how_to_avoid": (m.get("how_to_avoid") or "").strip(),
                }
            )


def cap_mistakes(bank: LayeredSkillBank, max_items: int) -> None:
    m = bank.skills.get("common_mistakes", [])
    if len(m) > max_items:
        bank.skills["common_mistakes"] = m[:max_items]
