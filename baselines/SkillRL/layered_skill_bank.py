"""
Claude-style layered skill bank compatible with SkillRL SkillsOnlyMemory JSON schema.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .text_utils import topic_slug


def empty_bank() -> Dict[str, Any]:
    return {
        "general_skills": [],
        "task_specific_skills": {},
        "common_mistakes": [],
    }


class LayeredSkillBank:
    """Load/save `claude_style_skills.json`, template retrieve, prompt formatting."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self.skills = data if data is not None else empty_bank()
        self._ensure_keys()

    def _ensure_keys(self) -> None:
        self.skills.setdefault("general_skills", [])
        self.skills.setdefault("task_specific_skills", {})
        self.skills.setdefault("common_mistakes", [])
        if not isinstance(self.skills["task_specific_skills"], dict):
            self.skills["task_specific_skills"] = {}

    @classmethod
    def from_path(cls, path: str) -> "LayeredSkillBank":
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=2)

    def add_skills(
        self,
        new_skills: List[Dict[str, Any]],
        category: str = "general",
    ) -> int:
        """Append skills; category 'general' or a task topic slug. Returns count added."""
        added = 0
        existing = self._all_skill_ids()
        for skill in new_skills:
            sid = skill.get("skill_id")
            if sid and sid in existing:
                continue
            if category == "general":
                self.skills.setdefault("general_skills", []).append(skill)
            else:
                self.skills.setdefault("task_specific_skills", {}).setdefault(category, []).append(
                    skill
                )
            if skill.get("skill_id"):
                existing.add(skill["skill_id"])
            added += 1
        return added

    def _all_skill_ids(self) -> set:
        ids: set = set()
        for s in self.skills.get("general_skills", []):
            if s.get("skill_id"):
                ids.add(s["skill_id"])
        for _tt, lst in self.skills.get("task_specific_skills", {}).items():
            for s in lst:
                if s.get("skill_id"):
                    ids.add(s["skill_id"])
        return ids

    def retrieve(
        self,
        task_description: str,
        topic: Optional[str] = None,
        top_k: int = 6,
        mistakes_cap: int = 5,
    ) -> Dict[str, Any]:
        """
        Template-style retrieval: first top_k general (dynamic dyn_* always included),
        all task skills for topic bucket, capped mistakes.
        """
        _ = task_description  # reserved for future embedding mode
        tkey = topic_slug(topic)

        all_general = self.skills.get("general_skills", [])
        dynamic_skills = [s for s in all_general if str(s.get("skill_id", "")).startswith("dyn_")]
        static_skills = [s for s in all_general if not str(s.get("skill_id", "")).startswith("dyn_")]
        n_static = max(0, top_k - len(dynamic_skills))
        general_skills = dynamic_skills + static_skills[:n_static]

        task_skills = list(self.skills.get("task_specific_skills", {}).get(tkey, []))
        mistakes = list(self.skills.get("common_mistakes", [])[:mistakes_cap])

        return {
            "general_skills": general_skills,
            "task_specific_skills": task_skills,
            "mistakes_to_avoid": mistakes,
            "task_type": tkey,
            "task_specific_examples": [],
            "retrieval_mode": "template",
        }

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        """Match SkillRL SkillsOnlyMemory.format_for_prompt structure."""
        sections: List[str] = []
        task_type = retrieved_memories.get("task_type", "unknown")
        mode = retrieved_memories.get("retrieval_mode", "template")

        general_skills = retrieved_memories.get("general_skills", [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                title = skill.get("title", "")
                principle = skill.get("principle", "")
                lines.append(f"- **{title}**: {principle}")
            sections.append("\n".join(lines))

        task_skills = retrieved_memories.get("task_specific_skills", [])
        if task_skills:
            if mode == "embedding":
                section_title = "### Task-Relevant Skills"
            else:
                task_name = str(task_type).replace("_", " ").title()
                section_title = f"### {task_name} Skills"
            lines = [section_title]
            for skill in task_skills:
                title = skill.get("title", "")
                principle = skill.get("principle", "")
                when = skill.get("when_to_apply", "")
                lines.append(f"- **{title}**: {principle}")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        mistakes = retrieved_memories.get("mistakes_to_avoid", [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                desc = mistake.get("description", "")
                fix = mistake.get("how_to_avoid", "")
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No relevant skills found for this task."

    def skill_counts(self) -> Dict[str, int]:
        ts = self.skills.get("task_specific_skills", {})
        n_task = sum(len(v) for v in ts.values()) if isinstance(ts, dict) else 0
        return {
            "general": len(self.skills.get("general_skills", [])),
            "task_specific": n_task,
            "common_mistakes": len(self.skills.get("common_mistakes", [])),
        }


def next_id(prefix: str, existing: set, width: int = 3) -> str:
    """Next prefix_NNN not in existing."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_n = 0
    for sid in existing:
        m = pattern.match(str(sid))
        if m:
            max_n = max(max_n, int(m.group(1)))
    n = max_n + 1
    while True:
        cand = f"{prefix}{n:0{width}d}"
        if cand not in existing:
            return cand
        n += 1


def assign_skill_ids(
    skills: List[Dict[str, Any]],
    prefix: str,
    existing_ids: set,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    used = set(existing_ids)
    for s in skills:
        d = dict(s)
        sid = next_id(prefix, used, width=3)
        d["skill_id"] = sid
        used.add(sid)
        out.append(d)
    return out
