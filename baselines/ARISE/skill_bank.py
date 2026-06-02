from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_skill_use_template() -> str:
    return (_prompt_dir() / "skill_use_math.txt").read_text(encoding="utf-8")


@dataclass
class AriseSkillRecord:
    skill_id: str
    skill_name: str
    problem_type: str
    key_insight: str
    method: List[str]
    check: str
    utility: float
    usage_count: int
    source: str

    def to_prompt_payload(self) -> Dict[str, Any]:
        method_text = "\n".join(
            f"{idx + 1}. {step}" for idx, step in enumerate(self.method) if str(step).strip()
        )
        return {
            "skill_name": self.skill_name,
            "problem_type": self.problem_type,
            "key_insight": self.key_insight,
            "method": method_text,
            "check": self.check,
            "utility": self.utility,
            "usage_count": self.usage_count,
            "source": self.source,
            "skill_id": self.skill_id,
        }

    def retrieval_text(self) -> str:
        parts: List[str] = []
        if self.problem_type:
            parts.append(self.problem_type.replace("_", " "))
        if self.key_insight:
            parts.append(self.key_insight)
        if self.method:
            parts.append(" ".join(step.strip() for step in self.method if str(step).strip()))
        if self.check:
            parts.append(self.check)
        return " ".join(part.strip() for part in parts if part and str(part).strip())


def _as_list_of_str(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


class AriseSkillBank:
    def __init__(self, records: Sequence[AriseSkillRecord]) -> None:
        self.records = list(records)

    @classmethod
    def from_path(cls, path: str | Path, *, include_reservoir: bool = False) -> "AriseSkillBank":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records: List[AriseSkillRecord] = []
        sections = [("cache", payload.get("cache", []))]
        if include_reservoir:
            sections.append(("reservoir", payload.get("reservoir", [])))
        for source, rows in sections:
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                doc = row.get("document") or {}
                if not isinstance(doc, dict):
                    continue
                skill_name = str(doc.get("skill_name") or "").strip()
                if not skill_name:
                    continue
                records.append(
                    AriseSkillRecord(
                        skill_id=skill_name,
                        skill_name=skill_name,
                        problem_type=str(doc.get("problem_type") or "").strip(),
                        key_insight=str(doc.get("key_insight") or "").strip(),
                        method=_as_list_of_str(doc.get("method")),
                        check=str(doc.get("check") or "").strip(),
                        utility=float(row.get("utility", 0.0) or 0.0),
                        usage_count=int(row.get("usage_count", 0) or 0),
                        source=source,
                    )
                )
        return cls(records)

    def build_candidates(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in self.records:
            text = rec.retrieval_text()
            if not text:
                continue
            item = rec.to_prompt_payload()
            out.append(
                {
                    "id": rec.skill_id,
                    "problem_type": text,
                    "utility": rec.utility,
                    "_item": item,
                }
            )
        return out


def format_skill_prompt(skills: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for skill in skills:
        payload = {
            "skill name": str(skill.get("skill_name") or ""),
            "problem type": str(skill.get("problem_type") or ""),
            "key insight": str(skill.get("key_insight") or ""),
            "method": str(skill.get("method") or ""),
        }
        parts.append(json.dumps(payload, ensure_ascii=False))
    return "\n\n---\n\n".join(parts)
