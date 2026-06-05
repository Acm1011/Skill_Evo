#!/usr/bin/env python3
"""Generate skill-augmented test datasets from input jsonl files."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_SKILLRL_OUTPUT_DIR = Path("/home/ycy/sdi/skill_saved/Skill_Evo/baseline/checkpoints/skillrl_qwen3_4b")
DEFAULT_ARISE_OUTPUT_DIR = Path("/home/ycy/sdi/Skill_Evo/baselines/ARISE/outputs/prepared")


def _add_skill_src_to_path() -> None:
    root = Path(__file__).resolve().parent
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_template(template_path: Path) -> str:
    with template_path.open(encoding="utf-8") as f:
        return f.read()


def _apply_first_user_content(prompt: Any, new_content: str) -> bool:
    if not isinstance(prompt, list):
        return False
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
            msg["content"] = new_content
            return True
    return False


def _skills_block_minimal(skills: list[dict[str, Any]]) -> str:
    """Only keep 4 fields for prompt injection."""
    if not skills:
        return ""
    parts: list[str] = []
    for s in skills:
        payload = {
            "skill name": str(s.get("skill_name") or ""),
            "problem type": str(s.get("problem_type") or ""),
            "key insight": str(s.get("key_insight") or ""),
            "method": str(s.get("method") or ""),
        }
        parts.append(json.dumps(payload, ensure_ascii=False))
    return "\n\n---\n\n".join(parts)


def _serialize_skill_obj(skill: Any) -> dict[str, Any]:
    return {
        "id": getattr(skill, "id", ""),
        "skill_name": getattr(skill, "skill_name", ""),
        "problem_type": getattr(skill, "problem_type", ""),
        "key_insight": getattr(skill, "key_insight", ""),
        "method": getattr(skill, "method", ""),
    }


class SkillEvoAdapter:
    name = "skill_evo"

    def __init__(self, manager: Any, top_k: int) -> None:
        self.manager = manager
        self.top_k = max(1, int(top_k))

    def retrieve(self, question: str, extra_info: dict[str, Any]) -> dict[str, Any]:
        skills = self.manager.retrieve(question, top_k=self.top_k)
        return {"skills": [_serialize_skill_obj(s) for s in skills]}

    def render_skill_block(self, payload: dict[str, Any]) -> str:
        return _skills_block_minimal(list(payload.get("skills") or []))

    def augment_extra_info(self, extra_info: dict[str, Any], payload: dict[str, Any]) -> None:
        skills = list(payload.get("skills") or [])
        extra_info["skill_id"] = [str(s.get("id") or "") for s in skills]


class SkillRLAdapter:
    name = "skillrl"

    def __init__(
        self,
        bank: Any,
        *,
        retriever_url: str,
        mode: str,
        retrieve_lambda: float,
        top_k_general: int,
        top_k_task: int,
        top_k_mistake: int,
        prepare_candidates_fn: Any,
        retrieve_bucket_fn: Any,
        topic_slug_fn: Any,
    ) -> None:
        self.bank = bank
        self.retriever_url = retriever_url
        self.mode = mode
        self.retrieve_lambda = float(retrieve_lambda)
        self.top_k_general = max(0, int(top_k_general))
        self.top_k_task = max(0, int(top_k_task))
        self.top_k_mistake = max(0, int(top_k_mistake))
        self.prepare_candidates_fn = prepare_candidates_fn
        self.retrieve_bucket_fn = retrieve_bucket_fn
        self.topic_slug_fn = topic_slug_fn

    def retrieve(self, question: str, extra_info: dict[str, Any]) -> dict[str, Any]:
        topic = extra_info.get("topic")
        topic_key = self.topic_slug_fn(topic)
        general_candidates, task_candidates, mistake_candidates = self.prepare_candidates_fn(self.bank, topic_key)
        retrieved_general = self.retrieve_bucket_fn(
            question=question,
            candidates=general_candidates,
            top_k=self.top_k_general,
            retriever_url=self.retriever_url,
            mode=self.mode,
            retrieve_lambda=self.retrieve_lambda,
        )
        retrieved_task = self.retrieve_bucket_fn(
            question=question,
            candidates=task_candidates,
            top_k=self.top_k_task,
            retriever_url=self.retriever_url,
            mode=self.mode,
            retrieve_lambda=self.retrieve_lambda,
        )
        retrieved_mistakes = self.retrieve_bucket_fn(
            question=question,
            candidates=mistake_candidates,
            top_k=self.top_k_mistake,
            retriever_url=self.retriever_url,
            mode=self.mode,
            retrieve_lambda=self.retrieve_lambda,
        )
        return {
            "topic": topic,
            "topic_key": topic_key,
            "retrieval_mode": self.mode,
            "retriever_url": self.retriever_url,
            "general_candidates_count": len(general_candidates),
            "task_candidates_count": len(task_candidates),
            "mistake_candidates_count": len(mistake_candidates),
            "retrieved_general": list(retrieved_general),
            "retrieved_task": list(retrieved_task),
            "retrieved_mistakes": list(retrieved_mistakes),
        }

    def render_skill_block(self, payload: dict[str, Any]) -> str:
        from baselines.SkillRL.prepare_rl_data import format_skill_prompt

        return format_skill_prompt(
            topic_key=str(payload.get("topic_key") or "unknown"),
            general_skills=list(payload.get("retrieved_general") or []),
            task_skills=list(payload.get("retrieved_task") or []),
            mistakes=list(payload.get("retrieved_mistakes") or []),
        )

    def augment_extra_info(self, extra_info: dict[str, Any], payload: dict[str, Any]) -> None:
        retrieved_general = list(payload.get("retrieved_general") or [])
        retrieved_task = list(payload.get("retrieved_task") or [])
        retrieved_mistakes = list(payload.get("retrieved_mistakes") or [])
        extra_info["topic"] = payload.get("topic")
        extra_info["topic_key"] = payload.get("topic_key")
        extra_info["retrieval_mode"] = payload.get("retrieval_mode")
        extra_info["retriever_url"] = payload.get("retriever_url")
        extra_info["top_k_general"] = self.top_k_general
        extra_info["top_k_task"] = self.top_k_task
        extra_info["top_k_mistake"] = self.top_k_mistake
        extra_info["retrieved_general_skill_ids"] = [
            str(s.get("_retrieval_id") or s.get("skill_id") or "") for s in retrieved_general
        ]
        extra_info["retrieved_task_skill_ids"] = [
            str(s.get("_retrieval_id") or s.get("skill_id") or "") for s in retrieved_task
        ]
        extra_info["retrieved_common_mistake_ids"] = [str(s.get("_retrieval_id") or "") for s in retrieved_mistakes]
        extra_info["retrieved_general_count"] = len(retrieved_general)
        extra_info["retrieved_task_count"] = len(retrieved_task)
        extra_info["retrieved_mistake_count"] = len(retrieved_mistakes)
        extra_info["general_candidates_count"] = int(payload.get("general_candidates_count") or 0)
        extra_info["task_candidates_count"] = int(payload.get("task_candidates_count") or 0)
        extra_info["mistake_candidates_count"] = int(payload.get("mistake_candidates_count") or 0)
        extra_info["skill_id"] = (
            extra_info["retrieved_general_skill_ids"]
            + extra_info["retrieved_task_skill_ids"]
            + extra_info["retrieved_common_mistake_ids"]
        )


class AriseAdapter:
    name = "arise"

    def __init__(
        self,
        bank: Any,
        *,
        retriever_url: str,
        mode: str,
        retrieve_lambda: float,
        top_k: int,
    ) -> None:
        self.bank = bank
        self.retriever_url = retriever_url
        self.mode = mode
        self.retrieve_lambda = float(retrieve_lambda)
        self.top_k = max(1, int(top_k))

    def retrieve(self, question: str, extra_info: dict[str, Any]) -> dict[str, Any]:
        from baselines.SkillRL.prepare_rl_data import retrieve_bucket

        candidates = self.bank.build_candidates()
        retrieved_skills = retrieve_bucket(
            question=question,
            candidates=candidates,
            top_k=self.top_k,
            retriever_url=self.retriever_url,
            mode=self.mode,
            retrieve_lambda=self.retrieve_lambda,
        )
        return {
            "retrieval_mode": self.mode,
            "retriever_url": self.retriever_url,
            "skill_candidates_count": len(candidates),
            "retrieved_skills": list(retrieved_skills),
        }

    def render_skill_block(self, payload: dict[str, Any]) -> str:
        from baselines.ARISE.skill_bank import format_skill_prompt

        return format_skill_prompt(list(payload.get("retrieved_skills") or []))

    def augment_extra_info(self, extra_info: dict[str, Any], payload: dict[str, Any]) -> None:
        retrieved_skills = list(payload.get("retrieved_skills") or [])
        extra_info["retrieval_mode"] = payload.get("retrieval_mode")
        extra_info["retriever_url"] = payload.get("retriever_url")
        extra_info["top_k"] = self.top_k
        extra_info["skill_candidates_count"] = int(payload.get("skill_candidates_count") or 0)
        extra_info["retrieved_skill_count"] = len(retrieved_skills)
        extra_info["retrieved_skill_ids"] = [str(s.get("skill_id") or "") for s in retrieved_skills]
        extra_info["skill_id"] = list(extra_info["retrieved_skill_ids"])


def _process_one_file(
    input_path: Path,
    output_dir: Path,
    *,
    adapter: Any,
    controller_cls: Any,
    template_text: str,
    write_jsonl: bool,
    write_parquet: bool,
) -> tuple[int, int]:
    stem = input_path.stem
    out_jsonl = output_dir / f"{stem}_skill.jsonl"
    out_parquet = output_dir / f"{stem}_skill.parquet"

    kept_rows: list[dict[str, Any]] = []
    total = 0
    kept = 0

    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[prepare_test_data] {input_path.name} skip line {line_no}: {e}", file=sys.stderr)
                continue
            if not isinstance(d, dict):
                print(
                    f"[prepare_test_data] {input_path.name} skip line {line_no}: not a JSON object",
                    file=sys.stderr,
                )
                continue

            extra = d.get("extra_info")
            if not isinstance(extra, dict):
                print(
                    f"[prepare_test_data] {input_path.name} skip line {line_no}: missing extra_info",
                    file=sys.stderr,
                )
                continue
            q = extra.get("problem")
            if not isinstance(q, str) or not q.strip():
                print(
                    f"[prepare_test_data] {input_path.name} skip line {line_no}: missing extra_info['problem']",
                    file=sys.stderr,
                )
                continue
            q = q.strip()

            try:
                retrieved = adapter.retrieve(q, extra)
            except Exception as e:
                print(
                    f"[prepare_test_data] {input_path.name} skip line {line_no}: retrieve failed: {e}",
                    file=sys.stderr,
                )
                continue

            skill_block = adapter.render_skill_block(retrieved)
            try:
                new_content = template_text.format(skill=skill_block, question=q)
            except Exception as e:
                print(
                    f"[prepare_test_data] {input_path.name} skip line {line_no}: template failed: {e}",
                    file=sys.stderr,
                )
                continue

            rec = copy.deepcopy(d)
            pl = rec.get("prompt")
            if isinstance(pl, list):
                pl = [m for m in pl if not (isinstance(m, dict) and m.get("role") == "system")]
                rec["prompt"] = pl
            if not _apply_first_user_content(pl, new_content):
                print(
                    f"[prepare_test_data] {input_path.name} skip line {line_no}: no user message in prompt",
                    file=sys.stderr,
                )
                continue

            ex = rec.get("extra_info")
            if not isinstance(ex, dict):
                ex = {}
                rec["extra_info"] = ex
            adapter.augment_extra_info(ex, retrieved)

            kept_rows.append(rec)
            kept += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    if write_jsonl:
        with out_jsonl.open("w", encoding="utf-8") as f:
            for rec in kept_rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[prepare_test_data] wrote {out_jsonl} ({kept}/{total})")

    if write_parquet:
        rows_for_parquet = copy.deepcopy(kept_rows)
        for rec in rows_for_parquet:
            ex = rec.get("extra_info")
            if isinstance(ex, dict):
                controller_cls._sanitize_extra_info_for_parquet(ex)
            controller_cls._parquet_flatten_prompt_and_extra_info(rec)
            controller_cls._coerce_remaining_nested_to_json_strings(rec)
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("write_parquet requires pandas and pyarrow") from e
        pd.DataFrame(rows_for_parquet).to_parquet(out_parquet, index=False)
        print(f"[prepare_test_data] wrote {out_parquet} ({kept}/{total})")

    return kept, total


def _find_latest_memory_after_sol(mem_dir: Path) -> Path:
    best_n = -1
    best_path: Path | None = None
    for p in mem_dir.glob("memory_after_sol_v*.jsonl"):
        m = re.match(r"memory_after_sol_v(\d+)\.jsonl$", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n = n
            best_path = p
    if best_path is None:
        raise FileNotFoundError(f"no memory_after_sol_v*.jsonl found under: {mem_dir}")
    return best_path


def _resolve_skill_evo_memory_path(args: argparse.Namespace) -> Path:
    memory_jsonl = (args.memory_jsonl or "").strip()
    if memory_jsonl:
        return Path(memory_jsonl)
    mem_dir_raw = (args.memory_dir or "").strip()
    if not mem_dir_raw:
        raise ValueError("provide --memory-jsonl or --memory-dir")
    return _find_latest_memory_after_sol(Path(mem_dir_raw))


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    raw = (args.output_dir or "").strip()
    if raw:
        return Path(raw)
    if args.adapter == "skillrl":
        return DEFAULT_SKILLRL_OUTPUT_DIR
    if args.adapter == "arise":
        return DEFAULT_ARISE_OUTPUT_DIR
    raise ValueError("provide --output-dir when --adapter skill_evo")


def _build_adapter(args: argparse.Namespace) -> tuple[Any, Path]:
    if args.adapter == "skill_evo":
        from skill_manager.skill_manager import SkillManager

        mem_path = _resolve_skill_evo_memory_path(args)
        if not mem_path.is_file():
            raise FileNotFoundError(f"memory file not found: {mem_path}")
        manager = SkillManager(
            persist_path=mem_path,
            retriever_url=args.retriever_url,
            max_capacity=8192,
            retrieve_mode=args.skill_evo_mode,
        )
        manager.load_jsonl(mem_path)
        return SkillEvoAdapter(manager, top_k=args.top_k), mem_path

    if args.adapter == "arise":
        from baselines.ARISE.skill_bank import AriseSkillBank

        library_json = (args.library_json or "").strip()
        if not library_json:
            raise ValueError("provide --library-json when --adapter arise")
        library_path = Path(library_json)
        if not library_path.is_file():
            raise FileNotFoundError(f"library file not found: {library_path}")
        return (
            AriseAdapter(
                AriseSkillBank.from_path(library_path, include_reservoir=args.include_reservoir),
                retriever_url=args.retriever_url,
                mode=args.mode,
                retrieve_lambda=args.retrieve_lambda,
                top_k=args.top_k,
            ),
            library_path,
        )

    from baselines.SkillRL.layered_skill_bank import LayeredSkillBank
    from baselines.SkillRL.prepare_rl_data import _prepare_candidates, retrieve_bucket
    from baselines.SkillRL.text_utils import topic_slug

    skills_json = (args.skills_json or "").strip()
    if not skills_json:
        raise ValueError("provide --skills-json when --adapter skillrl")
    skills_path = Path(skills_json)
    if not skills_path.is_file():
        raise FileNotFoundError(f"skills file not found: {skills_path}")

    return (
        SkillRLAdapter(
            LayeredSkillBank.from_path(str(skills_path)),
            retriever_url=args.retriever_url,
            mode=args.mode,
            retrieve_lambda=args.retrieve_lambda,
            top_k_general=args.top_k_general if args.top_k_general is not None else args.top_k,
            top_k_task=args.top_k_task if args.top_k_task is not None else args.top_k,
            top_k_mistake=args.top_k_mistake,
            prepare_candidates_fn=_prepare_candidates,
            retrieve_bucket_fn=retrieve_bucket,
            topic_slug_fn=topic_slug,
        ),
        skills_path,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="skill_evo", choices=["skill_evo", "skillrl", "arise"])
    p.add_argument("--memory-jsonl", default="", help="Path to memory_after_sol_vN.jsonl")
    p.add_argument("--memory-dir", default="", help="Directory to auto-find latest memory_after_sol_vN.jsonl")
    p.add_argument("--skills-json", default="", help="Path to SkillRL claude-style skill bank json")
    p.add_argument("--library-json", default="", help="Path to ARISE skill library checkpoint json")
    p.add_argument("--inputs", nargs="+", required=True, help="Input jsonl files")
    p.add_argument("--output-dir", default="", help="Output directory")
    p.add_argument("--top-k", type=int, default=3, help="Retriever top_k; also used as skillrl general/task default")
    p.add_argument("--top-k-general", type=int, default=None, help="SkillRL general skill top_k")
    p.add_argument("--top-k-task", type=int, default=None, help="SkillRL task skill top_k")
    p.add_argument("--top-k-mistake", type=int, default=2, help="SkillRL mistake top_k")
    p.add_argument(
        "--retriever-url",
        default="http://127.0.0.1:8766",
        help="Retriever url for skill_evo / skillrl / arise",
    )
    p.add_argument(
        "--skill-evo-mode",
        default="hybrid",
        choices=["embedding", "hybrid"],
        help="SkillEvo retriever mode; default hybrid to preserve existing behavior",
    )
    p.add_argument("--mode", default="embedding", choices=["embedding", "hybrid"], help="SkillRL retriever mode")
    p.add_argument("--retrieve-lambda", type=float, default=0.5, help="SkillRL hybrid retrieve lambda")
    p.add_argument("--include-reservoir", action="store_true", help="ARISE: also retrieve from reservoir")
    p.add_argument("--write-jsonl", action="store_true", help="Write *_skill.jsonl")
    p.add_argument("--write-parquet", action="store_true", help="Write *_skill.parquet")
    args = p.parse_args()

    _add_skill_src_to_path()
    from skill_manager.skill_controller import SkillController

    if not args.write_jsonl and not args.write_parquet:
        print("[prepare_test_data] neither --write-jsonl nor --write-parquet is set", file=sys.stderr)
        return 2

    try:
        adapter, source_path = _build_adapter(args)
        output_dir = _resolve_output_dir(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"[prepare_test_data] {e}", file=sys.stderr)
        return 2 if isinstance(e, ValueError) else 1

    input_paths = [Path(x) for x in args.inputs]
    for ip in input_paths:
        if not ip.is_file():
            print(f"[prepare_test_data] input file not found: {ip}", file=sys.stderr)
            return 1

    template_path = Path(__file__).resolve().parent / "prompt" / "skill_use_v1.txt"
    template_text = _load_template(template_path)

    total_kept = 0
    total_seen = 0
    for ip in input_paths:
        kept, seen = _process_one_file(
            ip,
            output_dir,
            adapter=adapter,
            controller_cls=SkillController,
            template_text=template_text,
            write_jsonl=args.write_jsonl,
            write_parquet=args.write_parquet,
        )
        total_kept += kept
        total_seen += seen

    print(
        f"[prepare_test_data] done. adapter={args.adapter} source={source_path} inputs={len(input_paths)} "
        f"rows={total_kept}/{total_seen} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
