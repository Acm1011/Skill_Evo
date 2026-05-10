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


def _skills_block_minimal(skills: list[Any]) -> str:
    """Only keep 4 fields for prompt injection."""
    if not skills:
        return ""
    parts: list[str] = []
    for s in skills:
        payload = {
            "skill name": getattr(s, "skill_name", ""),
            "problem type": getattr(s, "problem_type", ""),
            "key insight": getattr(s, "key_insight", ""),
            "method": getattr(s, "method", ""),
        }
        parts.append(json.dumps(payload, ensure_ascii=False))
    return "\n\n---\n\n".join(parts)


def _process_one_file(
    input_path: Path,
    output_dir: Path,
    *,
    manager: Any,
    controller_cls: Any,
    template_text: str,
    top_k: int,
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
                skills = manager.retrieve(q, top_k=top_k)
            except Exception as e:
                print(
                    f"[prepare_test_data] {input_path.name} skip line {line_no}: retrieve failed: {e}",
                    file=sys.stderr,
                )
                continue

            skill_block = _skills_block_minimal(skills)
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
            ex["skill_id"] = [s.id for s in skills]

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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-jsonl", default="", help="Path to memory_after_sol_vN.jsonl")
    p.add_argument("--memory-dir", default="", help="Directory to auto-find latest memory_after_sol_vN.jsonl")
    p.add_argument("--inputs", nargs="+", required=True, help="Input jsonl files")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--top-k", type=int, default=3, help="Retriever top_k")
    p.add_argument("--write-jsonl", action="store_true", help="Write *_skill.jsonl")
    p.add_argument("--write-parquet", action="store_true", help="Write *_skill.parquet")
    args = p.parse_args()

    _add_skill_src_to_path()
    from skill_manager.skill_controller import SkillController
    from skill_manager.skill_manager import DEFAULT_RETRIEVER_URL, SkillManager

    if not args.write_jsonl and not args.write_parquet:
        print("[prepare_test_data] neither --write-jsonl nor --write-parquet is set", file=sys.stderr)
        return 2

    memory_jsonl = (args.memory_jsonl or "").strip()
    if memory_jsonl:
        mem_path = Path(memory_jsonl)
    else:
        mem_dir_raw = (args.memory_dir or "").strip()
        if not mem_dir_raw:
            print("[prepare_test_data] provide --memory-jsonl or --memory-dir", file=sys.stderr)
            return 2
        mem_path = _find_latest_memory_after_sol(Path(mem_dir_raw))
    if not mem_path.is_file():
        print(f"[prepare_test_data] memory file not found: {mem_path}", file=sys.stderr)
        return 1

    input_paths = [Path(x) for x in args.inputs]
    for ip in input_paths:
        if not ip.is_file():
            print(f"[prepare_test_data] input file not found: {ip}", file=sys.stderr)
            return 1
    output_dir = Path(args.output_dir)

    template_path = Path(__file__).resolve().parent / "prompt" / "skill_use_v1.txt"
    template_text = _load_template(template_path)

    manager = SkillManager(
        persist_path=mem_path,
        retriever_url=DEFAULT_RETRIEVER_URL,
        max_capacity=8192,
    )
    manager.load_jsonl(mem_path)

    total_kept = 0
    total_seen = 0
    for ip in input_paths:
        kept, seen = _process_one_file(
            ip,
            output_dir,
            manager=manager,
            controller_cls=SkillController,
            template_text=template_text,
            top_k=max(1, int(args.top_k)),
            write_jsonl=args.write_jsonl,
            write_parquet=args.write_parquet,
        )
        total_kept += kept
        total_seen += seen
    print(
        f"[prepare_test_data] done. memory={mem_path} inputs={len(input_paths)} "
        f"rows={total_kept}/{total_seen} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
