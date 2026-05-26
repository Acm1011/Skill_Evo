#!/usr/bin/env python3
"""Prepare skill-augmented general benchmark data while preserving original folder layout."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _add_skill_src_to_path() -> None:
    root = Path(__file__).resolve().parent
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_template(template_path: Path) -> str:
    with template_path.open(encoding="utf-8") as f:
        return f.read()


def _find_latest_memory_after_sol(mem_dir: Path) -> Path:
    best_n = -1
    best_path: Path | None = None
    for p in mem_dir.glob("memory_after_sol_v*.jsonl"):
        name = p.name
        if not name.startswith("memory_after_sol_v") or not name.endswith(".jsonl"):
            continue
        raw_n = name[len("memory_after_sol_v") : -len(".jsonl")]
        if not raw_n.isdigit():
            continue
        n = int(raw_n)
        if n > best_n:
            best_n = n
            best_path = p
    if best_path is None:
        raise FileNotFoundError(f"no memory_after_sol_v*.jsonl found under: {mem_dir}")
    return best_path


def _resolve_memory_path(args: argparse.Namespace) -> Path:
    if args.memory_jsonl:
        return Path(args.memory_jsonl)
    if args.memory_dir:
        return _find_latest_memory_after_sol(Path(args.memory_dir))
    raise ValueError("provide --memory-jsonl or --memory-dir")


def _build_manager(memory_path: Path):
    from skill_manager.skill_manager import DEFAULT_RETRIEVER_URL, SkillManager

    manager = SkillManager(
        persist_path=memory_path,
        retriever_url=DEFAULT_RETRIEVER_URL,
        max_capacity=8192,
    )
    manager.load_jsonl(memory_path)
    return manager


def _inject_question(
    manager: Any,
    template_text: str,
    question: str,
    *,
    top_k: int,
) -> tuple[str, list[str]]:
    from skill_manager.skill_manager import SkillManager

    skills = manager.retrieve(question, top_k=top_k)
    skill_block = SkillManager.skills_block_for_template(skills)
    rendered = template_text.format(skill=skill_block, question=question)
    skill_ids = [str(getattr(s, "id", "")) for s in skills]
    return rendered, skill_ids


def _prepare_bbeh(
    manager: Any,
    template_text: str,
    src_dir: Path,
    dst_dir: Path,
    *,
    top_k: int,
) -> None:
    in_path = src_dir / "bbeh-eval" / "train.jsonl"
    out_path = dst_dir / "bbeh-eval" / "train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = str(item["question"])
            new_question, _skill_ids = _inject_question(manager, template_text, question, top_k=top_k)
            item["question"] = new_question
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def _prepare_mmlupro(
    manager: Any,
    template_text: str,
    src_dir: Path,
    dst_dir: Path,
    *,
    top_k: int,
) -> None:
    in_path = src_dir / "MMLU-Pro" / "data" / "test-00000-of-00001.parquet"
    out_path = dst_dir / "MMLU-Pro" / "data" / "test-00000-of-00001.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    new_questions = []
    for question in df["question"].tolist():
        question = str(question)
        new_question, _skill_ids = _inject_question(manager, template_text, question, top_k=top_k)
        new_questions.append(new_question)
    df["question"] = new_questions
    df.to_parquet(out_path, index=False)


def _prepare_supergpqa(
    manager: Any,
    template_text: str,
    src_dir: Path,
    dst_dir: Path,
    *,
    top_k: int,
) -> None:
    in_path = src_dir / "SuperGPQA" / "SuperGPQA-all.jsonl"
    out_path = dst_dir / "SuperGPQA" / "SuperGPQA-all.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = str(item["question"])
            new_question, _skill_ids = _inject_question(manager, template_text, question, top_k=top_k)
            item["question"] = new_question
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def _copy_support_files(src_root: Path, dst_root: Path) -> None:
    datasets = ["bbeh-eval", "MMLU-Pro", "SuperGPQA"]
    for dataset_name in datasets:
        src_dir = src_root / dataset_name
        dst_dir = dst_root / dataset_name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-jsonl", default="", help="Path to memory_after_sol_vN.jsonl")
    p.add_argument("--memory-dir", default="", help="Directory to auto-find latest memory_after_sol_vN.jsonl")
    p.add_argument("--source-data-dir", required=True, help="Directory containing raw bbeh-eval/MMLU-Pro/SuperGPQA")
    p.add_argument("--output-dir", required=True, help="Root output directory preserving benchmark subdirs")
    p.add_argument("--top-k", type=int, default=5, help="Retriever top-k")
    args = p.parse_args()

    _add_skill_src_to_path()

    source_data_dir = Path(args.source_data_dir)
    output_dir = Path(args.output_dir)
    memory_path = _resolve_memory_path(args)
    if not memory_path.is_file():
        raise FileNotFoundError(f"memory file not found: {memory_path}")

    template_path = Path(__file__).resolve().parent / "prompt" / "skill_use_v1.txt"
    template_text = _load_template(template_path)

    _copy_support_files(source_data_dir, output_dir)
    manager = _build_manager(memory_path)

    _prepare_bbeh(manager, template_text, source_data_dir, output_dir, top_k=args.top_k)
    _prepare_mmlupro(manager, template_text, source_data_dir, output_dir, top_k=args.top_k)
    _prepare_supergpqa(manager, template_text, source_data_dir, output_dir, top_k=args.top_k)

    print(
        f"[prepare_general_test_data] done. memory={memory_path} "
        f"source_data_dir={source_data_dir} output_dir={output_dir} top_k={args.top_k}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
