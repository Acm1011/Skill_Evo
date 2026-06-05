#!/usr/bin/env python3
"""Prepare skill-augmented MMLU-Pro and SuperGPQA data for baseline memories."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_template() -> str:
    path = REPO_ROOT / "skill_src" / "prompt" / "skill_use_v1.txt"
    with path.open(encoding="utf-8") as f:
        return f.read()


def _copy_support_files(src_root: Path, dst_root: Path) -> None:
    datasets = ["MMLU-Pro", "SuperGPQA"]
    for dataset_name in datasets:
        src_dir = src_root / dataset_name
        dst_dir = dst_root / dataset_name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _mmlu_topic(row: dict[str, Any]) -> str:
    parts = ["MMLU-Pro"]
    category = str(row.get("category") or "").strip()
    src = str(row.get("src") or "").strip()
    if category:
        parts.append(category)
    if src:
        parts.append(src)
    return " -> ".join(parts)


def _supergpqa_topic(row: dict[str, Any]) -> str:
    parts = ["SuperGPQA"]
    for key in ("discipline", "field", "subfield"):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(value)
    return " -> ".join(parts)


@dataclass
class RetrievalResult:
    skill_block: str
    retrieved_ids: list[str]


class BaseAdapter:
    source_name: str
    source_path: Path

    @property
    def memory_label(self) -> str:
        return f"{self.source_name}:{self.source_path.stem}"

    def retrieve(self, question: str, topic: str) -> RetrievalResult:
        raise NotImplementedError


class ExpeLMathAdapter(BaseAdapter):
    source_name = "ExpeLMath"

    def __init__(
        self,
        *,
        memory_bank: Path,
        embeddings: Path,
        top_k: int,
        backend: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        hash_dim: int,
        topic_bonus: float,
    ) -> None:
        from baselines.ExpeLMath.retrieve_memory import format_retrieved_prompt, retrieve_records
        from baselines.ReasoningBankMath.io_utils import read_jsonl

        self.source_path = memory_bank
        self.embeddings_path = embeddings
        self.top_k = max(1, int(top_k))
        self.backend = backend
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.hash_dim = int(hash_dim)
        self.topic_bonus = float(topic_bonus)
        self.memory_rows = read_jsonl(memory_bank)
        self.embedding_rows = read_jsonl(embeddings)
        self._retrieve_records = retrieve_records
        self._format_retrieved_prompt = format_retrieved_prompt

    def retrieve(self, question: str, topic: str) -> RetrievalResult:
        scored = self._retrieve_records(
            question=question,
            memory_rows=self.memory_rows,
            embedding_rows=self.embedding_rows,
            top_k=self.top_k,
            backend=self.backend,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout,
            hash_dim=self.hash_dim,
            query_topic=topic,
            topic_bonus=self.topic_bonus,
        )
        rows = [row for row, _score in scored]
        return RetrievalResult(
            skill_block=self._format_retrieved_prompt(rows),
            retrieved_ids=[str(row.get("memory_id") or "") for row in rows],
        )


class MementoMathAdapter(BaseAdapter):
    source_name = "MementoMath"

    def __init__(
        self,
        *,
        memory_bank: Path,
        embeddings: Path,
        top_k: int,
        backend: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        hash_dim: int,
        topic_bonus: float,
        same_status_bonus: float,
    ) -> None:
        from baselines.MementoMath.io_utils import read_jsonl
        from baselines.MementoMath.retrieve_memory import format_retrieved_prompt, retrieve_records

        self.source_path = memory_bank
        self.embeddings_path = embeddings
        self.top_k = max(1, int(top_k))
        self.backend = backend
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.hash_dim = int(hash_dim)
        self.topic_bonus = float(topic_bonus)
        self.same_status_bonus = float(same_status_bonus)
        self.memory_rows = read_jsonl(memory_bank)
        self.embedding_rows = read_jsonl(embeddings)
        self._retrieve_records = retrieve_records
        self._format_retrieved_prompt = format_retrieved_prompt

    def retrieve(self, question: str, topic: str) -> RetrievalResult:
        scored = self._retrieve_records(
            question=question,
            memory_rows=self.memory_rows,
            embedding_rows=self.embedding_rows,
            top_k=self.top_k,
            backend=self.backend,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout,
            hash_dim=self.hash_dim,
            query_topic=topic,
            topic_bonus=self.topic_bonus,
            same_status_bonus=self.same_status_bonus,
            status="",
        )
        rows = [row for row, _score in scored]
        return RetrievalResult(
            skill_block=self._format_retrieved_prompt(rows),
            retrieved_ids=[str(row.get("memory_id") or "") for row in rows],
        )


class ReasoningBankMathAdapter(BaseAdapter):
    source_name = "ReasoningBankMath"

    def __init__(
        self,
        *,
        memory_bank: Path,
        top_k: int,
        retriever_url: str,
        mode: str,
        retrieve_lambda: float,
    ) -> None:
        from baselines.ReasoningBankMath.prepare_prompt_data import (
            build_memory_candidates,
            format_memory_prompt,
            retrieve_memories,
        )
        from baselines.ReasoningBankMath.io_utils import read_jsonl
        from baselines.ReasoningBankMath.text_utils import topic_slug

        self.source_path = memory_bank
        self.top_k = max(1, int(top_k))
        self.retriever_url = retriever_url
        self.mode = mode
        self.retrieve_lambda = float(retrieve_lambda)
        self.topic_slug = topic_slug
        self.memory_rows = read_jsonl(memory_bank)
        self.candidates = build_memory_candidates(self.memory_rows)
        self._retrieve_memories = retrieve_memories
        self._format_memory_prompt = format_memory_prompt

    def retrieve(self, question: str, topic: str) -> RetrievalResult:
        rows = self._retrieve_memories(
            question=question,
            candidates=self.candidates,
            top_k=self.top_k,
            retriever_url=self.retriever_url,
            mode=self.mode,
            retrieve_lambda=self.retrieve_lambda,
            topic_key=self.topic_slug(topic),
        )
        return RetrievalResult(
            skill_block=self._format_memory_prompt(rows),
            retrieved_ids=[str(row.get("memory_id") or "") for row in rows],
        )


class SkillRLAdapter(BaseAdapter):
    source_name = "SkillRL"

    def __init__(
        self,
        *,
        skills_json: Path,
        retriever_url: str,
        mode: str,
        retrieve_lambda: float,
        top_k_general: int,
        top_k_task: int,
        top_k_mistake: int,
    ) -> None:
        from baselines.SkillRL.layered_skill_bank import LayeredSkillBank
        from baselines.SkillRL.prepare_rl_data import _prepare_candidates, format_skill_prompt, retrieve_bucket
        from baselines.SkillRL.text_utils import topic_slug

        self.source_path = skills_json
        self.retriever_url = retriever_url
        self.mode = mode
        self.retrieve_lambda = float(retrieve_lambda)
        self.top_k_general = max(0, int(top_k_general))
        self.top_k_task = max(0, int(top_k_task))
        self.top_k_mistake = max(0, int(top_k_mistake))
        self.topic_slug = topic_slug
        self.bank = LayeredSkillBank.from_path(str(skills_json))
        self._prepare_candidates = _prepare_candidates
        self._retrieve_bucket = retrieve_bucket
        self._format_skill_prompt = format_skill_prompt

    def retrieve(self, question: str, topic: str) -> RetrievalResult:
        topic_key = self.topic_slug(topic)
        general_candidates, task_candidates, mistake_candidates = self._prepare_candidates(self.bank, topic_key)
        retrieved_general = self._retrieve_bucket(
            question=question,
            candidates=general_candidates,
            top_k=self.top_k_general,
            retriever_url=self.retriever_url,
            mode=self.mode,
            retrieve_lambda=self.retrieve_lambda,
        )
        retrieved_task = self._retrieve_bucket(
            question=question,
            candidates=task_candidates,
            top_k=self.top_k_task,
            retriever_url=self.retriever_url,
            mode=self.mode,
            retrieve_lambda=self.retrieve_lambda,
        )
        retrieved_mistakes = self._retrieve_bucket(
            question=question,
            candidates=mistake_candidates,
            top_k=self.top_k_mistake,
            retriever_url=self.retriever_url,
            mode=self.mode,
            retrieve_lambda=self.retrieve_lambda,
        )
        retrieved_ids = (
            [str(s.get("_retrieval_id") or s.get("skill_id") or "") for s in retrieved_general]
            + [str(s.get("_retrieval_id") or s.get("skill_id") or "") for s in retrieved_task]
            + [str(s.get("_retrieval_id") or "") for s in retrieved_mistakes]
        )
        return RetrievalResult(
            skill_block=self._format_skill_prompt(
                topic_key=topic_key,
                general_skills=retrieved_general,
                task_skills=retrieved_task,
                mistakes=retrieved_mistakes,
            ),
            retrieved_ids=retrieved_ids,
        )


def _build_expel_adapter(args: argparse.Namespace) -> ExpeLMathAdapter:
    memory_bank = Path(args.expel_memory_bank) if args.expel_memory_bank else None
    if memory_bank is None:
        memory_bank = _first_existing(
            [
                REPO_ROOT / "baselines" / "ExpeLMath" / "outputs" / "memory_bank_v2.jsonl",
                REPO_ROOT / "baselines" / "ExpeLMath" / "outputs" / "memory_bank.jsonl",
            ]
        )
    embeddings = Path(args.expel_embeddings) if args.expel_embeddings else None
    if embeddings is None:
        embeddings = _first_existing(
            [
                REPO_ROOT / "baselines" / "ExpeLMath" / "outputs" / "memory_embeddings_v2.jsonl",
                REPO_ROOT / "baselines" / "ExpeLMath" / "outputs" / "memory_embeddings.jsonl",
            ]
        )
    if memory_bank is None or not memory_bank.is_file():
        raise FileNotFoundError("ExpeLMath memory bank not found; provide --expel-memory-bank")
    if embeddings is None or not embeddings.is_file():
        raise FileNotFoundError("ExpeLMath embeddings not found; provide --expel-embeddings")
    return ExpeLMathAdapter(
        memory_bank=memory_bank,
        embeddings=embeddings,
        top_k=args.top_k,
        backend=args.expel_backend,
        base_url=args.expel_embed_base_url,
        api_key=args.expel_embed_api_key,
        model=args.expel_embed_model,
        timeout=args.expel_timeout,
        hash_dim=args.expel_hash_dim,
        topic_bonus=args.expel_topic_bonus,
    )


def _build_rbm_adapter(args: argparse.Namespace) -> ReasoningBankMathAdapter:
    memory_bank = Path(args.rbm_memory_bank) if args.rbm_memory_bank else None
    if memory_bank is None:
        memory_bank = _first_existing(
            [
                REPO_ROOT / "baselines" / "ReasoningBankMath" / "outputs" / "memory_bank_v1_v2_refined.jsonl",
                REPO_ROOT / "baselines" / "ReasoningBankMath" / "outputs" / "memory_bank_refined.jsonl",
                REPO_ROOT / "baselines" / "ReasoningBankMath" / "outputs" / "memory_bank_v1_v2_compact.jsonl",
                REPO_ROOT / "baselines" / "ReasoningBankMath" / "outputs" / "memory_bank_v1_v2.jsonl",
                REPO_ROOT / "baselines" / "ReasoningBankMath" / "outputs" / "memory_bank.jsonl",
            ]
        )
    if memory_bank is None or not memory_bank.is_file():
        raise FileNotFoundError("ReasoningBankMath memory bank not found; provide --rbm-memory-bank")
    return ReasoningBankMathAdapter(
        memory_bank=memory_bank,
        top_k=args.top_k,
        retriever_url=args.retriever_url,
        mode=args.mode,
        retrieve_lambda=args.retrieve_lambda,
    )


def _build_memento_adapter(args: argparse.Namespace) -> MementoMathAdapter:
    memory_bank = Path(args.memento_memory_bank) if args.memento_memory_bank else None
    if memory_bank is None:
        memory_bank = _first_existing(
            [
                REPO_ROOT / "baselines" / "MementoMath" / "outputs" / "memory_bank_v2.jsonl",
                REPO_ROOT / "baselines" / "MementoMath" / "outputs" / "memory_bank_v1_v2.jsonl",
                REPO_ROOT / "baselines" / "MementoMath" / "outputs" / "memory_bank.jsonl",
            ]
        )
    embeddings = Path(args.memento_embeddings) if args.memento_embeddings else None
    if embeddings is None:
        embeddings = _first_existing(
            [
                REPO_ROOT / "baselines" / "MementoMath" / "outputs" / "memory_embeddings_v2.jsonl",
                REPO_ROOT / "baselines" / "MementoMath" / "outputs" / "memory_embeddings_v1_v2.jsonl",
                REPO_ROOT / "baselines" / "MementoMath" / "outputs" / "memory_embeddings.jsonl",
            ]
        )
    if memory_bank is None or not memory_bank.is_file():
        raise FileNotFoundError("MementoMath memory bank not found; provide --memento-memory-bank")
    if embeddings is None or not embeddings.is_file():
        raise FileNotFoundError("MementoMath embeddings not found; provide --memento-embeddings")
    return MementoMathAdapter(
        memory_bank=memory_bank,
        embeddings=embeddings,
        top_k=args.top_k,
        backend=args.memento_backend,
        base_url=args.memento_embed_base_url,
        api_key=args.memento_embed_api_key,
        model=args.memento_embed_model,
        timeout=args.memento_timeout,
        hash_dim=args.memento_hash_dim,
        topic_bonus=args.memento_topic_bonus,
        same_status_bonus=args.memento_same_status_bonus,
    )


def _build_skillrl_adapter(args: argparse.Namespace) -> SkillRLAdapter:
    skills_json = Path(args.skillrl_skills_json) if args.skillrl_skills_json else None
    if skills_json is None:
        skills_json = _first_existing(
            [
                REPO_ROOT / "baselines" / "SkillRL" / "outputs" / "skills_from_rollout_teacher.json",
            ]
        )
    if skills_json is None or not skills_json.is_file():
        raise FileNotFoundError("SkillRL skills json not found; provide --skillrl-skills-json")
    return SkillRLAdapter(
        skills_json=skills_json,
        retriever_url=args.retriever_url,
        mode=args.mode,
        retrieve_lambda=args.retrieve_lambda,
        top_k_general=args.top_k_general,
        top_k_task=args.top_k_task,
        top_k_mistake=args.top_k_mistake,
    )


def _build_adapter(source: str, args: argparse.Namespace) -> BaseAdapter:
    key = source.lower()
    if key == "expelmath":
        return _build_expel_adapter(args)
    if key == "mementomath":
        return _build_memento_adapter(args)
    if key == "reasoningbankmath":
        return _build_rbm_adapter(args)
    if key == "skillrl":
        return _build_skillrl_adapter(args)
    raise ValueError(f"unsupported source: {source}")


def _render_question(template_text: str, question: str, result: RetrievalResult) -> str:
    return template_text.format(skill=result.skill_block, question=question)


def _prepare_mmlupro(
    *,
    adapter: BaseAdapter,
    template_text: str,
    src_root: Path,
    dst_root: Path,
) -> None:
    in_path = src_root / "MMLU-Pro" / "data" / "test-00000-of-00001.parquet"
    out_path = dst_root / "MMLU-Pro" / "data" / "test-00000-of-00001.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    rendered_questions: list[str] = []
    memory_sources: list[str] = []
    memory_labels: list[str] = []
    memory_paths: list[str] = []
    benchmark_topics: list[str] = []
    retrieved_ids_json: list[str] = []
    retrieved_counts: list[int] = []

    for row in df.to_dict(orient="records"):
        question = str(row["question"])
        topic = _mmlu_topic(row)
        result = adapter.retrieve(question, topic)
        rendered_questions.append(_render_question(template_text, question, result))
        memory_sources.append(adapter.source_name)
        memory_labels.append(adapter.memory_label)
        memory_paths.append(str(adapter.source_path))
        benchmark_topics.append(topic)
        retrieved_ids_json.append(_json_dumps(result.retrieved_ids))
        retrieved_counts.append(len(result.retrieved_ids))

    df["question"] = rendered_questions
    df["memory_source"] = memory_sources
    df["memory_label"] = memory_labels
    df["memory_path"] = memory_paths
    df["benchmark_topic"] = benchmark_topics
    df["retrieved_ids_json"] = retrieved_ids_json
    df["retrieved_count"] = retrieved_counts
    df.to_parquet(out_path, index=False)


def _prepare_supergpqa(
    *,
    adapter: BaseAdapter,
    template_text: str,
    src_root: Path,
    dst_root: Path,
) -> None:
    in_path = src_root / "SuperGPQA" / "SuperGPQA-all.jsonl"
    out_path = dst_root / "SuperGPQA" / "SuperGPQA-all.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = str(item["question"])
            topic = _supergpqa_topic(item)
            result = adapter.retrieve(question, topic)
            item["question"] = _render_question(template_text, question, result)
            item["memory_source"] = adapter.source_name
            item["memory_label"] = adapter.memory_label
            item["memory_path"] = str(adapter.source_path)
            item["benchmark_topic"] = topic
            item["retrieved_ids"] = result.retrieved_ids
            item["retrieved_count"] = len(result.retrieved_ids)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def _prepare_for_source(
    *,
    source: str,
    adapter: BaseAdapter,
    source_data_root: Path,
    output_root: Path,
    template_text: str,
) -> None:
    dst_root = output_root / adapter.source_name
    _copy_support_files(source_data_root, dst_root)
    _prepare_mmlupro(adapter=adapter, template_text=template_text, src_root=source_data_root, dst_root=dst_root)
    _prepare_supergpqa(adapter=adapter, template_text=template_text, src_root=source_data_root, dst_root=dst_root)
    print(
        f"[prepare_general_benchmark_data] done source={source} memory={adapter.source_path} output={dst_root}",
        file=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sources",
        nargs="+",
        action="append",
        default=None,
        choices=["ExpeLMath", "MementoMath", "ReasoningBankMath", "SkillRL"],
        help="Which memory sources to prepare",
    )
    p.add_argument(
        "--source-data-root",
        default="/home/ycy/sdi/data",
        help="Root dir containing MMLU-Pro/ and SuperGPQA/",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "baselines" / "outputs" / "general_benchmark_data"),
        help="Output root dir. Each source writes to a separate subdir.",
    )

    p.add_argument("--top-k", type=int, default=5, help="ExpeLMath / ReasoningBankMath top-k")
    p.add_argument("--retriever-url", default="http://127.0.0.1:8766", help="Retriever URL for RBM/SkillRL")
    p.add_argument("--mode", default="embedding", choices=["embedding", "hybrid"])
    p.add_argument("--retrieve-lambda", type=float, default=0.5)
    p.add_argument("--top-k-general", type=int, default=5)
    p.add_argument("--top-k-task", type=int, default=5)
    p.add_argument("--top-k-mistake", type=int, default=2)

    p.add_argument("--expel-memory-bank", default="")
    p.add_argument("--expel-embeddings", default="")
    p.add_argument("--expel-backend", default="hash", choices=["hash", "openai"])
    p.add_argument("--expel-embed-base-url", default="")
    p.add_argument("--expel-embed-api-key", default="")
    p.add_argument("--expel-embed-model", default="")
    p.add_argument("--expel-timeout", type=float, default=600.0)
    p.add_argument("--expel-hash-dim", type=int, default=256)
    p.add_argument("--expel-topic-bonus", type=float, default=0.05)

    p.add_argument("--memento-memory-bank", default="")
    p.add_argument("--memento-embeddings", default="")
    p.add_argument("--memento-backend", default="hash", choices=["hash", "openai"])
    p.add_argument("--memento-embed-base-url", default="")
    p.add_argument("--memento-embed-api-key", default="")
    p.add_argument("--memento-embed-model", default="")
    p.add_argument("--memento-timeout", type=float, default=600.0)
    p.add_argument("--memento-hash-dim", type=int, default=256)
    p.add_argument("--memento-topic-bonus", type=float, default=0.05)
    p.add_argument("--memento-same-status-bonus", type=float, default=0.02)

    p.add_argument("--rbm-memory-bank", default="")
    p.add_argument("--skillrl-skills-json", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.sources:
        args.sources = [item for group in args.sources for item in group]
    else:
        args.sources = ["ExpeLMath", "MementoMath", "ReasoningBankMath", "SkillRL"]
    source_data_root = Path(args.source_data_root)
    if not (source_data_root / "MMLU-Pro" / "data" / "test-00000-of-00001.parquet").is_file():
        print("[prepare_general_benchmark_data] missing MMLU-Pro test parquet", file=sys.stderr)
        return 1
    if not (source_data_root / "SuperGPQA" / "SuperGPQA-all.jsonl").is_file():
        print("[prepare_general_benchmark_data] missing SuperGPQA jsonl", file=sys.stderr)
        return 1

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    template_text = _load_template()

    for source in args.sources:
        try:
            adapter = _build_adapter(source, args)
        except (FileNotFoundError, ValueError) as e:
            print(f"[prepare_general_benchmark_data] {e}", file=sys.stderr)
            return 1
        _prepare_for_source(
            source=source,
            adapter=adapter,
            source_data_root=source_data_root,
            output_root=output_root,
            template_text=template_text,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
