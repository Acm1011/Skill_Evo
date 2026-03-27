#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""与 rollout 相同的数据与写盘格式；user prompt 使用技能：direct=按 id 匹配；retrieve=按 problem type 嵌入检索 top-k。"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from prompts import prompt_use_skill

import rollout as roll_r

SKILL_FIELDS = ("skill name", "problem type", "key insight", "method")


def _normalize_embedding_device(device: Optional[str]) -> Optional[str]:
    """
    sentence-transformers / torch 需要合法 device 字符串。
    仅数字时视为 CUDA 卡号，例如 \"2\" -> \"cuda:2\"（避免 Invalid device string: '2'）。
    """
    if device is None:
        return None
    s = str(device).strip()
    if not s:
        return None
    if s.isdigit():
        return f"cuda:{s}"
    return s


def _load_skills_from_jsonl(
    path: Path,
) -> tuple[dict[Any, dict], list[dict], int]:
    """by_id 索引；corpus 为按文件顺序去重后的技能列表（供检索）；n_lines 为含 id 的行数。"""
    by_id: dict[Any, dict] = {}
    corpus: list[dict] = []
    seen_ids: set[Any] = set()
    n_lines = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[solver] skip skills line {line_no}: {e}", file=sys.stderr)
                continue
            sid = rec.get("id")
            if sid is None:
                continue
            n_lines += 1
            by_id[sid] = rec
            by_id[str(sid)] = rec
            if sid in seen_ids or str(sid) in seen_ids:
                continue
            seen_ids.add(sid)
            seen_ids.add(str(sid))
            corpus.append(rec)
    return by_id, corpus, n_lines


def _skill_retrieval_text(skill_rec: dict) -> str:
    pt = skill_rec.get("problem type")
    if pt is not None and str(pt).strip():
        return str(pt).strip()
    sn = skill_rec.get("skill name")
    if sn is not None and str(sn).strip():
        return str(sn).strip()
    return ""


def _format_skill_block(skill_rec: dict) -> str:
    lines = []
    for k in SKILL_FIELDS:
        v = skill_rec.get(k)
        if v is not None and str(v).strip() != "":
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _format_retrieved_skills_block(skills: list[dict], scores: list[float]) -> str:
    parts = []
    for rank, (sk, sc) in enumerate(zip(skills, scores), start=1):
        block = _format_skill_block(sk)
        parts.append(f"[Rank {rank}, similarity={float(sc):.6f}]\n{block}")
    return "\n\n---\n\n".join(parts)


def _lookup_skill(by_id: dict[Any, dict], qid: Any) -> Optional[dict]:
    if qid in by_id:
        return by_id[qid]
    if str(qid) in by_id:
        return by_id[str(qid)]
    return None


def _build_solver_messages(problem: str, system_prompt: str, skill_block: str):
    """direct / retrieve 均把技能（单条或 top-k 打包文本）填入 {skill}。"""
    user = prompt_use_skill.format(skill=skill_block, question=problem)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


class _SkillEmbeddingRetriever:
    """用句向量对「题目 vs skill problem type」做余弦相似度（向量已 L2 归一化则为点积）。"""

    def __init__(
        self,
        model_name: str,
        skills_corpus: list[dict],
        batch_size: int,
        device: Optional[str],
    ):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as e:
            print(
                "[solver] retrieve 模式需要 sentence-transformers，请: pip install sentence-transformers",
                file=sys.stderr,
            )
            raise e

        self.skills = skills_corpus
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        dev = _normalize_embedding_device(device)
        if dev:
            kwargs["device"] = dev
            print(f"[solver] 嵌入模型 device={dev}", file=sys.stderr)
        print(f"[solver] 加载嵌入模型: {model_name} …", file=sys.stderr)
        self._model = SentenceTransformer(model_name, **kwargs)
        texts = [_skill_retrieval_text(s) for s in skills_corpus]
        empty_n = sum(1 for t in texts if not t.strip())
        if empty_n:
            print(
                f"[solver] 警告: {empty_n} 条技能缺少 problem type / skill name，检索文本为空串",
                file=sys.stderr,
            )
        emb = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
        )
        self._doc_emb = np.asarray(emb, dtype=np.float32)

    def topk(self, problem: str, k: int) -> tuple[list[dict], list[float]]:
        if not self.skills:
            return [], []
        k = min(k, len(self.skills))
        q = self._model.encode(
            [problem],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        q = np.asarray(q, dtype=np.float32)
        sims = self._doc_emb @ q
        idx = np.argsort(-sims)[:k]
        picked = [self.skills[int(i)] for i in idx]
        scores = [float(sims[int(i)]) for i in idx]
        return picked, scores


def _mode_subdir(solver_mode: str, retrieve_k: int) -> str:
    if solver_mode == "direct":
        return "direct"
    return f"retrieve_top{retrieve_k}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to jsonl (e.g. datas/AIME-24.jsonl).",
    )
    parser.add_argument(
        "--skills_jsonl",
        type=Path,
        required=True,
        help="Path to skills.jsonl.",
    )
    parser.add_argument(
        "--solver_mode",
        type=str,
        choices=("direct", "retrieve"),
        default="direct",
        help="direct: 按题目 id 匹配技能；retrieve: 用嵌入对 problem type 检索 top-k。",
    )
    parser.add_argument(
        "--retrieve_k",
        type=int,
        default=3,
        help="retrieve 模式下注入的技能条数（嵌入 top-k）。",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="retrieve 模式下的句向量模型（sentence-transformers）。",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=32,
        help="编码技能库时的 batch 大小。",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default=None,
        help="如 cuda:2、cpu；仅写卡号数字如 2 会自动变为 cuda:2。默认由库自选。",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://127.0.0.1:5000",
    )
    parser.add_argument("--n", type=int, default=1, help="Rollouts per question.")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("runs") / "solver_with_skills",
    )
    parser.add_argument("--run_label", type=str, default="")
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--system", type=str, default=roll_r.DEFAULT_SYSTEM)
    parser.add_argument("--request_timeout", type=float, default=3600.0)
    parser.add_argument("--no_tqdm", action="store_true")
    args = parser.parse_args()

    if args.n < 1:
        print("[solver] --n must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.solver_mode == "retrieve" and args.retrieve_k < 1:
        print("[solver] --retrieve_k must be >= 1 in retrieve mode", file=sys.stderr)
        sys.exit(1)

    data_path = args.data.resolve()
    skills_path = args.skills_jsonl.resolve()
    if not data_path.is_file():
        print(f"[solver] data not found: {data_path}", file=sys.stderr)
        sys.exit(1)
    if not skills_path.is_file():
        print(f"[solver] skills not found: {skills_path}", file=sys.stderr)
        sys.exit(1)

    rows = roll_r._read_jsonl(data_path)
    if not rows:
        print("[solver] no rows loaded", file=sys.stderr)
        sys.exit(1)

    skills_by_id, skills_corpus, n_skill_rows = _load_skills_from_jsonl(skills_path)
    print(
        f"[solver] 已加载 skills: {skills_path}（{n_skill_rows} 行含 id，"
        f"corpus 去重 {len(skills_corpus)} 条）mode={args.solver_mode}",
        file=sys.stderr,
    )

    retriever: Optional[_SkillEmbeddingRetriever] = None
    if args.solver_mode == "retrieve":
        if not skills_corpus:
            print("[solver] retrieve 模式需要非空技能库", file=sys.stderr)
            sys.exit(1)
        retriever = _SkillEmbeddingRetriever(
            args.embedding_model,
            skills_corpus,
            args.embedding_batch_size,
            args.embedding_device,
        )

    dataset_stem = data_path.stem
    mode_seg = _mode_subdir(args.solver_mode, args.retrieve_k)

    if args.output_jsonl is not None:
        out_dir = args.output_jsonl.resolve().parent
        traj_path = args.output_jsonl.resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = args.output_root.resolve() / dataset_stem / mode_seg
        if args.run_label.strip():
            out_dir = out_dir / args.run_label.strip()
        out_dir = out_dir / ts
        traj_path = out_dir / "rollouts.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"

    if traj_path.is_file() and args.fresh:
        traj_path.unlink()

    merged: dict[str, dict] = {}
    if traj_path.is_file() and not args.fresh:
        merged = roll_r._load_checkpoint(traj_path)
        print(
            f"[solver] 断点缓存: 读取 {traj_path}（合并键 {len(merged)} 个）",
            file=sys.stderr,
        )
    elif traj_path.is_file() and args.fresh:
        print(f"[solver] --fresh：已忽略原文件，重新写入 {traj_path}", file=sys.stderr)
    else:
        print(f"[solver] 无缓存，输出将写入 {traj_path}", file=sys.stderr)

    meta = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "data": str(data_path),
        "skills_jsonl": str(skills_path),
        "solver_mode": args.solver_mode,
        "retrieve_k": args.retrieve_k if args.solver_mode == "retrieve" else None,
        "embedding_model": args.embedding_model if args.solver_mode == "retrieve" else None,
        "embedding_batch_size": args.embedding_batch_size
        if args.solver_mode == "retrieve"
        else None,
        "output_subdir_mode": mode_seg,
        "server": args.server,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "system_prompt": args.system,
        "output_jsonl": str(traj_path),
        "resume": traj_path.is_file() and not args.fresh and bool(merged),
        "solver_with_skills": True,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    health_url = args.server.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=10.0) as r:
            if r.status != 200:
                print(f"[solver] health check HTTP {r.status}", file=sys.stderr)
    except Exception as e:
        print(f"[solver] health check failed: {e}", file=sys.stderr)

    st0 = roll_r._rollout_progress_stats(rows, merged, args.n)
    print(
        f"[solver] 题目进度: 已完成 {st0['question_done']}/{st0['question_total']}，"
        f"尚需处理 {st0['question_need_work']} 条（其中需调用接口约 {st0['need_http_questions']} 题）",
        file=sys.stderr,
    )
    print(
        f"[solver] Rollout 进度: 缓存已有 {st0['rollout_done']}/{st0['rollout_target']}，"
        f"还差 {st0['rollout_remaining']} 条生成",
        file=sys.stderr,
    )

    use_tqdm = not args.no_tqdm
    pbar = tqdm(
        enumerate(rows),
        total=len(rows),
        desc="题目",
        unit="题",
        disable=not use_tqdm,
    )

    for idx, row in pbar:
        key = roll_r._stable_key(row, idx)
        qid = row.get("id", idx)

        try:
            problem = roll_r._question_text(row)
        except KeyError as e:
            rec = merged.get(key) or {
                "problem": "",
                "answer": row.get("answer"),
                "id": qid,
                "idx": idx,
                "rollouts": [],
            }
            rec["error"] = str(e)
            rec["idx"] = idx
            merged[key] = rec
            roll_r._atomic_write_jsonl(
                traj_path, roll_r._build_ordered_records(rows, merged)
            )
            st = roll_r._rollout_progress_stats(rows, merged, args.n)
            pbar.set_postfix(
                roll=f"{st['rollout_done']}/{st['rollout_target']}",
                q=f"{st['question_done']}/{st['question_total']}",
            )
            continue

        skill_block: str
        extra_rec: dict[str, Any] = {}

        if args.solver_mode == "direct":
            sk = _lookup_skill(skills_by_id, qid)
            if sk is None:
                rec = merged.get(key) or {
                    "problem": problem,
                    "answer": row.get("answer"),
                    "id": qid,
                    "idx": idx,
                    "rollouts": [],
                }
                rec = roll_r._normalize_saved_record(dict(rec))
                rec["problem"] = problem
                rec["answer"] = row.get("answer")
                rec["id"] = qid
                rec["idx"] = idx
                err_msg = f"no skill for id={qid!r} in {skills_path}"
                rec["error"] = err_msg
                rec["rollouts"] = [
                    {"text": "", "api_error": err_msg} for _ in range(args.n)
                ]
                merged[key] = rec
                roll_r._atomic_write_jsonl(
                    traj_path, roll_r._build_ordered_records(rows, merged)
                )
                st = roll_r._rollout_progress_stats(rows, merged, args.n)
                pbar.set_postfix(
                    roll=f"{st['rollout_done']}/{st['rollout_target']}",
                    q=f"{st['question_done']}/{st['question_total']}",
                )
                continue

            skill_block = _format_skill_block(sk)
            if not skill_block.strip():
                rec = merged.get(key) or {
                    "problem": problem,
                    "answer": row.get("answer"),
                    "id": qid,
                    "idx": idx,
                    "rollouts": [],
                }
                rec = roll_r._normalize_saved_record(dict(rec))
                rec["problem"] = problem
                rec["answer"] = row.get("answer")
                rec["id"] = qid
                rec["idx"] = idx
                err_msg = "skill record has empty skill fields"
                rec["error"] = err_msg
                rec["rollouts"] = [
                    {"text": "", "api_error": err_msg} for _ in range(args.n)
                ]
                merged[key] = rec
                roll_r._atomic_write_jsonl(
                    traj_path, roll_r._build_ordered_records(rows, merged)
                )
                st = roll_r._rollout_progress_stats(rows, merged, args.n)
                pbar.set_postfix(
                    roll=f"{st['rollout_done']}/{st['rollout_target']}",
                    q=f"{st['question_done']}/{st['question_total']}",
                )
                continue
        else:
            assert retriever is not None
            top_skills, top_scores = retriever.topk(problem, args.retrieve_k)
            skill_block = _format_retrieved_skills_block(top_skills, top_scores)
            extra_rec["retrieved_skill_ids"] = [s.get("id") for s in top_skills]
            extra_rec["retrieved_similarities"] = top_scores
            if not skill_block.strip():
                rec = merged.get(key) or {
                    "problem": problem,
                    "answer": row.get("answer"),
                    "id": qid,
                    "idx": idx,
                    "rollouts": [],
                }
                rec = roll_r._normalize_saved_record(dict(rec))
                rec["problem"] = problem
                rec["answer"] = row.get("answer")
                rec["id"] = qid
                rec["idx"] = idx
                err_msg = "retrieved skills format empty"
                rec["error"] = err_msg
                rec["rollouts"] = [
                    {"text": "", "api_error": err_msg} for _ in range(args.n)
                ]
                for ek, ev in extra_rec.items():
                    rec[ek] = ev
                merged[key] = rec
                roll_r._atomic_write_jsonl(
                    traj_path, roll_r._build_ordered_records(rows, merged)
                )
                st = roll_r._rollout_progress_stats(rows, merged, args.n)
                pbar.set_postfix(
                    roll=f"{st['rollout_done']}/{st['rollout_target']}",
                    q=f"{st['question_done']}/{st['question_total']}",
                )
                continue

        gold = row.get("answer")
        rec = merged.get(key)
        if rec is None:
            rec = {
                "problem": problem,
                "answer": gold,
                "id": qid,
                "idx": idx,
                "rollouts": [],
            }
        else:
            rec = roll_r._normalize_saved_record(dict(rec))
            rec["problem"] = problem
            rec["answer"] = gold
            rec["id"] = qid
            rec["idx"] = idx
            rec.setdefault("rollouts", [])
            if isinstance(rec["rollouts"], list):
                rec["rollouts"] = [
                    x if isinstance(x, dict) else {"text": str(x)}
                    for x in rec["rollouts"]
                ]
            else:
                rec["rollouts"] = []

        for ek, ev in extra_rec.items():
            rec[ek] = ev

        rollouts: list[dict] = list(rec["rollouts"])
        if len(rollouts) >= args.n:
            rec["rollouts"] = rollouts[: args.n]
            merged[key] = rec
            st = roll_r._rollout_progress_stats(rows, merged, args.n)
            pbar.set_postfix(
                roll=f"{st['rollout_done']}/{st['rollout_target']}",
                q=f"{st['question_done']}/{st['question_total']}",
                skip="cache",
            )
            continue

        need = args.n - len(rollouts)
        if use_tqdm:
            sb = roll_r._rollout_progress_stats(rows, merged, args.n)
            pbar.set_postfix(
                roll=f"{sb['rollout_done']}/{sb['rollout_target']}",
                q=f"{sb['question_done']}/{sb['question_total']}",
                need_r=need,
            )

        messages = _build_solver_messages(problem, args.system, skill_block)

        body = {
            "messages": messages,
            "n": need,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
        print(
            f"[solver] 请求 id={qid!r} idx={idx} mode={args.solver_mode} 补全 {need}/{args.n} 条 rollout …",
            file=sys.stderr,
        )
        resp = roll_r._post_generate(args.server, body, args.request_timeout)
        responses = roll_r._parse_text_response(resp)
        err = resp.get("error")
        err_s = (
            None
            if not err
            else err
            if isinstance(err, str)
            else json.dumps(err, ensure_ascii=False)
        )
        roll_r._append_rollouts(rollouts, need, responses, err_s)
        rec["rollouts"] = rollouts
        merged[key] = rec
        roll_r._atomic_write_jsonl(
            traj_path, roll_r._build_ordered_records(rows, merged)
        )
        st = roll_r._rollout_progress_stats(rows, merged, args.n)
        pbar.set_postfix(
            roll=f"{st['rollout_done']}/{st['rollout_target']}",
            q=f"{st['question_done']}/{st['question_total']}",
        )

    pbar.close()
    print(f"[solver] 完成，写入: {traj_path}", file=sys.stderr)
    print(str(traj_path))


if __name__ == "__main__":
    main()
