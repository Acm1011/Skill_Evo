from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.SkillRL.deepmath_io import load_records_in_range, resolve_rollout_server_urls
from baselines.SkillRL.student_rollout import extract_ground_truth, extract_meta, extract_problem, grade_if_possible
from baselines.SkillRL.vllm_http_client import VLLMHTTPClient

from .retrieve_memory import format_retrieved_prompt, retrieve_records
from .teacher import load_solve_instruction
from .text_utils import topic_slug


def _format_fewshot(rows: List[Dict[str, Any]], cap: int = 2) -> str:
    blocks: List[str] = []
    for row in rows[:cap]:
        traj = str(row.get("trajectory") or "").strip()
        if not traj:
            continue
        blocks.append(traj)
    if not blocks:
        return ""
    return "### Retrieved Example Trajectories\n\n" + "\n\n".join(blocks)


def build_eval_prompt(
    *,
    question: str,
    retrieved_rows: List[Dict[str, Any]],
    retrieval_mode: str,
) -> str:
    context_blocks: List[str] = []
    rules_block = format_retrieved_prompt(retrieved_rows)
    if rules_block:
        context_blocks.append(rules_block)
    if retrieval_mode == "fewshot":
        fewshot_rows = [r for r in retrieved_rows if str(r.get("memory_type") or "") != "failure_rule"]
        fewshot_block = _format_fewshot(fewshot_rows)
        if fewshot_block:
            context_blocks.append(fewshot_block)
    context = "\n\n".join(block for block in context_blocks if block.strip())
    template = load_solve_instruction()
    return template.format(retrieved_context=context or "No retrieved insights.", question=question)


def run_eval(args: argparse.Namespace) -> int:
    memory_rows = read_jsonl(args.memory_bank)
    embedding_rows = read_jsonl(args.embeddings)
    records = load_records_in_range(args.deepmath_jsonl, args.start, args.end)
    urls = resolve_rollout_server_urls(args.server_urls)
    client = VLLMHTTPClient(
        server_urls=urls,
        timeout=args.timeout,
        max_retries=args.max_retries,
        served_model_name=args.served_model_name or None,
        max_concurrent=max(0, args.max_concurrent),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampling = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    with out_path.open("w", encoding="utf-8") as fout:
        for local_i, row in enumerate(records):
            line_idx = args.start + local_i
            problem = extract_problem(row)
            if not problem:
                print(f"[eval] skip line {line_idx}: no problem", file=sys.stderr)
                continue
            gt = extract_ground_truth(row)
            meta = extract_meta(row, line_idx)
            scored = retrieve_records(
                question=problem,
                memory_rows=memory_rows,
                embedding_rows=embedding_rows,
                top_k=args.top_k,
                backend=args.backend,
                base_url=args.embed_base_url,
                api_key=args.embed_api_key,
                model=args.embed_model,
                timeout=args.timeout,
                hash_dim=args.hash_dim,
                query_topic=meta.get("topic"),
                topic_bonus=args.topic_bonus,
            )
            retrieved_rows = [dict(item) for item, _ in scored]
            prompt = build_eval_prompt(
                question=problem,
                retrieved_rows=retrieved_rows,
                retrieval_mode=args.retrieval_mode,
            )
            try:
                outs = client.generate_sync([prompt], sampling, request_timeout=args.timeout)
                text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
            except Exception as e:
                print(f"[eval] line {line_idx} HTTP error: {e}", file=sys.stderr)
                continue
            is_correct = grade_if_possible(text, gt)
            out_row = {
                "idx": meta["idx"],
                "problem": problem,
                "topic": meta.get("topic"),
                "topic_key": topic_slug(meta.get("topic")),
                "retrieved_memory_ids": [row.get("memory_id") for row in retrieved_rows],
                "retrieved_rules": [str(row.get("raw_rule") or "") for row in retrieved_rows],
                "student_response": text,
                "is_correct": is_correct,
                "ground_truth": gt,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()
    print(f"[eval] wrote evaluation records to {out_path}", file=sys.stderr)
    return 0


def build_eval_parser(sub: Any) -> None:
    p = sub.add_parser("eval", help="Solve DeepMath with retrieved ExpeLMath memories")
    p.add_argument("--deepmath-jsonl", required=True)
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--retrieval-mode", default="rules", choices=["rules", "fewshot"])
    p.add_argument("--backend", default="hash", choices=["hash", "openai"])
    p.add_argument("--embed-base-url", default="")
    p.add_argument("--embed-api-key", default="")
    p.add_argument("--embed-model", default="")
    p.add_argument("--topic-bonus", type=float, default=0.05)
    p.add_argument("--hash-dim", type=int, default=256)
    p.add_argument("--server-urls", nargs="*", default=None, help="vLLM base URLs")
    p.add_argument("--served-model-name", default="", help="Override VLLM_SERVED_MODEL_NAME")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-concurrent", type=int, default=0)
    p.set_defaults(_run=run_eval)

