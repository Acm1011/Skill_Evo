"""Prepare jsonl/parquet with parametric-retriever-ranked Memento skills injected into prompts."""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_skill_use_template() -> str:
    return (_prompt_dir() / "skill_use_math.txt").read_text(encoding="utf-8")


def extract_problem(row: Dict[str, Any]) -> Optional[str]:
    for key in ("problem", "question", "raw_question"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in ("problem", "question"):
            val = extra.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def extract_topic(row: Dict[str, Any]) -> Optional[str]:
    val = row.get("topic")
    if isinstance(val, str) and val.strip():
        return val.strip()
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        val = extra.get("topic")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_idx(row: Dict[str, Any], line_no: int) -> Any:
    if row.get("idx") is not None:
        return row.get("idx")
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("idx") is not None:
        return extra.get("idx")
    return line_no


def extract_ground_truth(row: Dict[str, Any]) -> Any:
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if gt is not None:
            return gt
    if row.get("gt") is not None:
        return row.get("gt")
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in ("answer", "solution"):
            val = extra.get(key)
            if val is not None:
                return val
    return ""


def extract_data_source(row: Dict[str, Any], default: str) -> str:
    val = row.get("data_source")
    if isinstance(val, str) and val.strip():
        return val.strip()
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        val = extra.get("data_source")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return default


def format_skill_prompt(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No relevant skills found."
    parts: List[str] = []
    for i, row in enumerate(rows, start=1):
        parts.append(f"### Relevant Skill {i}")
        parts.append(f"- Case Label: {row.get('case_label')}")
        parts.append(f"- Topic: {row.get('topic_key')}")
        takeaway = str(row.get("takeaway") or "").strip()
        if takeaway:
            parts.append(f"- Takeaway: {takeaway}")
        for step in row.get("plan_steps") or []:
            sid = step.get("id")
            desc = step.get("description")
            parts.append(f"{sid}. {desc}" if sid is not None else f"- {desc}")
        parts.append("")
    return "\n".join(parts).strip()


def _build_retrieval_context(
    memory_rows: List[Dict[str, Any]],
    pool_rows: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[tuple[str, str, str], Dict[str, Any]]]:
    from .parametric_memory import build_icl_text

    icl_pool: List[str] = []
    metadata: List[Dict[str, Any]] = []
    by_key: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for mem, pool in zip(memory_rows, pool_rows):
        case = pool.get("case")
        plan = pool.get("plan")
        icl_pool.append(build_icl_text(case, plan))
        metadata.append(
            {
                "case": case,
                "plan": plan,
                "case_label": pool.get("case_label", "unknown"),
                "topic_key": mem.get("topic_key"),
                "memory_id": mem.get("memory_id"),
            }
        )
        key = (
            str(pool.get("case") or ""),
            str(pool.get("plan") or ""),
            str(pool.get("case_label") or ""),
        )
        by_key[key] = mem
    return icl_pool, metadata, by_key


def _materialize_input_records(path: str, start: int, end: int | None) -> List[Tuple[int, str]]:
    records: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < start:
                continue
            if end is not None and line_no >= end:
                break
            s = line.strip()
            if s:
                records.append((line_no, s))
    return records


def _process_record(
    *,
    raw: Dict[str, Any],
    line_no: int,
    top_k: int,
    data_source: str,
    model_path: str,
    template: str,
    retriever,
    icl_pool: List[str],
    metadata: List[Dict[str, Any]],
    by_key: Dict[tuple[str, str, str], Dict[str, Any]],
    keep_raw_prompt: bool,
    keep_raw_row: bool,
) -> Dict[str, Any] | None:
    problem = extract_problem(raw)
    if not problem:
        return None
    topic = extract_topic(raw)
    gt = extract_ground_truth(raw)
    source = extract_data_source(raw, data_source)

    ranked = retriever.retrieve(problem, icl_pool, metadata)
    top_scan = min(len(ranked), max(top_k * 8, top_k + 16))
    picked: List[Dict[str, Any]] = []
    seen = set()
    for item in heapq.nlargest(top_scan, ranked, key=lambda x: float(x.get("score") or 0.0)):
        key = (
            str(item.get("case") or ""),
            str(item.get("plan") or ""),
            str(item.get("case_label") or ""),
        )
        mem = by_key.get(key)
        if mem is None:
            continue
        mid = str(mem.get("memory_id") or "")
        if mid and mid in seen:
            continue
        seen.add(mid)
        mem_copy = dict(mem)
        mem_copy["_score"] = float(item.get("score") or 0.0)
        picked.append(mem_copy)
        if len(picked) >= top_k:
            break

    skill_text = format_skill_prompt(picked)
    user_content = template.format(skill=skill_text, question=problem)
    prompt = [{"role": "user", "content": user_content}]
    extra = raw.get("extra_info") if isinstance(raw.get("extra_info"), dict) else {}
    ex = dict(extra)
    ex["problem"] = problem
    ex["topic"] = topic
    ex["idx"] = extract_idx(raw, line_no)
    ex["top_k_skill"] = top_k
    ex["retriever_model_path"] = model_path
    ex["retrieved_memory_ids"] = [str(r.get("memory_id") or "") for r in picked]
    ex["retrieved_memory_count"] = len(picked)

    rec: Dict[str, Any] = {
        "_prepare_line_no": line_no,
        "problem": problem,
        "ground_truth": gt,
        "prompt": prompt,
        "reward_model": {"ground_truth": gt},
        "data_source": source,
        "extra_info": ex,
    }
    if keep_raw_prompt and isinstance(raw.get("prompt"), list):
        rec["original_prompt"] = copy.deepcopy(raw["prompt"])
    if keep_raw_row:
        rec["raw_row"] = copy.deepcopy(raw)
    return rec


def _worker_prepare_chunk(
    *,
    shard_id: int,
    device: str,
    model_path: str,
    model_name: str,
    score_batch_size: int,
    max_length: int,
    records: List[Tuple[int, str]],
    top_k: int,
    data_source: str,
    template: str,
    icl_pool: List[str],
    metadata: List[Dict[str, Any]],
    by_key: Dict[tuple[str, str, str], Dict[str, Any]],
    keep_raw_prompt: bool,
    keep_raw_row: bool,
    fail_on_retrieve_error: bool,
    progress_queue,
    result_queue,
    part_path: str,
) -> None:
    from .parametric_memory import CaseRetriever

    rows_out: List[Dict[str, Any]] = []
    skipped = 0
    flush_every = 8
    progress_local = 0
    try:
        retriever = CaseRetriever(
            model_path=model_path,
            model_name=model_name,
            device=device or None,
            score_batch_size=score_batch_size,
            max_length=max_length,
        )
        for line_no, line in records:
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    skipped += 1
                else:
                    rec = _process_record(
                        raw=raw,
                        line_no=line_no,
                        top_k=top_k,
                        data_source=data_source,
                        model_path=model_path,
                        template=template,
                        retriever=retriever,
                        icl_pool=icl_pool,
                        metadata=metadata,
                        by_key=by_key,
                        keep_raw_prompt=keep_raw_prompt,
                        keep_raw_row=keep_raw_row,
                    )
                    if rec is None:
                        skipped += 1
                    else:
                        rows_out.append(rec)
            except Exception as e:
                if fail_on_retrieve_error:
                    raise
                print(f"[prepare-prompt-data][worker {shard_id}] skip line {line_no}: {e}", file=sys.stderr)
                skipped += 1
            progress_local += 1
            if progress_local >= flush_every:
                progress_queue.put(progress_local)
                progress_local = 0

        if progress_local:
            progress_queue.put(progress_local)
        with Path(part_path).open("w", encoding="utf-8") as f:
            for rec in rows_out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        result_queue.put(
            {
                "ok": True,
                "shard_id": shard_id,
                "part_path": part_path,
                "rows": len(rows_out),
                "skipped": skipped,
            }
        )
    except Exception as e:
        result_queue.put(
            {
                "ok": False,
                "shard_id": shard_id,
                "part_path": part_path,
                "rows": len(rows_out),
                "skipped": skipped,
                "error": repr(e),
            }
        )
    finally:
        progress_queue.put(None)


def run_prepare_prompt_data(args: argparse.Namespace) -> int:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("prepare-prompt-data 需要 pandas、pyarrow: pip install pandas pyarrow") from e
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    from .io_utils import read_jsonl
    from .parametric_memory import CaseRetriever

    memory_rows = read_jsonl(args.memory_bank)
    pool_rows = read_jsonl(args.case_pool)
    template = load_skill_use_template()
    icl_pool, metadata, by_key = _build_retrieval_context(memory_rows, pool_rows)
    records = _materialize_input_records(args.input_jsonl, args.start, args.end)
    if not records:
        print("[prepare-prompt-data] no rows", file=sys.stderr)
        return 1

    devices_raw = (args.devices or "").strip()
    devices = [x.strip() for x in devices_raw.split(",") if x.strip()] if devices_raw else []
    if not devices:
        if args.device.strip():
            devices = [args.device.strip()]
        elif args.prefer_multi_gpu:
            devices = _default_cuda_devices() or ["cpu"]
        else:
            devices = ["cuda" if _torch_cuda_available() else "cpu"]
    n_workers = min(len(devices), len(records))

    rows_out: List[Dict[str, Any]] = []
    skipped = 0
    if n_workers <= 1:
        retriever = CaseRetriever(
            model_path=args.model_path,
            model_name=args.model_name,
            device=devices[0] if devices else (args.device or None),
            score_batch_size=args.score_batch_size,
            max_length=args.max_length,
        )
        iterator = records
        if tqdm is not None:
            iterator = tqdm(records, desc="prepare-prompt-data", unit="row")
        for line_no, line in iterator:
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    skipped += 1
                    continue
                rec = _process_record(
                    raw=raw,
                    line_no=line_no,
                    top_k=args.top_k,
                    data_source=args.data_source,
                    model_path=args.model_path,
                    template=template,
                    retriever=retriever,
                    icl_pool=icl_pool,
                    metadata=metadata,
                    by_key=by_key,
                    keep_raw_prompt=args.keep_raw_prompt,
                    keep_raw_row=args.keep_raw_row,
                )
                if rec is None:
                    skipped += 1
                else:
                    rows_out.append(rec)
            except Exception as e:
                if args.fail_on_retrieve_error:
                    raise SystemExit(f"prepare-prompt-data: retrieve failed at line {line_no}: {e}") from e
                print(f"[prepare-prompt-data] skip line {line_no}: retrieve failed: {e}", file=sys.stderr)
                skipped += 1
    else:
        ctx = mp.get_context("spawn")
        progress_q = ctx.Queue()
        result_q = ctx.Queue()
        tmp_dir = tempfile.mkdtemp(prefix="memento_prepare_", dir=str(Path(args.output_jsonl).resolve().parent))
        workers = []
        shards = [records[i::n_workers] for i in range(n_workers)]
        for i in range(n_workers):
            part_path = str(Path(tmp_dir) / f"part_{i:02d}.jsonl")
            proc = ctx.Process(
                target=_worker_prepare_chunk,
                kwargs={
                    "shard_id": i,
                    "device": devices[i],
                    "model_path": args.model_path,
                    "model_name": args.model_name,
                    "score_batch_size": args.score_batch_size,
                    "max_length": args.max_length,
                    "records": shards[i],
                    "top_k": args.top_k,
                    "data_source": args.data_source,
                    "template": template,
                    "icl_pool": icl_pool,
                    "metadata": metadata,
                    "by_key": by_key,
                    "keep_raw_prompt": args.keep_raw_prompt,
                    "keep_raw_row": args.keep_raw_row,
                    "fail_on_retrieve_error": args.fail_on_retrieve_error,
                    "progress_queue": progress_q,
                    "result_queue": result_q,
                    "part_path": part_path,
                },
            )
            proc.start()
            workers.append(proc)

        done_workers = 0
        pbar = tqdm(total=len(records), desc="prepare-prompt-data", unit="row") if tqdm is not None else None
        try:
            while done_workers < n_workers:
                msg = progress_q.get()
                if msg is None:
                    done_workers += 1
                elif pbar is not None:
                    pbar.update(int(msg))
        finally:
            if pbar is not None:
                pbar.close()

        summaries = [result_q.get() for _ in range(n_workers)]
        for proc in workers:
            proc.join()
            if proc.exitcode not in (0, None):
                raise SystemExit(f"prepare-prompt-data worker failed with exit code {proc.exitcode}")

        bad = [x for x in summaries if not x.get("ok")]
        if bad:
            raise SystemExit(f"prepare-prompt-data worker failed: {bad[0].get('error')}")
        skipped = sum(int(x.get("skipped") or 0) for x in summaries)
        for summary in summaries:
            part_path = Path(str(summary["part_path"]))
            if not part_path.is_file():
                continue
            with part_path.open("r", encoding="utf-8") as f:
                for line in f:
                    rows_out.append(json.loads(line))
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not rows_out:
        print("[prepare-prompt-data] no rows", file=sys.stderr)
        return 1
    rows_out.sort(key=lambda x: int(x.get("_prepare_line_no", 0)))
    for rec in rows_out:
        rec.pop("_prepare_line_no", None)

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in rows_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[prepare-prompt-data] jsonl={len(rows_out)} -> {out_jsonl}", file=sys.stderr)

    out_parquet = Path(args.output_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_parquet(out_parquet, index=False)
    print(f"[prepare-prompt-data] parquet={len(rows_out)} -> {out_parquet}", file=sys.stderr)
    print(f"[prepare-prompt-data] skipped={skipped}", file=sys.stderr)
    return 0


def build_prepare_prompt_data_parser(sub: Any) -> None:
    p = sub.add_parser("prepare-prompt-data", help="Input jsonl + retrieved skills -> output jsonl/parquet")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--memory-bank", required=True)
    p.add_argument("--case-pool", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default="princeton-nlp/sup-simcse-roberta-base")
    p.add_argument("--device", default="")
    p.add_argument("--devices", default="")
    p.add_argument("--prefer-multi-gpu", action="store_true")
    p.add_argument("--score-batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--data-source", default="MementoMath")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--keep-raw-prompt", action="store_true")
    p.add_argument("--keep-raw-row", action="store_true")
    p.add_argument("--fail-on-retrieve-error", action="store_true")
    p.set_defaults(_run=run_prepare_prompt_data)


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _default_cuda_devices() -> List[str]:
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
