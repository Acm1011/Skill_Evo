#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from typing import Optional
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

DEFAULT_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def _read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[rollout] skip line {line_no}: {e}", file=sys.stderr)
    return rows


def _question_text(row: dict) -> str:
    if "problem" in row:
        return str(row["problem"])
    if "question" in row:
        return str(row["question"])
    raise KeyError("row must contain 'problem' or 'question'")


def _stable_key(row: dict, idx: int) -> str:
    # 与写入/断点一致：qid = row.get("id", idx)，避免缓存键 id:N 与查找 idx:N 不一致。
    qid = row.get("id", idx)
    if qid is not None:
        return f"id:{qid}"
    return f"idx:{idx}"


def _normalize_saved_record(rec: dict) -> dict:
    """兼容旧版 responses 字段，统一到 rollouts。"""
    if "rollouts" in rec and rec["rollouts"]:
        return rec
    if "responses" in rec:
        out = {k: v for k, v in rec.items() if k != "responses"}
        out["rollouts"] = [{"text": str(t)} for t in rec["responses"]]
        return out
    rec = dict(rec)
    rec.setdefault("rollouts", [])
    return rec


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """key -> 该样本最新一条记录（按 stable_key）。"""
    merged: dict[str, dict] = {}
    if not path.is_file():
        return merged
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = _normalize_saved_record(json.loads(line))
        except json.JSONDecodeError:
            continue
        k = None
        if "id" in rec and rec["id"] is not None:
            k = f"id:{rec['id']}"
        if k is None and "idx" in rec:
            k = f"idx:{rec['idx']}"
        if k is None:
            continue
        merged[k] = rec
    return merged


def _build_ordered_records(rows: list[dict], merged: dict[str, dict]) -> list[dict]:
    """与输入行顺序一致，用于整文件重写。"""
    out = []
    for idx, row in enumerate(rows):
        key = _stable_key(row, idx)
        rec = merged.get(key)
        if rec is not None:
            out.append(rec)
            continue
        try:
            problem = _question_text(row)
        except KeyError:
            qid = row.get("id", idx)
            out.append(
                {
                    "problem": "",
                    "answer": row.get("answer"),
                    "id": qid,
                    "idx": idx,
                    "rollouts": [],
                    "error": "row must contain 'problem' or 'question'",
                }
            )
            continue
        qid = row.get("id", idx)
        out.append(
            {
                "problem": problem,
                "answer": row.get("answer"),
                "id": qid,
                "idx": idx,
                "rollouts": [],
            }
        )
    return out


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".jsonl.tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.isfile(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _build_messages(problem: str, system_prompt: str):
    user = problem
    user = f"{problem}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def _post_generate(base_url: str, body: dict, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/generate"
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(err_body)
        except json.JSONDecodeError:
            return {"error": err_body or str(e)}


def _parse_text_response(data: dict) -> list[str]:
    if "error" in data:
        return []
    t = data.get("text")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [str(x) for x in t]
    return []


def _rollout_progress_stats(rows: list[dict], merged: dict[str, dict], n: int) -> dict:
    """统计题目与 rollout 完成度（与主循环判定一致）。"""
    rollout_target = 0
    rollout_done = 0
    question_done = 0
    question_total = len(rows)
    need_http_questions = 0
    for idx, row in enumerate(rows):
        key = _stable_key(row, idx)
        try:
            _question_text(row)
        except KeyError:
            rec = merged.get(key)
            if rec and rec.get("error"):
                question_done += 1
            continue
        rollout_target += n
        rec = merged.get(key)
        if rec:
            rec = _normalize_saved_record(dict(rec))
            ro = [
                x if isinstance(x, dict) else {"text": str(x)}
                for x in (rec.get("rollouts") or [])
            ]
            rollout_done += min(len(ro), n)
            if len(ro) >= n:
                question_done += 1
            else:
                need_http_questions += 1
        else:
            need_http_questions += 1
    return {
        "question_total": question_total,
        "question_done": question_done,
        "question_need_work": question_total - question_done,
        "rollout_target": rollout_target,
        "rollout_done": rollout_done,
        "rollout_remaining": rollout_target - rollout_done,
        "need_http_questions": need_http_questions,
    }


def _append_rollouts(
    rollouts: list[dict],
    need: int,
    responses: list[str],
    api_error: Optional[str],
) -> None:
    if api_error:
        for _ in range(need):
            rollouts.append({"text": "", "api_error": api_error})
        return
    for t in responses[:need]:
        rollouts.append({"text": t})
    short = need - len(responses[:need])
    for _ in range(short):
        rollouts.append({"text": "", "api_error": "missing_completion"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to jsonl (e.g. datas/AIME-24.jsonl).",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://127.0.0.1:5000",
        help="Base URL of start_verl_server (no trailing path).",
    )
    parser.add_argument("--n", type=int, default=1, help="Rollouts per question.")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("runs") / "rollout",
        help="Root directory; files go under <root>/<dataset>/<ts>/ when --output_jsonl unset.",
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default="",
        help="Optional subfolder after dataset name, e.g. model or config tag.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=None,
        help="Explicit rollouts.jsonl path. If file exists and not --fresh, resume.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing output file and start from scratch (same path as --output_jsonl or run dir).",
    )
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=3600.0,
        help="Seconds for each HTTP generate call.",
    )
    parser.add_argument(
        "--no_tqdm",
        action="store_true",
        help="Disable tqdm progress bar (e.g. when piping logs).",
    )
    args = parser.parse_args()

    if args.n < 1:
        print("[rollout] --n must be >= 1", file=sys.stderr)
        sys.exit(1)

    data_path = args.data.resolve()
    if not data_path.is_file():
        print(f"[rollout] data not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    rows = _read_jsonl(data_path)
    if not rows:
        print("[rollout] no rows loaded", file=sys.stderr)
        sys.exit(1)

    dataset_stem = data_path.stem
    if args.output_jsonl is not None:
        out_dir = args.output_jsonl.resolve().parent
        traj_path = args.output_jsonl.resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = args.output_root.resolve() / dataset_stem
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
        merged = _load_checkpoint(traj_path)
        print(
            f"[rollout] 断点缓存: 读取 {traj_path}（合并键 {len(merged)} 个）",
            file=sys.stderr,
        )
    elif traj_path.is_file() and args.fresh:
        print(f"[rollout] --fresh：已忽略原文件，重新写入 {traj_path}", file=sys.stderr)
    else:
        print(f"[rollout] 无缓存，输出将写入 {traj_path}", file=sys.stderr)

    meta = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "data": str(data_path),
        "server": args.server,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "system_prompt": args.system,
        "output_jsonl": str(traj_path),
        "resume": traj_path.is_file() and not args.fresh and bool(merged),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    health_url = args.server.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=10.0) as r:
            if r.status != 200:
                print(f"[rollout] health check HTTP {r.status}", file=sys.stderr)
    except Exception as e:
        print(f"[rollout] health check failed: {e}", file=sys.stderr)

    st0 = _rollout_progress_stats(rows, merged, args.n)
    print(
        f"[rollout] 题目进度: 已完成 {st0['question_done']}/{st0['question_total']}，"
        f"尚需处理 {st0['question_need_work']} 条（其中需调用接口约 {st0['need_http_questions']} 题）",
        file=sys.stderr,
    )
    print(
        f"[rollout] Rollout 进度: 缓存已有 {st0['rollout_done']}/{st0['rollout_target']}，"
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
        key = _stable_key(row, idx)
        qid = row.get("id", idx)

        try:
            problem = _question_text(row)
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
            _atomic_write_jsonl(traj_path, _build_ordered_records(rows, merged))
            st = _rollout_progress_stats(rows, merged, args.n)
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
            rec = _normalize_saved_record(dict(rec))
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

        rollouts: list[dict] = list(rec["rollouts"])
        if len(rollouts) >= args.n:
            rec["rollouts"] = rollouts[: args.n]
            merged[key] = rec
            st = _rollout_progress_stats(rows, merged, args.n)
            pbar.set_postfix(
                roll=f"{st['rollout_done']}/{st['rollout_target']}",
                q=f"{st['question_done']}/{st['question_total']}",
                skip="cache",
            )
            continue

        need = args.n - len(rollouts)
        if use_tqdm:
            sb = _rollout_progress_stats(rows, merged, args.n)
            pbar.set_postfix(
                roll=f"{sb['rollout_done']}/{sb['rollout_target']}",
                q=f"{sb['question_done']}/{sb['question_total']}",
                need_r=need,
            )
        messages = _build_messages(problem, args.system)
        body = {
            "messages": messages,
            "n": need,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
        print(
            f"[rollout] 请求 id={qid!r} idx={idx} 补全 {need}/{args.n} 条 rollout …",
            file=sys.stderr,
        )
        resp = _post_generate(args.server, body, args.request_timeout)
        responses = _parse_text_response(resp)
        err = resp.get("error")
        err_s = None if not err else err if isinstance(err, str) else json.dumps(err, ensure_ascii=False)
        _append_rollouts(rollouts, need, responses, err_s)
        rec["rollouts"] = rollouts
        merged[key] = rec
        _atomic_write_jsonl(traj_path, _build_ordered_records(rows, merged))
        st = _rollout_progress_stats(rows, merged, args.n)
        pbar.set_postfix(
            roll=f"{st['rollout_done']}/{st['rollout_target']}",
            q=f"{st['question_done']}/{st['question_total']}",
        )

    pbar.close()
    print(f"[rollout] 完成，写入: {traj_path}", file=sys.stderr)
    print(str(traj_path))


if __name__ == "__main__":
    main()
