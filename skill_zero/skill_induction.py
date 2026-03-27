#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 rollouts.jsonl 采样轨迹，调用 vLLM 归纳技能，写出带时间戳的 meta + skills.jsonl。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from eval import is_rollout_correct
from prompts import prompt_skill_induction

SKILL_KEYS = ("skill name", "problem type", "key insight", "method")
SKILL_FROM_SUCCESS = "success_rollout"
SKILL_FROM_FAIL = "fail_rollout"
SKILL_FROM_MIXED = "混合成功和失败轨迹"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[skill_induction] skip line {line_no}: {e}", file=sys.stderr)
    return rows


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".jsonl.tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as wf:
            for rec in records:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.isfile(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _has_usable_rollouts(rec: dict, gold: Any) -> bool:
    s, f = _collect_labeled_rollouts(rec, gold)
    return bool(s or f)


def _collect_labeled_rollouts(rec: dict, gold: Any) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """返回 (success 列表, fail 列表)，元素为 (rollout_index, text)，跳过空文本。"""
    success: list[tuple[int, str]] = []
    fail: list[tuple[int, str]] = []
    for i, r in enumerate(rec.get("rollouts") or []):
        if isinstance(r, dict):
            t = (r.get("text") or "").strip()
        else:
            t = str(r).strip()
        if not t:
            continue
        if is_rollout_correct(t, gold):
            success.append((i, t))
        else:
            fail.append((i, t))
    return success, fail


def _stratified_sample_k(
    success: list[tuple[int, str]],
    fail: list[tuple[int, str]],
    k: int,
    rng: random.Random,
) -> list[tuple[int, str, bool]]:
    """
    无放回采样至多 k 条；(roll_idx, text, is_success)。
    尽量同时包含成功与失败；仅一类则从该类取。
    """
    if k <= 0:
        return []
    S, F = success, fail
    if not S and not F:
        return []
    if not S:
        fp: list[tuple[int, str]] = rng.sample(F, min(k, len(F)))
        return [(i, t, False) for i, t in fp]
    if not F:
        sp: list[tuple[int, str]] = rng.sample(S, min(k, len(S)))
        return [(i, t, True) for i, t in sp]
    if k == 1:
        pool = [(i, t, True) for i, t in S] + [(i, t, False) for i, t in F]
        i, t, ok = rng.choice(pool)
        return [(i, t, ok)]

    take_s = min(len(S), max(1, k // 2))
    take_f = min(len(F), k - take_s)
    if take_f < 1:
        take_f = 1
        take_s = min(len(S), k - take_f)
    if take_s < 1:
        take_s = 1
        take_f = min(len(F), k - take_s)

    ps = rng.sample(S, take_s)
    pf = rng.sample(F, take_f)
    picked: list[tuple[int, str, bool]] = [(i, t, True) for i, t in ps] + [
        (i, t, False) for i, t in pf
    ]
    picked_keys = {(i, True) for i, _ in ps} | {(i, False) for i, _ in pf}
    rest = k - len(picked)
    if rest > 0:
        remain: list[tuple[int, str, bool]] = []
        for i, t in S:
            if (i, True) not in picked_keys:
                remain.append((i, t, True))
        for i, t in F:
            if (i, False) not in picked_keys:
                remain.append((i, t, False))
        if remain:
            add = rng.sample(remain, min(rest, len(remain)))
            picked.extend(add)
    rng.shuffle(picked)
    return picked[:k]


def _skill_from_label(sampled: list[tuple[int, str, bool]]) -> str:
    oks = [x for x in sampled if x[2]]
    bads = [x for x in sampled if not x[2]]
    if oks and bads:
        return SKILL_FROM_MIXED
    if oks:
        return SKILL_FROM_SUCCESS
    return SKILL_FROM_FAIL


def _pack_trajectories(sampled: list[tuple[int, str, bool]]) -> str:
    lines = []
    for _, text, ok in sampled:
        tag = "[SUCCESS]" if ok else "[FAIL]"
        lines.append(f"{tag} {text}")
    return "\n\n".join(lines)


def _parse_model_json(raw: str) -> tuple[Optional[dict], Optional[str]]:
    s = raw.strip()
    if s.startswith("```"):
        parts = s.split("\n", 1)
        s = parts[1] if len(parts) > 1 else ""
        if "```" in s:
            s = s.rsplit("```", 1)[0]
        s = s.strip()
    try:
        return json.loads(s), None
    except json.JSONDecodeError as e:
        return None, str(e)


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


def _parse_text_response(data: dict) -> str:
    if "error" in data:
        return ""
    t = data.get("text")
    if isinstance(t, str):
        return t
    if isinstance(t, list) and t:
        return str(t[0])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts_jsonl",
        type=Path,
        required=True,
        help="Path to rollouts.jsonl (e.g. runs/rollout/.../rollouts.jsonl).",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://127.0.0.1:5000",
    )
    parser.add_argument("--k", type=int, default=4, help="Sample k trajectories per question.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("runs") / "skill_induction",
    )
    parser.add_argument("--run_label", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant. Follow the user instructions exactly.",
    )
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument(
        "--no_tqdm",
        action="store_true",
        help="Disable tqdm progress bar (e.g. when piping logs).",
    )
    args = parser.parse_args()

    if args.k < 1:
        print("[skill_induction] --k must be >= 1", file=sys.stderr)
        sys.exit(1)

    roll_path = args.rollouts_jsonl.resolve()
    if not roll_path.is_file():
        print(f"[skill_induction] file not found: {roll_path}", file=sys.stderr)
        sys.exit(1)

    rows = _read_jsonl(roll_path)
    if not rows:
        print("[skill_induction] no rows", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = roll_path.stem
    out_dir = args.output_root.resolve() / stem
    if args.run_label.strip():
        out_dir = out_dir / args.run_label.strip()
    out_dir = out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    skills_path = out_dir / "skills.jsonl"
    meta_path = out_dir / "meta.json"

    n_usable = sum(1 for r in rows if _has_usable_rollouts(r, r.get("answer")))
    n_empty = len(rows) - n_usable
    print(
        f"[skill_induction] 输出目录: {out_dir}\nskills: {skills_path}",
        file=sys.stderr,
    )
    print(
        f"[skill_induction] 共 {len(rows)} 题；有非空轨迹可采样约 {n_usable} 题，"
        f"将跳过（无可用轨迹）约 {n_empty} 题；每题至多请求 1 次技能归纳",
        file=sys.stderr,
    )

    health_url = args.server.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=10.0) as r:
            if r.status != 200:
                print(
                    f"[skill_induction] health check HTTP {r.status}",
                    file=sys.stderr,
                )
    except Exception as e:
        print(f"[skill_induction] health check failed: {e}", file=sys.stderr)

    rng = random.Random(args.seed)
    records_out: list[dict] = []
    n_parse_fail = 0
    n_api_fail = 0
    n_skip = 0
    n_json_ok = 0

    use_tqdm = not args.no_tqdm
    pbar = tqdm(
        enumerate(rows),
        total=len(rows),
        desc="技能归纳",
        unit="题",
        disable=not use_tqdm,
    )

    for idx, rec in pbar:
        qid = rec.get("id")
        problem = rec.get("problem", "")
        gold = rec.get("answer")
        success, fail = _collect_labeled_rollouts(rec, gold)
        sampled = _stratified_sample_k(success, fail, args.k, rng)
        if not sampled:
            n_skip += 1
            row = {
                "skill name": None,
                "problem type": None,
                "key insight": None,
                "method": None,
                "skill_from": None,
                "id": qid,
                "problem": problem,
                "sampled_rollout_indices": [],
                "parse_error": "no_non_empty_rollouts",
                "raw_model_output": None,
            }
            records_out.append(row)
            _atomic_write_jsonl(skills_path, records_out)
            pbar.set_postfix(
                skip=n_skip,
                api_err=n_api_fail,
                json_err=n_parse_fail,
                json_ok=n_json_ok,
            )
            continue

        skill_from = _skill_from_label(sampled)
        traj_block = _pack_trajectories(sampled)
        user_prompt = prompt_skill_induction.format(
            question=problem, trajectories=traj_block
        )
        n_succ = sum(1 for x in sampled if x[2])
        n_fail = len(sampled) - n_succ
        if use_tqdm:
            pbar.set_postfix(
                skip=n_skip,
                api_err=n_api_fail,
                json_err=n_parse_fail,
                json_ok=n_json_ok,
                sample=f"S{n_succ}/F{n_fail}",
            )
        print(
            f"[skill_induction] 请求 id={qid!r} idx={idx} "
            f"采样 {len(sampled)}/{args.k} 条 (成功轨迹 {n_succ}, 失败 {n_fail}) "
            f"skill_from={skill_from!r} …",
            file=sys.stderr,
        )
        body = {
            "messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": user_prompt},
            ],
            "n": 1,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
        resp = _post_generate(args.server, body, args.request_timeout)
        raw = _parse_text_response(resp)
        if resp.get("error"):
            n_api_fail += 1
            err = (
                resp["error"]
                if isinstance(resp["error"], str)
                else json.dumps(resp["error"], ensure_ascii=False)
            )
            row = {
                "skill name": None,
                "problem type": None,
                "key insight": None,
                "method": None,
                "skill_from": skill_from,
                "id": qid,
                "problem": problem,
                "sampled_rollout_indices": [x[0] for x in sampled],
                "parse_error": f"api_error: {err}",
                "raw_model_output": raw or None,
            }
            records_out.append(row)
            _atomic_write_jsonl(skills_path, records_out)
            pbar.set_postfix(
                skip=n_skip,
                api_err=n_api_fail,
                json_err=n_parse_fail,
                json_ok=n_json_ok,
            )
            continue

        parsed, perr = _parse_model_json(raw)
        if perr:
            n_parse_fail += 1
        else:
            n_json_ok += 1
        row = {
            "skill name": None,
            "problem type": None,
            "key insight": None,
            "method": None,
            "skill_from": skill_from,
            "id": qid,
            "problem": problem,
            "sampled_rollout_indices": [x[0] for x in sampled],
            "parse_error": perr,
            "raw_model_output": raw if perr else None,
        }
        if parsed:
            for key in SKILL_KEYS:
                row[key] = parsed.get(key)
        records_out.append(row)
        _atomic_write_jsonl(skills_path, records_out)
        pbar.set_postfix(
            skip=n_skip,
            api_err=n_api_fail,
            json_err=n_parse_fail,
            json_ok=n_json_ok,
        )

    pbar.close()

    meta = {
        "created_at": ts,
        "rollouts_jsonl": str(roll_path),
        "server": args.server,
        "k": args.k,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "output_dir": str(out_dir),
        "skills_jsonl": str(skills_path),
        "num_questions": len(rows),
        "num_skipped_empty_rollouts": n_skip,
        "num_api_errors": n_api_fail,
        "num_json_parse_errors": n_parse_fail,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[skill_induction] 完成: 跳过无轨迹 {n_skip}, API 错误 {n_api_fail}, "
        f"JSON 解析失败 {n_parse_fail}, 解析成功 {n_json_ok}",
        file=sys.stderr,
    )
    print(str(skills_path))


if __name__ == "__main__":
    main()
