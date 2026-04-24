"""Generate student trajectories from DeepMath-103K via vLLM HTTP (OpenAI completions)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .deepmath_io import load_records_in_range, resolve_rollout_server_urls
from .text_utils import extract_all_boxed_content, topic_slug
from .vllm_http_client import VLLMHTTPClient


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def load_student_template() -> str:
    p = _prompt_dir() / "student_math.txt"
    return p.read_text(encoding="utf-8")


def extract_problem(row: Dict[str, Any]) -> Optional[str]:
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        prob = ei.get("problem")
        if isinstance(prob, str) and prob.strip():
            return prob.strip()
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


def extract_ground_truth(row: Dict[str, Any]) -> Optional[str]:
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if isinstance(gt, str) and gt.strip():
            return gt.strip()
        if isinstance(gt, list) and gt:
            if isinstance(gt[0], str):
                return gt[0].strip()
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        a = ei.get("answer") or ei.get("solution")
        if isinstance(a, str) and a.strip():
            return a.strip()
    return None


def extract_meta(row: Dict[str, Any], line_idx: int) -> Dict[str, Any]:
    ei = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    idx = row.get("idx", line_idx)
    if idx is None and isinstance(ei, dict):
        rqi = ei.get("raw_q_info") or {}
        if isinstance(rqi, dict):
            idx = rqi.get("idx", line_idx)
    topic = ei.get("topic") if isinstance(ei, dict) else None
    difficulty = ei.get("difficulty") if isinstance(ei, dict) else None
    return {
        "idx": idx if idx is not None else line_idx,
        "topic": topic,
        "difficulty": difficulty,
    }


def grade_if_possible(student_text: str, ground_truth: Optional[str]) -> Optional[bool]:
    if not ground_truth:
        return None
    boxed_list = extract_all_boxed_content(student_text)
    if not boxed_list:
        return None
    pred = boxed_list[-1].strip()
    gt = ground_truth.strip()
    try:
        from mathruler.grader import grade_answer

        return bool(grade_answer(pred, gt))
    except Exception:
        return pred.lower().rstrip(".") == gt.lower().rstrip(".")


def run_gen_traj(args: argparse.Namespace) -> int:
    template = load_student_template()
    records = load_records_in_range(args.data_path, args.start, args.end)
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

    mode = "w"
    if args.append and out_path.is_file():
        mode = "a"

    sampling = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    with out_path.open(mode, encoding="utf-8") as fout:
        for local_i, row in enumerate(records):
            line_idx = args.start + local_i
            problem = extract_problem(row)
            if not problem:
                print(f"[gen-traj] skip line {line_idx}: no problem", file=sys.stderr)
                continue
            gt = extract_ground_truth(row)
            meta = extract_meta(row, line_idx)
            user_prompt = template.format(problem=problem)
            try:
                outs = client.generate_sync([user_prompt], sampling, request_timeout=args.timeout)
                text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
            except Exception as e:
                print(f"[gen-traj] line {line_idx} HTTP error: {e}", file=sys.stderr)
                continue

            is_correct = grade_if_possible(text, gt)
            out_row = {
                "idx": meta["idx"],
                "line_idx": line_idx,
                "problem": problem,
                "topic": meta.get("topic"),
                "topic_key": topic_slug(meta.get("topic")),
                "difficulty": meta.get("difficulty"),
                "student_response": text,
                "is_correct": is_correct,
                "ground_truth": gt,
                "raw": {"keys": list(row.keys())[:20]} if args.debug_raw else None,
            }
            if out_row["raw"] is None:
                del out_row["raw"]
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[gen-traj] wrote trajectories to {out_path}")
    return 0


def build_gen_traj_parser(sub: Any) -> None:
    p = sub.add_parser("gen-traj", help="Student model rollout -> trajectories.jsonl")
    p.add_argument("--data-path", required=True, help="DeepMath-103K.jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, required=True, help="Exclusive end index")
    p.add_argument("--output", required=True, help="Output trajectories.jsonl")
    p.add_argument("--server-urls", nargs="*", default=None, help="vLLM base URLs")
    p.add_argument("--served-model-name", default="", help="Override VLLM_SERVED_MODEL_NAME")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-concurrent", type=int, default=0)
    p.add_argument("--append", action="store_true")
    p.add_argument("--debug-raw", action="store_true")
    p.set_defaults(_run=run_gen_traj)
