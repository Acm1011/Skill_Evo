"""DeepMath rollout for SkillOpt."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.envs.deepmath.evaluator import evaluate_response
from skillopt.model import chat_target
from skillopt.prompts import load_prompt


def _build_system(skill_content: str) -> str:
    skill_section = f"## Skill Memory\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return load_prompt("rollout_system", env="deepmath").format(skill_section=skill_section)


def _build_user(item: dict) -> str:
    return f"## Problem\n{item['question']}"


def process_one(item: dict, out_root: str, skill_content: str, *, max_completion_tokens: int = 4096, timeout: int | None = 600) -> dict:
    item_id = str(item["id"])
    result = {
        "id": item_id,
        "question": item.get("question", ""),
        "task_description": item.get("question", ""),
        "task_type": item.get("task_type") or item.get("topic") or "general_math",
        "topic": item.get("topic"),
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "ground_truth": item.get("ground_truth", ""),
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
    }
    pred_dir = os.path.join(out_root, "predictions", item_id)
    os.makedirs(pred_dir, exist_ok=True)
    try:
        system = _build_system(skill_content)
        user = _build_user(item)
        response, _ = chat_target(system=system, user=user, max_completion_tokens=max_completion_tokens, retries=5, stage="rollout", timeout=timeout)
        eval_result = evaluate_response(response, str(item.get("ground_truth") or ""))
        result.update(eval_result)
        result["response"] = response
        result["agent_ok"] = True
        result["n_turns"] = 1
        if not result["hard"]:
            result["fail_reason"] = f"math_answer_mismatch: predicted {eval_result['predicted_answer']!r} but expected {eval_result['ground_truth']!r}"
        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
            {"role": "system", "content": json.dumps(eval_result, ensure_ascii=False)},
        ]
        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        result["fail_reason"] = f"error: {type(exc).__name__}: {exc}"
    return result


def run_batch(items: list[dict], out_root: str, skill_content: str, *, workers: int = 32, exec_timeout: int | None = 600, max_completion_tokens: int = 4096, task_timeout: int | None = 900) -> list[dict]:
    os.makedirs(out_root, exist_ok=True)
    results_path = os.path.join(out_root, "results.jsonl")
    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                done_ids.add(str(row.get("id")))
                existing.append(row)
    pending = [item for item in items if str(item.get("id")) not in done_ids]
    if not pending:
        return existing
    results = list(existing)
    started_at: dict[str, float] = {}

    def _timeout_result(item: dict) -> dict:
        return {
            "id": str(item.get("id")),
            "question": item.get("question", ""),
            "task_description": item.get("question", ""),
            "task_type": item.get("task_type") or item.get("topic") or "general_math",
            "topic": item.get("topic"),
            "hard": 0,
            "soft": 0.0,
            "predicted_answer": "",
            "ground_truth": item.get("ground_truth", ""),
            "response": "",
            "fail_reason": f"task-timeout-{task_timeout}s",
            "agent_ok": False,
            "n_turns": 0,
        }

    def _run_one(item: dict) -> dict:
        started_at[str(item.get("id"))] = time.time()
        return process_one(item, out_root, skill_content, max_completion_tokens=max_completion_tokens, timeout=exec_timeout)

    with open(results_path, "a", encoding="utf-8") as out:
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {executor.submit(_run_one, item): item for item in pending}
            active = set(futures)
            while active:
                done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                timed_out = [fut for fut in active - done if task_timeout is not None and str(futures[fut].get("id")) in started_at and now - started_at[str(futures[fut].get("id"))] >= task_timeout]
                for fut in done:
                    active.remove(fut)
                    item = futures[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = _timeout_result(item)
                        row["fail_reason"] = f"error: {type(exc).__name__}: {exc}"
                    results.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"    [rollout] {len(results)}/{len(items)} id={row['id']} hard={row.get('hard')}", flush=True)
                for fut in timed_out:
                    active.remove(fut)
                    row = _timeout_result(futures[fut])
                    results.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    return results
