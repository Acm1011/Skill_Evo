from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import httpx
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

from baselines.ReasoningBankMath.io_utils import read_jsonl, write_jsonl
from baselines.SkillRL.student_rollout import grade_if_possible
from baselines.preliminary.eval_source_linked_skills import group_questions

DEFAULT_TRAJECTORIES = "Skill_Evo/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl"
METHODS = ("skillrl", "reasoningbank", "expelmath")
TEACHER_BACKENDS = ("api_teacher", "server_teacher")


class _ProgressBar:
    def __init__(self, *, total: int, desc: str, leave: bool = True) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.leave = leave
        self.count = 0
        self._last_len = 0
        self._use_tqdm = _tqdm is not None and sys.stderr.isatty()
        self._bar = _tqdm(total=self.total, desc=self.desc, leave=self.leave, dynamic_ncols=True) if self._use_tqdm else None
        if self._bar is None:
            self._render()

    def update(self, n: int = 1) -> None:
        self.count = min(self.total, self.count + max(0, int(n)))
        if self._bar is not None:
            self._bar.update(n)
            return
        self._render()

    def write(self, message: str) -> None:
        if self._bar is not None:
            self._bar.write(message)
            return
        if self._last_len:
            sys.stderr.write("\r" + (" " * self._last_len) + "\r")
            self._last_len = 0
        sys.stderr.write(message.rstrip() + "\n")
        sys.stderr.flush()
        self._render()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            return
        if self._last_len:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._last_len = 0

    def _render(self) -> None:
        total = self.total or 1
        width = 24
        filled = int(width * self.count / total)
        bar = "#" * filled + "-" * (width - filled)
        pct = int(100 * self.count / total) if self.total else 100
        line = f"{self.desc}: [{bar}] {self.count}/{self.total} ({pct}%)"
        sys.stderr.write("\r" + line)
        sys.stderr.flush()
        self._last_len = len(line)

    def __enter__(self) -> "_ProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "ReasoningBankMath" / "prompts" / "skill_use_math.txt"


def _load_solve_template() -> str:
    return _prompt_dir().read_text(encoding="utf-8")


def _question_lookup_key(question: Dict[str, Any]) -> Tuple[str, str]:
    meta = question["meta"]
    source_idx = meta.get("source_idx")
    if source_idx is not None:
        return "source_idx", str(source_idx)
    return "problem", str(meta.get("problem") or "")


def _skill_row_keys(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    if row.get("source_idx") is not None:
        keys.append(("source_idx", str(row.get("source_idx"))))
    problem = str(row.get("problem") or "").strip()
    if problem:
        keys.append(("problem", problem))
    return keys


def _parse_choice_filter(values: Optional[Sequence[str]], allowed: Sequence[str]) -> List[str]:
    if not values:
        return list(allowed)
    chosen = [str(v).strip().lower() for v in values if str(v).strip()]
    bad = [v for v in chosen if v not in allowed]
    if bad:
        raise SystemExit(f"invalid filter values: {bad}")
    return chosen


def _looks_like_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    weight_patterns = (
        "*.safetensors",
        "pytorch_model*.bin",
        "adapter_model*.bin",
        "model*.safetensors",
    )
    marker_files = (
        "config.json",
        "adapter_config.json",
        "generation_config.json",
    )
    has_marker = any((path / name).is_file() for name in marker_files)
    has_weight = any(any(path.glob(pattern)) for pattern in weight_patterns)
    if has_marker:
        return has_weight
    return has_weight


def _extract_checkpoint_order(path: Path) -> Tuple[int, str]:
    text = path.name
    patterns = [
        r"checkpoint[-_]?(\d+)",
        r"global[_-]?step[_-]?(\d+)",
        r"step[_-]?(\d+)",
        r"epoch[_-]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), text
    nums = re.findall(r"(\d+)", text)
    if nums:
        return int(nums[-1]), text
    return 10**18, text


def _checkpoint_display_name(root: Path, path: Path) -> str:
    try:
        rel_parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        rel_parts = path.parts
    if not rel_parts:
        return path.name
    for idx, part in enumerate(rel_parts):
        if _extract_checkpoint_order(Path(part))[0] != 10**18:
            suffix = rel_parts[idx:]
            return "__".join(suffix)
    return "__".join(rel_parts)


def _checkpoint_sort_key(root: Path, path: Path) -> Tuple[int, str]:
    try:
        rel_parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        rel_parts = path.parts
    for idx, part in enumerate(rel_parts):
        order, _ = _extract_checkpoint_order(Path(part))
        if order != 10**18:
            return order, "__".join(rel_parts[idx:])
    return _extract_checkpoint_order(path)


def discover_checkpoints(root: str | Path, limit: int = 0) -> List[Dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists():
        raise SystemExit(f"checkpoint root does not exist: {root_path}")

    candidates: List[Path] = []
    if _looks_like_checkpoint_dir(root_path):
        candidates.append(root_path)
    for path in root_path.rglob("*"):
        if _looks_like_checkpoint_dir(path):
            candidates.append(path)

    deduped = sorted({p.resolve() for p in candidates}, key=lambda p: (_checkpoint_sort_key(root_path, p), str(p)))
    out: List[Dict[str, Any]] = []
    for idx, path in enumerate(deduped):
        sort_key = _checkpoint_sort_key(root_path, path)
        out.append(
            {
                "checkpoint_path": str(path),
                "checkpoint_name": _checkpoint_display_name(root_path, path),
                "checkpoint_order": idx,
                "_sort_key": sort_key,
            }
        )
    if limit > 0:
        out = out[:limit]
        for idx, item in enumerate(out):
            item["checkpoint_order"] = idx
    return out


def load_skill_sets(
    skills_run_dir: str | Path,
    methods: Sequence[str],
    teacher_backends: Sequence[str],
) -> Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, Any]]]:
    root = Path(skills_run_dir) / "generated_skills"
    skill_sets: Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, Any]]] = {}
    for method in methods:
        for teacher_backend in teacher_backends:
            path = root / method / f"{teacher_backend}.jsonl"
            if not path.is_file():
                continue
            rows = read_jsonl(path)
            mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for row in rows:
                for key in _skill_row_keys(row):
                    mapping[key] = row
            skill_sets[(method, teacher_backend)] = mapping
    if not skill_sets:
        raise SystemExit(f"no generated skills found under {root}")
    return skill_sets


def _resolve_server_urls(host: str, base_port: int, n_servers: int) -> List[str]:
    return [f"http://{host}:{base_port + i}" for i in range(n_servers)]


def _check_health(url: str) -> bool:
    try:
        with urlrequest.urlopen(f"{url}/health", timeout=5) as resp:
            return resp.status == 200
    except (urlerror.URLError, TimeoutError, ValueError):
        return False


def _rollout_prompt(
    *,
    server_urls: List[str],
    prompt: str,
    question: str,
    ground_truth: Any,
    args: argparse.Namespace,
) -> List[str]:
    if not server_urls:
        raise ValueError("server_urls is empty")
    base_url = random.choice(server_urls).rstrip("/")
    payload: Dict[str, Any] = {
        "data_records": [{"prompt": prompt, "question": question, "gt": ground_truth or ""}],
        "num_questions": 1,
        "suffix": "preliminary_skill_drift_eval",
        "rollout_n": args.student_rollout_n,
        "max_tokens": args.student_max_tokens,
        "temperature": args.student_temperature,
        "top_p": args.student_top_p,
        "top_k": getattr(args, "student_top_k", 50),
    }
    with httpx.Client(timeout=args.student_timeout) as client:
        resp = client.post(f"{base_url}/rollout", json=payload)
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"No results in rollout response: {data.keys()}")
    responses = (results[0] or {}).get("responses") or []
    return [str(x) for x in responses]


class RolloutServerManager:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.script = Path(args.rollout_start_script).resolve()

    def start(self, checkpoint: Dict[str, Any]) -> subprocess.Popen[str]:
        log_dir = Path(self.args.rollout_log_root) / checkpoint["checkpoint_name"]
        log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.repo_root / 'Skill_Evo'}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env["SE_WORKING_DIR"] = str(self.repo_root / "Skill_Evo")
        env["SE_PROJECT_NAME"] = "Skill_Evo"
        env["SE_GPU_IDS"] = self.args.gpu_ids
        env["SE_N_GPUS"] = str(self.args.n_gpus)
        env["SE_ROLLOUT_HOST"] = self.args.rollout_host
        env["SE_ROLLOUT_BASE_PORT"] = str(self.args.rollout_base_port)
        env["SE_ROLLOUT_N_SERVERS"] = str(self.args.n_gpus)
        env["SE_ROLLOUT_LOG_DIR"] = str(log_dir)
        env["SE_ROLLOUT_MODEL"] = checkpoint["checkpoint_path"]
        proc = subprocess.Popen(
            ["bash", str(self.script), "--model", checkpoint["checkpoint_path"]],
            cwd=str(self.repo_root / "Skill_Evo"),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        urls = _resolve_server_urls(self.args.rollout_host, self.args.rollout_base_port, self.args.n_gpus)
        deadline = time.time() + self.args.rollout_health_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"rollout server exited early for {checkpoint['checkpoint_name']}")
            if all(_check_health(url) for url in urls):
                return proc
            time.sleep(2)
        self.stop(proc)
        raise RuntimeError(f"rollout health check timed out for {checkpoint['checkpoint_name']}")

    def stop(self, proc: subprocess.Popen[str]) -> None:
        urls = _resolve_server_urls(self.args.rollout_host, self.args.rollout_base_port, self.args.n_gpus)
        pgid: Optional[int]
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        subprocess.run(["pkill", "python"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.time() + 30
        while time.time() < deadline:
            if not any(_check_health(url) for url in urls):
                break
            time.sleep(1)
        time.sleep(3)


def _run_prompt_rollout(
    *,
    server_urls: List[str],
    prompt: str,
    question: str,
    ground_truth: Any,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], float, int]:
    responses = _rollout_prompt(
        server_urls=server_urls,
        prompt=prompt,
        question=question,
        ground_truth=ground_truth,
        args=args,
    )
    rows: List[Dict[str, Any]] = []
    correct = 0
    for attempt_idx, text in enumerate(responses):
        is_correct = grade_if_possible(text, ground_truth)
        if is_correct is True:
            correct += 1
        rows.append(
            {
                "attempt_idx": attempt_idx,
                "student_response": text,
                "is_correct": is_correct,
            }
        )
    acc = (correct / len(responses)) if responses else 0.0
    return rows, acc, correct


def _question_prompt(question: Dict[str, Any], solve_template: str, skill_text: str) -> str:
    meta = question["meta"]
    return solve_template.format(
        skill=skill_text,
        retrieved_context=skill_text,
        question=meta["problem"],
    )


def _baseline_prompt(question: Dict[str, Any]) -> str:
    return f"Please reason step by step, and put your final answer within \\boxed{{}}.\nQuestion: {question['meta']['problem']}"


def summarize_details(details: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in details:
        buckets[(row["method"], row["teacher_backend"])].append(row)
    out: List[Dict[str, Any]] = []
    for (method, teacher_backend), rows in sorted(buckets.items()):
        evaluated = [r for r in rows if r["skill_acc"] is not None]
        baseline_acc = (
            sum(float(r["baseline_acc"]) for r in evaluated) / len(evaluated)
            if evaluated
            else None
        )
        skill_acc = (
            sum(float(r["skill_acc"]) for r in evaluated) / len(evaluated)
            if evaluated
            else None
        )
        abs_delta = (skill_acc - baseline_acc) if (skill_acc is not None and baseline_acc is not None) else None
        rel_delta = (abs_delta / baseline_acc) if (abs_delta is not None and baseline_acc not in (None, 0.0)) else None
        sample = rows[0]
        out.append(
            {
                "checkpoint_name": sample["checkpoint_name"],
                "checkpoint_path": sample["checkpoint_path"],
                "checkpoint_order": sample["checkpoint_order"],
                "method": method,
                "teacher_backend": teacher_backend,
                "evaluated_questions": len(evaluated),
                "baseline_acc": baseline_acc,
                "skill_acc": skill_acc,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
                "improved": sum(1 for r in evaluated if float(r["delta"]) > 0),
                "degraded": sum(1 for r in evaluated if float(r["delta"]) < 0),
                "unchanged": sum(1 for r in evaluated if float(r["delta"]) == 0),
            }
        )
    return out


def _write_summary(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def _canonical_checkpoint_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def _read_json_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = [data]
    else:
        raise RuntimeError(f"unsupported JSON structure in {path}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"expected object at {path}[{idx}]")
        out.append(row)
    return out


def _first_existing_path(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def _load_resumable_outputs(output_dir: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    per_checkpoint_root = output_dir / "per_checkpoint"
    if not per_checkpoint_root.is_dir():
        return {}
    resumable: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for child in sorted(per_checkpoint_root.iterdir()):
        if not child.is_dir():
            continue
        summary_path = _first_existing_path([child / "summary.json"])
        details_path = _first_existing_path([child / "details.jsonl", child / "details.json"])
        attempts_path = _first_existing_path([child / "attempts.jsonl", child / "attempts.json"])
        if summary_path is None or details_path is None:
            continue
        summary_rows = _read_json_rows(summary_path)
        details_rows = _read_json_rows(details_path)
        attempts_rows = _read_json_rows(attempts_path) if attempts_path is not None else []
        checkpoint_path = ""
        for row in summary_rows + details_rows + attempts_rows:
            text = str(row.get("checkpoint_path") or "").strip()
            if text:
                checkpoint_path = _canonical_checkpoint_path(text)
                break
        if not checkpoint_path:
            continue
        resumable[checkpoint_path] = {
            "summary": summary_rows,
            "details": details_rows,
            "attempts": attempts_rows,
        }
    return resumable


def _normalized_checkpoint_rows(
    rows: Sequence[Dict[str, Any]],
    checkpoint: Dict[str, Any],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["checkpoint_name"] = checkpoint["checkpoint_name"]
        item["checkpoint_path"] = checkpoint["checkpoint_path"]
        item["checkpoint_order"] = checkpoint["checkpoint_order"]
        normalized.append(item)
    return normalized


def _default_eval_workers(*, n_gpus: int, requested: int) -> int:
    if requested > 0:
        return requested
    return max(1, int(n_gpus))


def evaluate_checkpoint(
    *,
    checkpoint: Dict[str, Any],
    questions: Sequence[Dict[str, Any]],
    skill_sets: Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, Any]]],
    args: argparse.Namespace,
    solve_template: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    manager = RolloutServerManager(args)
    proc = manager.start(checkpoint)
    try:
        server_urls = _resolve_server_urls(args.rollout_host, args.rollout_base_port, args.n_gpus)
        eval_workers = _default_eval_workers(
            n_gpus=int(args.n_gpus),
            requested=int(getattr(args, "eval_max_workers", 0) or 0),
        )
        baseline_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        question_count = len(questions)

        def _baseline_task(item: Tuple[int, Dict[str, Any]]) -> Tuple[Tuple[str, str], Dict[str, Any]]:
            _, question = item
            key = _question_lookup_key(question)
            baseline_rows, baseline_acc, baseline_correct = _run_prompt_rollout(
                server_urls=server_urls,
                prompt=_baseline_prompt(question),
                question=question["meta"]["problem"],
                ground_truth=question["meta"]["ground_truth"],
                args=args,
            )
            return key, {
                "baseline_rows": baseline_rows,
                "baseline_acc": baseline_acc,
                "baseline_correct_count": baseline_correct,
                "baseline_rollout_count": len(baseline_rows),
            }

        with _ProgressBar(total=question_count, desc=f"{checkpoint['checkpoint_name']} baseline", leave=False) as progress:
            with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as executor:
                futures = [executor.submit(_baseline_task, item) for item in enumerate(questions)]
                for future in concurrent.futures.as_completed(futures):
                    key, baseline = future.result()
                    baseline_results[key] = baseline
                    progress.update(1)

        details: List[Dict[str, Any]] = []
        attempt_rows: List[Dict[str, Any]] = []
        for (method, teacher_backend), mapping in sorted(skill_sets.items()):
            ordered_results: List[Tuple[int, Dict[str, Any], List[Dict[str, Any]]]] = []

            def _skill_task(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
                question_idx, question = item
                meta = question["meta"]
                qkey = _question_lookup_key(question)
                baseline = baseline_results[qkey]
                skill_row = mapping.get(qkey)
                if skill_row is None and qkey[0] == "source_idx":
                    skill_row = mapping.get(("problem", meta["problem"]))
                if skill_row is None:
                    return question_idx, {
                        "checkpoint_name": checkpoint["checkpoint_name"],
                        "checkpoint_path": checkpoint["checkpoint_path"],
                        "checkpoint_order": checkpoint["checkpoint_order"],
                        "source_idx": meta["source_idx"],
                        "problem": meta["problem"],
                        "method": method,
                        "teacher_backend": teacher_backend,
                        "baseline_acc": baseline["baseline_acc"],
                        "skill_acc": None,
                        "delta": None,
                        "skip_reason": "missing_skill",
                    }, []
                if skill_row.get("status") != "ok":
                    return question_idx, {
                        "checkpoint_name": checkpoint["checkpoint_name"],
                        "checkpoint_path": checkpoint["checkpoint_path"],
                        "checkpoint_order": checkpoint["checkpoint_order"],
                        "source_idx": meta["source_idx"],
                        "problem": meta["problem"],
                        "method": method,
                        "teacher_backend": teacher_backend,
                        "baseline_acc": baseline["baseline_acc"],
                        "skill_acc": None,
                        "delta": None,
                        "skip_reason": str(skill_row.get("status") or "invalid_skill"),
                    }, []
                try:
                    prompt = _question_prompt(question, solve_template, str(skill_row.get("skill_text") or ""))
                    rows, skill_acc, skill_correct = _run_prompt_rollout(
                        server_urls=server_urls,
                        prompt=prompt,
                        question=meta["problem"],
                        ground_truth=meta["ground_truth"],
                        args=args,
                    )
                except Exception as e:
                    return question_idx, {
                        "checkpoint_name": checkpoint["checkpoint_name"],
                        "checkpoint_path": checkpoint["checkpoint_path"],
                        "checkpoint_order": checkpoint["checkpoint_order"],
                        "source_idx": meta["source_idx"],
                        "problem": meta["problem"],
                        "method": method,
                        "teacher_backend": teacher_backend,
                        "baseline_acc": baseline["baseline_acc"],
                        "skill_acc": None,
                        "delta": None,
                        "skip_reason": f"student_error:{type(e).__name__}",
                    }, []
                attempt_batch: List[Dict[str, Any]] = []
                for row in rows:
                    attempt_batch.append(
                        {
                            "checkpoint_name": checkpoint["checkpoint_name"],
                            "checkpoint_path": checkpoint["checkpoint_path"],
                            "checkpoint_order": checkpoint["checkpoint_order"],
                            "source_idx": meta["source_idx"],
                            "problem": meta["problem"],
                            "method": method,
                            "teacher_backend": teacher_backend,
                            "prompt": prompt,
                            "ground_truth": meta["ground_truth"],
                            **row,
                        }
                    )
                return question_idx, {
                    "checkpoint_name": checkpoint["checkpoint_name"],
                    "checkpoint_path": checkpoint["checkpoint_path"],
                    "checkpoint_order": checkpoint["checkpoint_order"],
                    "source_idx": meta["source_idx"],
                    "problem": meta["problem"],
                    "method": method,
                    "teacher_backend": teacher_backend,
                    "baseline_acc": baseline["baseline_acc"],
                    "baseline_correct_count": baseline["baseline_correct_count"],
                    "baseline_rollout_count": baseline["baseline_rollout_count"],
                    "skill_acc": skill_acc,
                    "skill_correct_count": skill_correct,
                    "skill_rollout_count": len(rows),
                    "delta": skill_acc - baseline["baseline_acc"],
                    "skip_reason": "",
                }, attempt_batch

            desc = f"{checkpoint['checkpoint_name']} {method}/{teacher_backend}"
            with _ProgressBar(total=question_count, desc=desc, leave=False) as progress:
                with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as executor:
                    futures = [executor.submit(_skill_task, item) for item in enumerate(questions)]
                    for future in concurrent.futures.as_completed(futures):
                        ordered_results.append(future.result())
                        progress.update(1)

            ordered_results.sort(key=lambda item: item[0])
            for _, detail, attempt_batch in ordered_results:
                details.append(detail)
                attempt_rows.extend(attempt_batch)
        return details, attempt_rows
    finally:
        manager.stop(proc)


def run_eval(args: argparse.Namespace) -> int:
    methods = _parse_choice_filter(args.methods, METHODS)
    teacher_backends = _parse_choice_filter(args.teacher_backends, TEACHER_BACKENDS)
    trajectories = read_jsonl(args.trajectories)
    questions = group_questions(trajectories, args.sample_size)
    checkpoints = discover_checkpoints(args.checkpoint_root, args.checkpoint_limit)
    skill_sets = load_skill_sets(args.skills_run_dir, methods, teacher_backends)
    solve_template = _load_solve_template()
    output_dir = Path(args.output_dir)
    all_details: List[Dict[str, Any]] = []
    all_attempts: List[Dict[str, Any]] = []
    cross_summary: List[Dict[str, Any]] = []
    resumable_outputs = _load_resumable_outputs(output_dir) if getattr(args, "resume", False) else {}

    with _ProgressBar(total=len(checkpoints), desc="checkpoints") as checkpoint_progress:
        for checkpoint in checkpoints:
            checkpoint_path = _canonical_checkpoint_path(str(checkpoint["checkpoint_path"]))
            existing = resumable_outputs.get(checkpoint_path)
            if existing is not None:
                checkpoint_progress.write(f"[resume] reuse {checkpoint['checkpoint_name']}")
                details = _normalized_checkpoint_rows(existing["details"], checkpoint)
                attempts = _normalized_checkpoint_rows(existing["attempts"], checkpoint)
                summary = _normalized_checkpoint_rows(existing["summary"], checkpoint)
                ckpt_dir = output_dir / "per_checkpoint" / checkpoint["checkpoint_name"]
                write_jsonl(ckpt_dir / "details.jsonl", details)
                write_jsonl(ckpt_dir / "attempts.jsonl", attempts)
                _write_summary(ckpt_dir / "summary.json", summary)
                all_details.extend(details)
                all_attempts.extend(attempts)
                cross_summary.extend(summary)
                checkpoint_progress.update(1)
                continue
            checkpoint_progress.write(f"[run] {checkpoint['checkpoint_name']}")
            details, attempts = evaluate_checkpoint(
                checkpoint=checkpoint,
                questions=questions,
                skill_sets=skill_sets,
                args=args,
                solve_template=solve_template,
            )
            ckpt_dir = output_dir / "per_checkpoint" / checkpoint["checkpoint_name"]
            write_jsonl(ckpt_dir / "details.jsonl", details)
            write_jsonl(ckpt_dir / "attempts.jsonl", attempts)
            summary = summarize_details(details)
            _write_summary(ckpt_dir / "summary.json", summary)
            all_details.extend(details)
            all_attempts.extend(attempts)
            cross_summary.extend(summary)
            checkpoint_progress.update(1)

    write_jsonl(output_dir / "cross_checkpoint_details.jsonl", all_details)
    write_jsonl(output_dir / "cross_checkpoint_attempts.jsonl", all_attempts)
    _write_summary(output_dir / "cross_checkpoint_summary.json", cross_summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track skill effectiveness drift across student checkpoints")
    parser.add_argument("--skills-run-dir", required=True, help="output dir from eval_source_linked_skills")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trajectories", default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--teacher-backends", nargs="*", default=None)
    parser.add_argument("--student-rollout-n", type=int, default=4)
    parser.add_argument("--student-temperature", type=float, default=0.7)
    parser.add_argument("--student-top-p", type=float, default=0.95)
    parser.add_argument("--student-top-k", type=int, default=50)
    parser.add_argument("--student-max-tokens", type=int, default=4096)
    parser.add_argument("--student-timeout", type=float, default=600.0)
    parser.add_argument("--student-max-retries", type=int, default=3)
    parser.add_argument("--student-max-concurrent", type=int, default=0)
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--checkpoint-limit", type=int, default=0)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--rollout-start-script", default=str(Path(__file__).resolve().parents[2] / "skill_src" / "Zero" / "start_rollout_servers.sh"))
    parser.add_argument("--rollout-log-root", default=str(Path(__file__).resolve().parent / "outputs" / "skill_drift_across_checkpoints" / "logs" / "rollout_servers"))
    parser.add_argument("--gpu-ids", default=os.environ.get("SE_GPU_IDS", "4,5,6,7"))
    parser.add_argument("--n-gpus", type=int, default=int(os.environ.get("SE_N_GPUS", "4")))
    parser.add_argument("--rollout-host", default=os.environ.get("SE_ROLLOUT_HOST", "127.0.0.1"))
    parser.add_argument("--rollout-base-port", type=int, default=int(os.environ.get("SE_ROLLOUT_BASE_PORT", "8760")))
    parser.add_argument("--rollout-health-timeout", type=int, default=240)
    parser.add_argument("--eval-max-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.student_rollout_n <= 0:
        raise SystemExit("--student-rollout-n must be > 0")
    if args.n_gpus <= 0:
        raise SystemExit("--n-gpus must be > 0")
    return int(run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
