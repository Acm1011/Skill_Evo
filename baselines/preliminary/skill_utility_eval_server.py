from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from baselines.ReasoningBankMath.io_utils import read_jsonl, write_jsonl
from baselines.preliminary.eval_skill_drift_across_checkpoints import (
    METHODS,
    TEACHER_BACKENDS,
    _load_solve_template,
    _parse_choice_filter,
    evaluate_checkpoint,
    load_skill_sets,
    summarize_details,
)
from baselines.preliminary.eval_source_linked_skills import group_questions

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [skill_utility_eval_server] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False


def select_questions(
    trajectories: Sequence[Dict[str, Any]],
    *,
    sample_size: int,
    selection: str,
) -> List[Dict[str, Any]]:
    grouped = group_questions(trajectories, len(trajectories))
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if selection not in {"head", "tail"}:
        raise ValueError(f"unsupported selection: {selection}")
    if selection == "tail":
        return grouped[-sample_size:]
    return grouped[:sample_size]


def _checkpoint_name(global_step: int, checkpoint_path: str) -> str:
    if global_step > 0:
        return f"global_step_{global_step}"
    path = Path(checkpoint_path)
    if path.name == "actor" and path.parent.name:
        return path.parent.name
    return path.name or "checkpoint"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class JobState:
    checkpoint_path: str
    global_step: int
    checkpoint_name: str
    status: str
    created_at: float
    updated_at: float
    output_dir: str
    error: str = ""
    metrics_path: str = ""
    details_path: str = ""
    attempts_path: str = ""
    summary: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TriggerRequest:
    checkpoint_path: str
    global_step: int = 0
    checkpoint_name: str = ""
    force: bool = False

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TriggerRequest":
        checkpoint_path = str(payload.get("checkpoint_path") or "").strip()
        if not checkpoint_path:
            raise ValueError("missing checkpoint_path")
        global_step = int(payload.get("global_step") or 0)
        checkpoint_name = str(payload.get("checkpoint_name") or "").strip()
        force = bool(payload.get("force", False))
        return cls(
            checkpoint_path=checkpoint_path,
            global_step=global_step,
            checkpoint_name=checkpoint_name,
            force=force,
        )


class SkillUtilityEvalService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        methods = _parse_choice_filter(args.methods, METHODS)
        teacher_backends = _parse_choice_filter(args.teacher_backends, TEACHER_BACKENDS)

        trajectories = read_jsonl(args.trajectories)
        self.questions = select_questions(
            trajectories,
            sample_size=args.sample_size,
            selection=args.question_selection,
        )
        self.skill_sets = load_skill_sets(args.skills_run_dir, methods, teacher_backends)
        self.solve_template = _load_solve_template()
        self.output_root = Path(args.output_dir).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_root / "server_state.json"
        self.queue: queue.Queue[JobState] = queue.Queue()
        self.lock = threading.Lock()
        self.jobs: Dict[str, JobState] = {}
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def enqueue(self, request: TriggerRequest) -> JobState:
        checkpoint_path = str(Path(request.checkpoint_path).resolve())
        if not Path(checkpoint_path).exists():
            raise ValueError(f"checkpoint_path does not exist: {checkpoint_path}")
        checkpoint_name = str(request.checkpoint_name or "").strip() or _checkpoint_name(request.global_step, checkpoint_path)
        output_dir = self.output_root / "per_checkpoint" / checkpoint_name
        now = time.time()
        with self.lock:
            existing = self.jobs.get(checkpoint_path)
            if existing and existing.status in {"queued", "running"} and not request.force:
                return existing
            if existing and existing.status == "done" and not request.force:
                return existing
            job = JobState(
                checkpoint_path=checkpoint_path,
                global_step=int(request.global_step),
                checkpoint_name=checkpoint_name,
                status="queued",
                created_at=now,
                updated_at=now,
                output_dir=str(output_dir),
            )
            self.jobs[checkpoint_path] = job
            self._persist_state_locked()
        self.queue.put(job)
        logger.info("queued evaluation for %s", checkpoint_name)
        return job

    def status(self) -> Dict[str, Any]:
        with self.lock:
            jobs = [asdict(job) for job in sorted(self.jobs.values(), key=lambda item: item.created_at)]
        return {
            "ok": True,
            "pending": self.queue.qsize(),
            "jobs": jobs,
            "sample_size": len(self.questions),
            "question_selection": self.args.question_selection,
        }

    def _persist_state_locked(self) -> None:
        payload = {
            "updated_at": time.time(),
            "pending": self.queue.qsize(),
            "jobs": [asdict(job) for job in sorted(self.jobs.values(), key=lambda item: item.created_at)],
        }
        _write_json(self.state_path, payload)

    def _build_eval_args(self) -> argparse.Namespace:
        return SimpleNamespace(
            student_rollout_n=self.args.student_rollout_n,
            student_temperature=self.args.student_temperature,
            student_top_p=self.args.student_top_p,
            student_top_k=self.args.student_top_k,
            student_max_tokens=self.args.student_max_tokens,
            student_timeout=self.args.student_timeout,
            student_max_retries=self.args.student_max_retries,
            served_model_name="",
            checkpoint_limit=0,
            repo_root=self.args.repo_root,
            rollout_start_script=self.args.rollout_start_script,
            rollout_log_root=self.args.rollout_log_root,
            gpu_ids=self.args.gpu_ids,
            n_gpus=self.args.n_gpus,
            rollout_host=self.args.rollout_host,
            rollout_base_port=self.args.rollout_base_port,
            rollout_health_timeout=self.args.rollout_health_timeout,
            eval_max_workers=self.args.eval_max_workers,
            resume=False,
            methods=self.args.methods,
            teacher_backends=self.args.teacher_backends,
        )

    def _worker_loop(self) -> None:
        while True:
            job = self.queue.get()
            try:
                self._run_job(job)
            except Exception as exc:
                logger.exception("evaluation failed for %s", job.checkpoint_name)
                with self.lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.updated_at = time.time()
                    self._persist_state_locked()
            finally:
                self.queue.task_done()

    def _run_job(self, job: JobState) -> None:
        with self.lock:
            job.status = "running"
            job.error = ""
            job.updated_at = time.time()
            self._persist_state_locked()

        checkpoint = {
            "checkpoint_name": job.checkpoint_name,
            "checkpoint_path": job.checkpoint_path,
            "checkpoint_order": int(job.global_step),
        }
        details, attempts = evaluate_checkpoint(
            checkpoint=checkpoint,
            questions=self.questions,
            skill_sets=self.skill_sets,
            args=self._build_eval_args(),
            solve_template=self.solve_template,
        )
        output_dir = Path(job.output_dir)
        details_path = output_dir / "details.jsonl"
        attempts_path = output_dir / "attempts.jsonl"
        summary_path = output_dir / "summary.json"
        write_jsonl(details_path, details)
        write_jsonl(attempts_path, attempts)
        summary = summarize_details(details)
        _write_json(summary_path, summary)
        self._refresh_aggregate_outputs()

        with self.lock:
            job.status = "done"
            job.updated_at = time.time()
            job.details_path = str(details_path)
            job.attempts_path = str(attempts_path)
            job.metrics_path = str(summary_path)
            job.summary = summary
            self._persist_state_locked()
        logger.info("evaluation finished for %s", job.checkpoint_name)

    def _refresh_aggregate_outputs(self) -> None:
        all_details: List[Dict[str, Any]] = []
        all_attempts: List[Dict[str, Any]] = []
        all_summaries: List[Dict[str, Any]] = []
        per_checkpoint_root = self.output_root / "per_checkpoint"
        if not per_checkpoint_root.is_dir():
            return
        for child in sorted(per_checkpoint_root.iterdir()):
            if not child.is_dir():
                continue
            details_path = child / "details.jsonl"
            attempts_path = child / "attempts.jsonl"
            summary_path = child / "summary.json"
            if details_path.is_file():
                all_details.extend(read_jsonl(details_path))
            if attempts_path.is_file():
                all_attempts.extend(read_jsonl(attempts_path))
            if summary_path.is_file():
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    all_summaries.extend(data)
        write_jsonl(self.output_root / "cross_checkpoint_details.jsonl", all_details)
        write_jsonl(self.output_root / "cross_checkpoint_attempts.jsonl", all_attempts)
        _write_json(self.output_root / "cross_checkpoint_summary.json", all_summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async server for checkpoint-wise skill utility evaluation")
    parser.add_argument("--skills-run-dir", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--question-selection", choices=("head", "tail"), default="tail")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--teacher-backends", nargs="*", default=["api_teacher", "server_teacher"])
    parser.add_argument("--student-rollout-n", type=int, default=4)
    parser.add_argument("--student-temperature", type=float, default=0.7)
    parser.add_argument("--student-top-p", type=float, default=0.95)
    parser.add_argument("--student-top-k", type=int, default=50)
    parser.add_argument("--student-max-tokens", type=int, default=4096)
    parser.add_argument("--student-timeout", type=float, default=600.0)
    parser.add_argument("--student-max-retries", type=int, default=3)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--rollout-start-script",
        default=str(Path(__file__).resolve().parents[2] / "skill_src" / "Zero" / "start_rollout_servers.sh"),
    )
    parser.add_argument("--rollout-log-root", required=True)
    parser.add_argument("--gpu-ids", default=os.environ.get("SE_GPU_IDS", "2,3"))
    parser.add_argument("--n-gpus", type=int, default=int(os.environ.get("SE_N_GPUS", "2")))
    parser.add_argument("--rollout-host", default=os.environ.get("SE_ROLLOUT_HOST", "127.0.0.1"))
    parser.add_argument("--rollout-base-port", type=int, default=int(os.environ.get("SE_ROLLOUT_BASE_PORT", "8760")))
    parser.add_argument("--rollout-health-timeout", type=int, default=240)
    parser.add_argument("--eval-max-workers", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    return parser


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def create_handler(service: SkillUtilityEvalService):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    {"ok": True, "pending": service.queue.qsize(), "jobs": len(service.jobs)},
                )
                return
            if self.path == "/status":
                _json_response(self, HTTPStatus.OK, service.status())
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/enqueue":
                _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                request = TriggerRequest.from_payload(payload)
                job = service.enqueue(request)
                _json_response(self, HTTPStatus.OK, {"ok": True, "job": asdict(job)})
            except ValueError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except Exception as exc:
                logger.exception("enqueue failed")
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

    return Handler


def run_server(args: argparse.Namespace) -> int:
    _configure_logging()
    service = SkillUtilityEvalService(args)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(service))
    logger.info("serving on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("received keyboard interrupt, shutting down")
    finally:
        server.server_close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
