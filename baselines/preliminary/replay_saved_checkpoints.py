from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from baselines.preliminary.eval_skill_drift_across_checkpoints import discover_checkpoints


def _has_any(path: Path, patterns: Sequence[str]) -> bool:
    return any(any(path.glob(pattern)) for pattern in patterns)


def _is_hf_model_dir(path: Path) -> bool:
    marker = any((path / name).is_file() for name in ("config.json", "generation_config.json", "adapter_config.json"))
    weights = _has_any(path, ("*.safetensors", "pytorch_model*.bin", "model*.safetensors"))
    return marker and weights


def _is_hf_assets_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    required = ("config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json")
    return any((path / name).is_file() for name in required)


def _is_fsdp_actor_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_shards = _has_any(path, ("model_world_size_*_rank_*.pt",))
    has_hf_assets = _is_hf_assets_dir(path / "huggingface")
    return has_shards and has_hf_assets


def discover_eval_checkpoints(root: str | Path, limit: int = 0) -> List[Dict[str, Any]]:
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise SystemExit(f"checkpoint root does not exist: {root_path}")

    items: List[Dict[str, Any]] = []
    for item in discover_checkpoints(root_path, limit=0):
        ckpt_path = Path(str(item["checkpoint_path"]))
        if _is_hf_model_dir(ckpt_path):
            items.append(item)

    seen = {str(Path(str(item["checkpoint_path"])).resolve()) for item in items}
    for actor_dir in sorted(root_path.rglob("actor")):
        actor_dir = actor_dir.resolve()
        if str(actor_dir) in seen:
            continue
        if not _is_fsdp_actor_dir(actor_dir):
            continue
        parent = actor_dir.parent
        step = parent.name if parent.name else actor_dir.name
        items.append(
            {
                "checkpoint_path": str(actor_dir),
                "checkpoint_name": step,
                "_sort_key": _sort_key_from_name(step),
            }
        )
        seen.add(str(actor_dir))

    items.sort(key=lambda item: (item.get("_sort_key", (10**18, item["checkpoint_name"])), str(item["checkpoint_path"])))
    for idx, item in enumerate(items):
        item["checkpoint_order"] = idx
    if limit > 0:
        return items[:limit]
    return items


def _sort_key_from_name(name: str) -> tuple[int, str]:
    import re

    patterns = [
        r"checkpoint[-_]?(\d+)",
        r"global[_-]?step[_-]?(\d+)",
        r"step[_-]?(\d+)",
        r"epoch[_-]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), name
    return 10**18, name


def _read_json(url: str, *, timeout: float) -> Dict[str, Any]:
    with urlrequest.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _post_json(url: str, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _extract_step(item: Dict[str, Any]) -> int:
    sort_key = item.get("_sort_key")
    if isinstance(sort_key, (list, tuple)) and sort_key:
        try:
            return int(sort_key[0])
        except (TypeError, ValueError):
            return -1
    return -1


def _merged_model_path(merged_root: Path, checkpoint_name: str) -> Path:
    return merged_root / checkpoint_name


def ensure_loadable_checkpoint(
    item: Dict[str, Any],
    *,
    merged_root: Optional[Path],
    merge_timeout: float,
) -> Dict[str, Any]:
    checkpoint_path = Path(str(item["checkpoint_path"])).resolve()
    if _is_hf_model_dir(checkpoint_path):
        return dict(item, checkpoint_path=str(checkpoint_path))

    if not _is_fsdp_actor_dir(checkpoint_path):
        raise RuntimeError(f"checkpoint is not a loadable HF dir or supported FSDP actor dir: {checkpoint_path}")

    if merged_root is None:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} is an FSDP actor dir and needs merging; "
            "set --merged-root or MERGED_ROOT"
        )

    target_dir = _merged_model_path(merged_root, str(item["checkpoint_name"]))
    if _is_hf_model_dir(target_dir):
        return dict(item, checkpoint_path=str(target_dir))

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint_path),
        "--target_dir",
        str(target_dir),
    ]
    subprocess.run(
        cmd,
        check=True,
        timeout=None if merge_timeout <= 0 else merge_timeout,
        env={**os.environ},
    )
    if not _is_hf_model_dir(target_dir):
        raise RuntimeError(f"merged checkpoint is still not loadable: {target_dir}")
    return dict(item, checkpoint_path=str(target_dir))


def _summary_path(output_dir: Path, checkpoint_name: str) -> Path:
    return output_dir / "per_checkpoint" / checkpoint_name / "summary.json"


def filter_checkpoints(
    checkpoints: Sequence[Dict[str, Any]],
    *,
    output_dir: Optional[Path],
    min_step: int,
    max_step: int,
    force: bool,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for item in checkpoints:
        step = _extract_step(item)
        if min_step >= 0 and step >= 0 and step < min_step:
            continue
        if max_step >= 0 and step >= 0 and step > max_step:
            continue
        if output_dir is not None and not force:
            if _summary_path(output_dir, str(item["checkpoint_name"])).is_file():
                continue
        selected.append(item)
    return selected


def wait_until_done(server_url: str, checkpoint_names: Sequence[str], *, poll_interval: float, timeout: float) -> None:
    deadline = time.time() + timeout if timeout > 0 else None
    target = set(checkpoint_names)
    while True:
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"timed out waiting for checkpoints: {sorted(target)}")
        status = _read_json(f"{server_url.rstrip('/')}/status", timeout=max(5.0, poll_interval + 1.0))
        jobs = {str(job.get("checkpoint_name") or ""): job for job in status.get("jobs") or []}
        done = True
        failed: List[str] = []
        for name in target:
            job = jobs.get(name)
            if job is None:
                done = False
                continue
            state = str(job.get("status") or "")
            if state == "failed":
                failed.append(f"{name}: {job.get('error') or 'unknown error'}")
            elif state != "done":
                done = False
        if failed:
            raise RuntimeError("some checkpoints failed: " + "; ".join(failed))
        if done:
            return
        time.sleep(poll_interval)


def run(args: argparse.Namespace) -> int:
    server_url = args.server_url.rstrip("/")
    _read_json(f"{server_url}/health", timeout=args.request_timeout)

    checkpoints = discover_eval_checkpoints(args.checkpoint_root, args.checkpoint_limit)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    merged_root = Path(args.merged_root).resolve() if args.merged_root else None
    selected = filter_checkpoints(
        checkpoints,
        output_dir=output_dir,
        min_step=args.min_step,
        max_step=args.max_step,
        force=args.force,
    )

    if not selected:
        print("no checkpoints selected")
        return 0

    print(f"selected {len(selected)} checkpoints")
    enqueued_names: List[str] = []
    for item in selected:
        item = ensure_loadable_checkpoint(
            item,
            merged_root=merged_root,
            merge_timeout=args.merge_timeout,
        )
        step = _extract_step(item)
        payload = {
            "checkpoint_path": item["checkpoint_path"],
            "checkpoint_name": item["checkpoint_name"],
            "global_step": max(step, 0),
            "force": bool(args.force),
        }
        resp = _post_json(f"{server_url}/enqueue", payload, timeout=args.request_timeout)
        job = resp.get("job") or {}
        name = str(job.get("checkpoint_name") or item["checkpoint_name"])
        enqueued_names.append(name)
        print(f"enqueued {name} -> {item['checkpoint_path']}")

    if args.wait:
        wait_until_done(
            server_url,
            enqueued_names,
            poll_interval=args.poll_interval,
            timeout=args.wait_timeout,
        )
        print("all selected checkpoints finished")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay saved checkpoints through skill utility eval server")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8899")
    parser.add_argument("--output-dir", default="", help="Used to skip checkpoints with existing summary.json")
    parser.add_argument("--merged-root", default="", help="Where to store merged HF models for FSDP checkpoints")
    parser.add_argument("--checkpoint-limit", type=int, default=0)
    parser.add_argument("--min-step", type=int, default=-1)
    parser.add_argument("--max-step", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=0.0, help="0 means wait indefinitely")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--merge-timeout", type=float, default=0.0, help="0 means no timeout for model merging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
