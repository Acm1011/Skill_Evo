#!/usr/bin/env python3
"""供 memory_func_after_*.sh 调用：Synthesizer 后 ingest+prepare，Solver 后按 reward 更新 utility。"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _add_skill_src_to_path() -> None:
    root = Path(__file__).resolve().parent.parent
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def version_slug(exp_version: str) -> str:
    """V1 / v1 -> v1, V12 -> v12。"""
    v = exp_version.strip()
    m = re.match(r"^v?(\d+)$", v, re.I)
    if m:
        return "v" + m.group(1)
    m = re.match(r"^V(\d+)$", v)
    if m:
        return "v" + m.group(1)
    return "v" + re.sub(r"^v", "", v, flags=re.I).lower() or "1"


def _find_latest_train_reward(solver_dir: Path) -> Path | None:
    """与 ``SolverRewardManager`` 一致：训练写入 ``reward_info/expdata/step_*.jsonl``；
    亦兼容 ``Solver_dapo_ray_trainer._save_step_reward_infos`` 的 ``reward_info/train_data/``。"""
    files: list[Path] = []
    for sub in ("expdata", "train_data"):
        d = solver_dir / "reward_info" / sub
        if d.is_dir():
            files.extend(d.glob("step_*.jsonl"))
    if not files:
        return None

    def _sort_key(path: Path) -> tuple[int, int]:
        m = re.search(r"step_(\d+)", path.name)
        step = int(m.group(1)) if m else -1
        # 同 step 优先 expdata（reward_manager 主路径）
        pref = 0 if path.parent.name == "expdata" else 1
        return (step, -pref)

    return max(files, key=_sort_key)


def _retriever_url() -> str:
    from skill_manager.skill_manager import DEFAULT_RETRIEVER_URL

    return os.environ.get("SE_RETRIEVER_URL", DEFAULT_RETRIEVER_URL)


def cmd_after_sync(args: argparse.Namespace) -> int:
    _add_skill_src_to_path()
    from skill_manager.skill_controller import SkillController
    from skill_manager.skill_manager import SkillManager

    r_url = _retriever_url()

    exp = args.exp_version
    syn_storage = Path(args.synthesizer_path_dir) / exp
    step = int(args.synth_step)
    data_jsonl = Path(args.data_file)
    mem_dir = Path(args.memory_path_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)
    sl = version_slug(exp)

    m_prev = re.match(r"v(\d+)$", sl)
    prev_sol_path: Path | None = None
    if m_prev and int(m_prev.group(1)) > 1:
        prev_n = int(m_prev.group(1)) - 1
        prev_sol_path = mem_dir / f"memory_after_sol_v{prev_n}.jsonl"

    out_mem = mem_dir / f"memory_after_syn_{sl}.jsonl"
    manager = SkillManager(persist_path=out_mem, retriever_url=r_url)
    if prev_sol_path and prev_sol_path.is_file():
        manager.load_jsonl(prev_sol_path)
    ctrl = SkillController(manager)
    try:
        mem_min = float((os.environ.get("SE_MEMORY_MIN_UTILITY") or "0").strip() or 0.0)
    except ValueError:
        mem_min = 0.0
    try:
        ctrl.ingest_skills_from_synth_reward_step(
            syn_storage,
            step,
            assign_id_if_missing=True,
            use_eviction=True,
            memory_min_utility=mem_min,
        )
    except FileNotFoundError as e:
        print(f"[memory_hook] ingest: {e}", file=sys.stderr)
        return 1

    try:
        manager.sync_retriever_doc_cache()
    except Exception as e:
        print(f"[memory_hook] sync_retriever_doc_cache: {e}", file=sys.stderr)
        return 1

    tdf = (getattr(args, "test_data_file", None) or "").strip()
    if tdf:
        test_path = Path(tdf)
        if not test_path.is_file():
            print(f"[memory_hook] after_sync: --test-data-file 不存在: {test_path}", file=sys.stderr)
            return 1

    out_parq = Path(args.solver_path_dir) / exp / "train_data.parquet"
    out_parq.parent.mkdir(parents=True, exist_ok=True)
    from skill_manager.data_cursor_io import (
        DATA_CURSOR_FILENAME,
        read_data_cursor,
        read_jsonl_slice,
        write_data_cursor,
    )

    sbs = int(
        (os.environ.get("SE_SOLVER_BATCH_SIZE") or os.environ.get("solver_batch_size") or "16").strip()
    )
    sol_steps = int(
        (
            os.environ.get("SE_SOLVER_RETRAIN_STEPS")
            or os.environ.get("solver_retrain_steps")
            or "2"
        ).strip()
    )
    train_n = sbs * sol_steps
    sol_dir = Path(args.solver_path_dir) / exp
    cur_path = sol_dir / DATA_CURSOR_FILENAME
    start = read_data_cursor(cur_path)
    try:
        _slice, next_c = read_jsonl_slice(data_jsonl, start, train_n)
    except ValueError as e:
        print(
            f"[memory_hook] after_sync: 无法从原始语料取 {train_n} 条 "
            f"(batch={sbs}×steps={sol_steps}, cursor={start}): {e}",
            file=sys.stderr,
        )
        return 1
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        encoding="utf-8",
        delete=False,
        dir=str(sol_dir),
    ) as tmp:
        for rec in _slice:
            tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path = tmp.name
    try:
        ctrl.prepare_solver_skills(tmp_path, out_parquet=out_parq)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if tdf:
        out_test = sol_dir / "test_data.parquet"
        ctrl.prepare_solver_skills(Path(tdf), out_parquet=out_test)
        print(f"[memory_hook] after_sync: wrote {out_test!s}", file=sys.stderr)

    manager.save_jsonl()
    write_data_cursor(cur_path, next_c)
    print(
        f"[memory_hook] after_sync: 原始语料 slice [{start},{next_c}) 共 {train_n} 条 "
        f"(batch={sbs}×steps={sol_steps}); cursor -> {next_c!s}",
        file=sys.stderr,
    )
    print(f"[memory_hook] after_sync: wrote {out_mem!s}, {out_parq!s}")
    return 0


def cmd_after_solver(args: argparse.Namespace) -> int:
    _add_skill_src_to_path()
    from skill_manager.skill_controller import SkillController
    from skill_manager.skill_manager import SkillManager

    exp = args.exp_version
    mem_dir = Path(args.memory_path_dir)
    sl = version_slug(exp)
    mem_syn = mem_dir / f"memory_after_syn_{sl}.jsonl"
    if not mem_syn.is_file():
        print(f"[memory_hook] missing {mem_syn}", file=sys.stderr)
        return 1
    out_sol = mem_dir / f"memory_after_sol_{sl}.jsonl"

    reward_path: Path | None = None
    rj = getattr(args, "reward_jsonl", None)
    if rj:
        p = Path(rj)
        if p.is_file():
            reward_path = p
    if reward_path is None:
        reward_path = _find_latest_train_reward(Path(args.solver_path_dir) / exp)
    if reward_path is None or not reward_path.is_file():
        print(
            "[memory_hook] no reward jsonl; pass path or run solver to emit reward_info",
            file=sys.stderr,
        )
        return 1

    r_url = _retriever_url()
    manager = SkillManager(persist_path=out_sol, retriever_url=r_url)
    manager.load_jsonl(mem_syn)
    ctrl = SkillController(manager)
    ctrl.update_utilities_from_rewards_jsonl(reward_path, persist=True)
    print(f"[memory_hook] after_solver: {reward_path!s} -> {out_sol!s}")
    return 0


def cmd_prepare_test(args: argparse.Namespace) -> int:
    """仅用当前 memory_after_syn 生成 test_data.parquet，不 ingest、不碰 data_cursor 与 train。"""
    _add_skill_src_to_path()
    from skill_manager.skill_controller import SkillController
    from skill_manager.skill_manager import SkillManager

    exp = args.exp_version
    mem_dir = Path(args.memory_path_dir)
    sl = version_slug(exp)
    mem_syn = mem_dir / f"memory_after_syn_{sl}.jsonl"
    if not mem_syn.is_file():
        print(f"[memory_hook] prepare-test: missing {mem_syn}", file=sys.stderr)
        return 1
    test_path = Path(args.test_data_file)
    if not test_path.is_file():
        print(f"[memory_hook] prepare-test: test file not found: {test_path}", file=sys.stderr)
        return 1
    sol_dir = Path(args.solver_path_dir) / exp
    sol_dir.mkdir(parents=True, exist_ok=True)
    out_test = sol_dir / "test_data.parquet"

    r_url = _retriever_url()
    manager = SkillManager(persist_path=mem_syn, retriever_url=r_url)
    manager.load_jsonl(mem_syn)
    ctrl = SkillController(manager)
    ctrl.prepare_solver_skills(test_path, out_parquet=out_test)
    print(f"[memory_hook] prepare-test: wrote {out_test!s}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("after-sync", help="Synthesizer 后 ingest + prepare solver parquet")
    ps.add_argument("exp_version", help="e.g. V1")
    ps.add_argument("--synth-step", dest="synth_step", default="20")
    ps.add_argument("--synthesizer-path-dir", required=True)
    ps.add_argument("--solver-path-dir", required=True)
    ps.add_argument("--memory-path-dir", required=True)
    ps.add_argument("--data-file", required=True)
    ps.add_argument(
        "--test-data-file",
        dest="test_data_file",
        default="",
        help="可选；若给定则整文件 prepare 为 solver 版本目录下 test_data.parquet（不推进 data_cursor）",
    )

    pso = sub.add_parser("after-solver", help="Solver 后按 reward 更新 utility")
    pso.add_argument("exp_version", help="e.g. V1")
    pso.add_argument(
        "reward_jsonl",
        nargs="?",
        default="",
        help="若省略则在 solver 版本目录下选最新 train_data/step_*.jsonl",
    )
    pso.add_argument("--solver-path-dir", required=True)
    pso.add_argument("--memory-path-dir", required=True)

    ppt = sub.add_parser(
        "prepare-test",
        help="仅根据已有 memory_after_syn 生成 solver 目录下 test_data.parquet（不推进游标、不写 train）",
    )
    ppt.add_argument("exp_version", help="e.g. V1")
    ppt.add_argument("--solver-path-dir", required=True)
    ppt.add_argument("--memory-path-dir", required=True)
    ppt.add_argument(
        "--test-data-file",
        required=True,
        help="test 语料 jsonl 路径",
    )

    args = p.parse_args()
    if args.cmd == "after-sync":
        return cmd_after_sync(args)
    if args.cmd == "prepare-test":
        return cmd_prepare_test(args)
    if args.cmd == "after-solver":
        a = type(
            "NS",
            (),
            {
                "exp_version": args.exp_version,
                "reward_jsonl": (args.reward_jsonl or "").strip() or None,
                "solver_path_dir": args.solver_path_dir,
                "memory_path_dir": args.memory_path_dir,
            },
        )()
        return cmd_after_solver(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
