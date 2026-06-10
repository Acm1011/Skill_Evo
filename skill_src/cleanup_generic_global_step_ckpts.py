#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")


@dataclass
class CkptRootPlan:
    ckpt_root: Path
    keep_dir: Path
    delete_dirs: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "递归扫描给定目录，寻找直接包含多个 global_step_* 子目录的 checkpoint 根目录；"
            "每处仅保留步数最大的那个 global_step_*。默认 dry-run。"
        )
    )
    parser.add_argument("root", type=Path, help="要扫描的根目录")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行删除；默认仅打印计划",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印只有一个或零个 global_step_* 的 checkpoint 目录",
    )
    return parser.parse_args()


def find_step_dirs(root: Path) -> list[tuple[int, Path]]:
    step_dirs: list[tuple[int, Path]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = STEP_DIR_RE.match(path.name)
        if not match:
            continue
        step_dirs.append((int(match.group(1)), path))
    return step_dirs


def collect_plans(scan_root: Path) -> list[CkptRootPlan]:
    candidate_roots: dict[Path, None] = {}

    for step_dir in scan_root.rglob("global_step_*"):
        if not step_dir.is_dir():
            continue
        if STEP_DIR_RE.match(step_dir.name):
            candidate_roots[step_dir.parent] = None

    plans: list[CkptRootPlan] = []
    for ckpt_root in sorted(candidate_roots):
        step_dirs = find_step_dirs(ckpt_root)
        if not step_dirs:
            continue

        step_dirs.sort(key=lambda item: item[0])
        keep_dir = step_dirs[-1][1]
        delete_dirs = [path for _, path in step_dirs[:-1]]
        plans.append(
            CkptRootPlan(
                ckpt_root=ckpt_root,
                keep_dir=keep_dir,
                delete_dirs=delete_dirs,
            )
        )

    return plans


def remove_dirs(paths: list[Path]) -> None:
    for path in paths:
        shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        print(f"Error: root 不存在: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"Error: root 不是目录: {root}", file=sys.stderr)
        return 1

    plans = collect_plans(root)
    if not plans:
        print(f"未发现任何包含 global_step_* 的 checkpoint 目录: {root}")
        return 0

    scanned_roots = 0
    cleaned_roots = 0
    deleted_dirs_count = 0

    print("==============================================")
    print("Cleanup generic global_step ckpts")
    print(f"  root:     {root}")
    print(f"  mode:     {'execute' if args.execute else 'dry-run'}")
    print(f"  ckptdirs: {len(plans)}")
    print("==============================================")

    for plan in plans:
        scanned_roots += 1
        rel_root = plan.ckpt_root.relative_to(root) if plan.ckpt_root != root else Path(".")

        if not plan.delete_dirs:
            if args.verbose:
                print()
                print(f"==> {rel_root}")
                print(f"    跳过: 仅有一个 ckpt，保留 {plan.keep_dir.name}")
            continue

        print()
        print(f"==> {rel_root}")
        print(f"    保留: {plan.keep_dir}")
        print(f"    删除目录数: {len(plan.delete_dirs)}")
        for path in plan.delete_dirs:
            print(f"      - {path}")

        if args.execute:
            remove_dirs(plan.delete_dirs)
            cleaned_roots += 1
            deleted_dirs_count += len(plan.delete_dirs)
            print("    已删除")

    print()
    print("==============================================")
    print("完成")
    print(f"  扫描 ckpt 根目录数:   {scanned_roots}")
    print(f"  执行清理目录数:       {cleaned_roots}")
    print(f"  删除 ckpt 目录数:     {deleted_dirs_count}")
    print("==============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
