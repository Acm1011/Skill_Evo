#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")
ROLE_DIRS = ("Synthesizer", "Solver")
VERSION_DIR_RE = re.compile(r"^V\d+$")


@dataclass
class VersionCleanupPlan:
    role: str
    version_dir: Path
    ckpt_root: Path
    keep_dir: Path
    delete_dirs: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "清理主实验结果中的历史 checkpoint。扫描给定根目录下的 "
            "Synthesizer/V*/ckpts 和 Solver/V*/ckpts，仅保留每个版本步数最大的 "
            "global_step_* 目录。默认 dry-run。"
        )
    )
    parser.add_argument("root", type=Path, help="实验结果根目录")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行删除；默认仅打印计划",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印没有可删除 ckpt 的版本",
    )
    return parser.parse_args()


def find_step_dirs(ckpt_root: Path) -> list[tuple[int, Path]]:
    step_dirs: list[tuple[int, Path]] = []
    for path in sorted(ckpt_root.iterdir()):
        if not path.is_dir():
            continue
        match = STEP_DIR_RE.match(path.name)
        if not match:
            continue
        step_dirs.append((int(match.group(1)), path))
    return step_dirs


def collect_cleanup_plans(root: Path) -> list[VersionCleanupPlan]:
    plans: list[VersionCleanupPlan] = []

    for role in ROLE_DIRS:
        role_dir = root / role
        if not role_dir.is_dir():
            continue

        for version_dir in sorted(role_dir.iterdir()):
            if not version_dir.is_dir() or not VERSION_DIR_RE.match(version_dir.name):
                continue

            ckpt_root = version_dir / "ckpts"
            if not ckpt_root.is_dir():
                continue

            step_dirs = find_step_dirs(ckpt_root)
            if len(step_dirs) <= 1:
                if len(step_dirs) == 1:
                    keep_dir = step_dirs[0][1]
                else:
                    keep_dir = ckpt_root
                plans.append(
                    VersionCleanupPlan(
                        role=role,
                        version_dir=version_dir,
                        ckpt_root=ckpt_root,
                        keep_dir=keep_dir,
                        delete_dirs=[],
                    )
                )
                continue

            step_dirs.sort(key=lambda item: item[0])
            keep_dir = step_dirs[-1][1]
            delete_dirs = [path for _, path in step_dirs[:-1]]
            plans.append(
                VersionCleanupPlan(
                    role=role,
                    version_dir=version_dir,
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

    plans = collect_cleanup_plans(root)
    if not plans:
        print(f"未发现可扫描的版本目录: {root}")
        return 0

    scanned_versions = 0
    cleaned_versions = 0
    deleted_dirs_count = 0

    print("==============================================")
    print("Cleanup experiment ckpts")
    print(f"  root:    {root}")
    print(f"  mode:    {'execute' if args.execute else 'dry-run'}")
    print(f"  versions:{len(plans)}")
    print("==============================================")

    for plan in plans:
        scanned_versions += 1
        rel_version = plan.version_dir.relative_to(root)

        if not plan.delete_dirs:
            if args.verbose:
                print()
                print(f"==> {rel_version}")
                if plan.keep_dir == plan.ckpt_root:
                    print("    跳过: 未发现 global_step_*")
                else:
                    print(f"    跳过: 仅有一个 ckpt，保留 {plan.keep_dir.name}")
            continue

        print()
        print(f"==> {rel_version}")
        print(f"    保留: {plan.keep_dir}")
        print(f"    删除目录数: {len(plan.delete_dirs)}")
        for path in plan.delete_dirs:
            print(f"      - {path}")

        if args.execute:
            remove_dirs(plan.delete_dirs)
            cleaned_versions += 1
            deleted_dirs_count += len(plan.delete_dirs)
            print("    已删除")

    print()
    print("==============================================")
    print("完成")
    print(f"  扫描版本数:         {scanned_versions}")
    print(f"  执行清理版本数:     {cleaned_versions}")
    print(f"  删除 ckpt 目录数:   {deleted_dirs_count}")
    print("==============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
