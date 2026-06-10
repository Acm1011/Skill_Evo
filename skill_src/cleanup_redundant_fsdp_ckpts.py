#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


FSDP_PATTERNS = (
    "model_world_size_*_rank_*.pt",
    "optim_world_size_*_rank_*.pt",
    "extra_state_world_size_*_rank_*.pt",
    "fsdp_config.json",
)

HF_WEIGHT_PATTERNS = (
    "model.safetensors",
    "pytorch_model.bin",
    "model-*.safetensors",
    "pytorch_model-*.bin",
)

HF_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

HF_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)


@dataclass
class ActorCleanupState:
    actor_dir: Path
    hf_dir: Path
    fsdp_files: list[Path]
    hf_complete: bool
    hf_state: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "递归查找含 FSDP ckpt 的 actor 目录；仅在对应 actor/huggingface "
            "导出完整时，删除冗余 FSDP ckpt 文件。默认 dry-run。"
        )
    )
    parser.add_argument("root", type=Path, help="要递归扫描的根目录")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行删除；默认仅打印将要删除的文件",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更多跳过原因",
    )
    return parser.parse_args()


def list_fsdp_files(actor_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in FSDP_PATTERNS:
        files.extend(sorted(actor_dir.glob(pattern)))
    return files


def has_hf_weight_files(hf_dir: Path) -> bool:
    if not hf_dir.is_dir():
        return False

    if (hf_dir / "model.safetensors").is_file() or (hf_dir / "pytorch_model.bin").is_file():
        return True

    has_sharded_weights = any(hf_dir.glob("model-*.safetensors")) or any(hf_dir.glob("pytorch_model-*.bin"))
    has_index = any((hf_dir / index_name).is_file() for index_name in HF_INDEX_FILES)
    return has_sharded_weights and has_index


def has_hf_tokenizer_assets(hf_dir: Path) -> bool:
    return any((hf_dir / name).is_file() for name in HF_TOKENIZER_FILES)


def is_hf_export_complete(hf_dir: Path) -> bool:
    return (
        hf_dir.is_dir()
        and (hf_dir / "config.json").is_file()
        and has_hf_weight_files(hf_dir)
        and has_hf_tokenizer_assets(hf_dir)
    )


def describe_hf_export_state(hf_dir: Path) -> str:
    if not hf_dir.exists():
        return "hf_dir=missing"

    parts = [
        f"config={'ok' if (hf_dir / 'config.json').is_file() else 'missing'}",
        f"weights={'ok' if has_hf_weight_files(hf_dir) else 'missing'}",
        f"tokenizer={'ok' if has_hf_tokenizer_assets(hf_dir) else 'missing'}",
    ]
    return " ".join(parts)


def collect_cleanup_targets(root: Path) -> list[ActorCleanupState]:
    actor_dirs: dict[Path, None] = {}
    for pattern in FSDP_PATTERNS:
        for path in root.rglob(pattern):
            if path.is_file():
                actor_dirs[path.parent] = None

    states: list[ActorCleanupState] = []
    for actor_dir in sorted(actor_dirs):
        hf_dir = actor_dir / "huggingface"
        states.append(
            ActorCleanupState(
                actor_dir=actor_dir,
                hf_dir=hf_dir,
                fsdp_files=list_fsdp_files(actor_dir),
                hf_complete=is_hf_export_complete(hf_dir),
                hf_state=describe_hf_export_state(hf_dir),
            )
        )
    return states


def delete_files(files: list[Path]) -> None:
    for path in files:
        path.unlink()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        print(f"Error: root 不存在: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"Error: root 不是目录: {root}", file=sys.stderr)
        return 1

    states = collect_cleanup_targets(root)
    if not states:
        print(f"未发现任何 FSDP ckpt 文件: {root}")
        return 0

    deleted_actor_count = 0
    deleted_file_count = 0
    skipped_incomplete_count = 0

    print("==============================================")
    print("Cleanup redundant FSDP ckpts")
    print(f"  root:    {root}")
    print(f"  mode:    {'execute' if args.execute else 'dry-run'}")
    print(f"  actors:  {len(states)}")
    print("==============================================")

    for state in states:
        rel_actor = state.actor_dir.relative_to(root) if state.actor_dir != root else Path(".")
        print()
        print(f"==> actor: {rel_actor}")

        if not state.fsdp_files:
            if args.verbose:
                print("    跳过: 未找到可删除的 FSDP 文件")
            continue

        if not state.hf_complete:
            print("    跳过: huggingface 导出不完整")
            print(f"    状态: {state.hf_state}")
            skipped_incomplete_count += 1
            continue

        print(f"    huggingface: ok ({state.hf_dir})")
        print(f"    待删除文件数: {len(state.fsdp_files)}")
        for path in state.fsdp_files:
            print(f"      - {path}")

        if args.execute:
            delete_files(state.fsdp_files)
            deleted_actor_count += 1
            deleted_file_count += len(state.fsdp_files)
            print("    已删除")

    print()
    print("==============================================")
    print("完成")
    print(f"  扫描 actor:         {len(states)}")
    print(f"  HF 不完整而跳过:   {skipped_incomplete_count}")
    print(f"  删除 actor 数:      {deleted_actor_count}")
    print(f"  删除文件数:         {deleted_file_count}")
    print("==============================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
