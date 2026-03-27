#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""有容量上限的 skill 存储，主键为 SkillItem.id。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

from .skill_item import SkillItem


class SkillMemoryFullError(Exception):
    """当前条数已达 max_capacity 时再 add 会抛出。"""


class SkillMemoryDuplicateIdError(Exception):
    """已存在相同 id 时再 add 会抛出。"""


class SkillMemory:
    """按 id 索引的 SkillItem 容器，带最大容量。"""

    def __init__(self, max_capacity: int = 10_000) -> None:
        """
        Args:
            max_capacity: 最多可容纳的 skill 条数（默认 10000，可按业务调整）。
        """
        if max_capacity < 0:
            raise ValueError("max_capacity must be non-negative")
        self._max_capacity = max_capacity
        self._by_id: dict[str, SkillItem] = {}

    @property
    def max_capacity(self) -> int:
        return self._max_capacity

    def set_max_capacity(self, n: int) -> None:
        """调整上限；若新上限小于当前条数则抛出 ValueError。"""
        if n < 0:
            raise ValueError("max_capacity must be non-negative")
        if n < len(self._by_id):
            raise ValueError(
                f"cannot shrink max_capacity to {n}: have {len(self._by_id)} items"
            )
        self._max_capacity = n

    @property
    def capacity(self) -> int:
        """当前存储的 skill 数量。"""
        return len(self._by_id)

    @property
    def is_full(self) -> bool:
        return self.capacity >= self._max_capacity

    @property
    def remaining_slots(self) -> int:
        return max(0, self._max_capacity - self.capacity)

    def get_by_id(self, skill_id: str) -> SkillItem | None:
        return self._by_id.get(skill_id)

    def add(self, item: SkillItem) -> None:
        """加入一条 skill；已满或 id 已存在时抛错。"""
        if item.id in self._by_id:
            raise SkillMemoryDuplicateIdError(f"skill id already exists: {item.id!r}")
        if self.is_full:
            raise SkillMemoryFullError(
                f"skill memory full ({self.capacity}/{self._max_capacity})"
            )
        self._by_id[item.id] = item

    def remove(self, skill_id: str) -> bool:
        """按 id 删除；存在则 True，否则 False。"""
        if skill_id not in self._by_id:
            return False
        del self._by_id[skill_id]
        return True

    def clear(self) -> None:
        self._by_id.clear()

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._by_id

    def items(self) -> Iterator[tuple[str, SkillItem]]:
        yield from self._by_id.items()

    def values(self) -> Iterator[SkillItem]:
        yield from self._by_id.values()

    def bulk_load_from_jsonl(self, path: str | Path) -> int:
        """
        从 skills.jsonl 逐行加载；跳过空行与坏行（打印到 stderr）。
        已存在的 id 会跳过（不覆盖）；满则停止加载并返回已成功加入条数。
        Returns:
            新加入的条数。
        """
        p = Path(path)
        added = 0
        with p.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = SkillItem.from_jsonl_line(line)
                except (json.JSONDecodeError, ValueError) as e:
                    print(
                        f"[SkillMemory] skip line {line_no}: {e}",
                        file=sys.stderr,
                    )
                    continue
                if not item.id:
                    print(
                        f"[SkillMemory] skip line {line_no}: missing id",
                        file=sys.stderr,
                    )
                    continue
                if item.id in self._by_id:
                    continue
                if self.is_full:
                    print(
                        "[SkillMemory] bulk_load stopped: memory full",
                        file=sys.stderr,
                    )
                    break
                try:
                    self.add(item)
                except SkillMemoryFullError:
                    print(
                        "[SkillMemory] bulk_load stopped: memory full",
                        file=sys.stderr,
                    )
                    break
                added += 1
        return added
