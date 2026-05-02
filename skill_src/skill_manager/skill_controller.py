#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""与文件系统 / jsonl 的桥接：加载 skill、按 reward 更新 utility、对问题做检索。

jsonl 字段名与 SkillItem 及 update 协议对齐；若后续统一改字段名，可集中改本模块的解析处。
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .skill_item import (
    JSON_KEY_ID,
    JSON_KEY_KEY_INSIGHT,
    JSON_KEY_METHOD,
    JSON_KEY_PROBLEM,
    JSON_KEY_PROBLEM_TYPE,
    JSON_KEY_SKILL_FROM,
    JSON_KEY_SKILL_NAME,
    JSON_KEY_UTILITY,
    SkillItem,
)
from .skill_manager import SkillManager


def synth_reward_info_jsonl_path(storage_path: str | Path, step: int) -> Path:
    """与 ``SynthsizerRewardManager`` 一致：``{storage_path}/reward_info/exp_data_step_{step:03}.jsonl``。"""
    return Path(storage_path) / "reward_info" / f"exp_data_step_{str(int(step)).zfill(3)}.jsonl"


def synth_reward_info_jsonl_paths(storage_path: str | Path) -> list[Path]:
    """``{storage_path}/reward_info/exp_data_step_*.jsonl``，按 step 编号升序（用于整合整目录入库）。"""
    rd = Path(storage_path) / "reward_info"
    if not rd.is_dir():
        return []
    keyed: list[tuple[int, Path]] = []
    for p in rd.glob("exp_data_step_*.jsonl"):
        m = re.match(r"^exp_data_step_(\d+)\.jsonl$", p.name)
        if not m:
            continue
        keyed.append((int(m.group(1)), p))
    keyed.sort(key=lambda t: t[0])
    return [p for _, p in keyed]


def _is_synth_reward_jsonl_row(d: dict[str, Any]) -> bool:
    """与 ``SynthsizerRewardManager`` 写入的 ``reward_info/exp_data_step_*.jsonl`` 行结构对齐。"""
    return isinstance(d.get("skill_info"), dict) and isinstance(d.get("raw_q_info"), dict)


def _skill_payload_from_reward_skill_info(si: dict[str, Any]) -> dict[str, Any] | None:
    """训练侧 ``check_skill_format`` 成功时，解析后的四字段在 ``is_format`` 里；``skill`` 为 None。"""
    fmt = si.get("is_format")
    if isinstance(fmt, dict) and fmt:
        return fmt
    sp = si.get("skill")
    if isinstance(sp, dict) and sp:
        return sp
    return None


def _skill_item_from_synth_reward_row(
    d: dict[str, Any],
    *,
    assign_id_if_missing: bool,
) -> SkillItem | None:
    si = d.get("skill_info")
    if not isinstance(si, dict):
        return None
    payload = _skill_payload_from_reward_skill_info(si)
    if payload is None:
        return None
    need = {JSON_KEY_SKILL_NAME, JSON_KEY_PROBLEM_TYPE, JSON_KEY_KEY_INSIGHT, JSON_KEY_METHOD}
    if not need.intersection(payload.keys()):
        return None

    raw_q = d.get("raw_q_info")
    problem = str(raw_q.get("question", "")) if isinstance(raw_q, dict) else ""

    step = d.get("step", "")
    idx = d.get("idx", "")
    traj = d.get("traj_prompt_group", "")
    skill_from = f"synth_reward|step={step}|idx={idx}|{traj}"

    tid = str(d.get(JSON_KEY_ID, "") or "").strip()
    if not tid:
        if assign_id_if_missing:
            tid = ""
        else:
            tid = f"synth_s{step}_i{idx}"

    flat: dict[str, Any] = {
        JSON_KEY_SKILL_NAME: str(payload.get(JSON_KEY_SKILL_NAME, "")),
        JSON_KEY_PROBLEM_TYPE: str(payload.get(JSON_KEY_PROBLEM_TYPE, "")),
        JSON_KEY_KEY_INSIGHT: str(payload.get(JSON_KEY_KEY_INSIGHT, "")),
        JSON_KEY_METHOD: str(payload.get(JSON_KEY_METHOD, "")),
        JSON_KEY_SKILL_FROM: skill_from,
        JSON_KEY_ID: tid,
        JSON_KEY_PROBLEM: problem,
        JSON_KEY_UTILITY: d.get("reward", 0.0),
    }
    return SkillItem.from_json_dict(flat)


def _synth_question_group_key(d: dict[str, Any], line_no: int) -> str:
    """同一题的分组键（可跨文件）；空题目用行号避免误合并。"""
    raw = d.get("raw_q_info")
    if isinstance(raw, dict):
        q = str(raw.get("question", "")).strip()
        if q:
            return q
    return f"__missing_question__:{line_no}"


def _synth_row_rank_key(d: dict[str, Any]) -> tuple[float, int, int]:
    """供 max 比较：reward 大者优先；reward 相同则训练 step 大者优先；再平手 idx 小者优先。"""
    try:
        r = float(d.get("reward", 0.0))
    except (TypeError, ValueError):
        r = float("-inf")
    try:
        st = int(d.get("step", 0))
    except (TypeError, ValueError):
        st = 0
    try:
        idx = int(d.get("idx", 0))
    except (TypeError, ValueError):
        idx = 0
    return (r, st, -idx)


def _pick_synth_rows_per_question(
    synth_rows: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """各 `raw_q_info.question` 在跨文件合并后只保留 reward 最大的一条（平手偏好更高 ``step`` / 更小 ``idx``）。"""
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _ln, d in synth_rows:
        key = _synth_question_group_key(d, _ln)
        by_q[key].append(d)
    out: list[dict[str, Any]] = []
    for rows in by_q.values():
        out.append(max(rows, key=_synth_row_rank_key))
    return out


class SkillController:
    """持有一个 `SkillManager`，封装从 jsonl 读写的常见流程。"""

    def __init__(self, manager: SkillManager) -> None:
        self._manager = manager

    @property
    def manager(self) -> SkillManager:
        return self._manager

    def ingest_skills_from_jsonl_paths(
        self,
        paths: Sequence[str | Path],
        *,
        assign_id_if_missing: bool = True,
        use_eviction: bool = True,
        memory_min_utility: float = 0.0,
    ) -> list[dict[str, Any]]:
        """从多个 jsonl 顺序读入，**合并**所有 synth 行后再按题取全局最优（与单文件行为一致）。"""
        items: list[SkillItem] = []
        synth_buffer: list[tuple[int, dict[str, Any]]] = []
        line_no = 0
        for p in paths:
            pp = Path(p)
            with pp.open(encoding="utf-8") as f:
                for line in f:
                    line_no += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[SkillController] {pp.name} skip line {line_no}: {e}", file=sys.stderr)
                        continue
                    if not isinstance(d, dict):
                        print(
                            f"[SkillController] {pp.name} skip line {line_no}: not a JSON object",
                            file=sys.stderr,
                        )
                        continue

                    if _is_synth_reward_jsonl_row(d):
                        synth_buffer.append((line_no, d))
                        continue

                    raw_id = str(d.get(JSON_KEY_ID, "") or "").strip()
                    if not raw_id and not assign_id_if_missing:
                        print(f"[SkillController] {pp.name} skip line {line_no}: missing id", file=sys.stderr)
                        continue
                    try:
                        item = SkillItem.from_json_dict(d)
                    except (TypeError, ValueError) as e:
                        print(f"[SkillController] {pp.name} skip line {line_no}: {e}", file=sys.stderr)
                        continue
                    if item.utility < float(memory_min_utility):
                        print(
                            f"[SkillController] {pp.name} skip line {line_no}: utility {item.utility} < "
                            f"memory_min_utility {memory_min_utility}",
                            file=sys.stderr,
                        )
                        continue
                    items.append(item)

        for d in _pick_synth_rows_per_question(synth_buffer):
            # try:
            #     rw = float(d.get("reward", 0.0))
            # except (TypeError, ValueError):
            #     rw = float("-inf")
            # if rw < float(memory_min_utility):
            #     k = _synth_question_group_key(d, 0)
                # qshow = (k[:48] + "…") if len(k) > 48 else k
                # print(
                #     f"[SkillController] skip synth (question={qshow!r}): max reward {rw} < "
                #     f"memory_min_utility {memory_min_utility}",
                #     file=sys.stderr,
                # )
                # continue
            item = _skill_item_from_synth_reward_row(d, assign_id_if_missing=assign_id_if_missing)
            if item is None:
                print(
                    f"[SkillController] skip synth row: not a valid synth reward skill row",
                    file=sys.stderr,
                )
                continue
            items.append(item)
        return self._manager.insert_skills(items, use_eviction=use_eviction)

    def ingest_skills_from_jsonl(
        self,
        path: str | Path,
        *,
        assign_id_if_missing: bool = True,
        use_eviction: bool = True,
        memory_min_utility: float = 0.0,
    ) -> list[dict[str, Any]]:
        """从单个 jsonl 读取 skill 行，解析为 `SkillItem` 后批量插入 memory。

        支持两种行格式（自动识别）：

        1. **标准 skills.jsonl**：键与 `SkillItem.to_json_dict` 一致。缺 id 时：若
           ``assign_id_if_missing`` 为 True 则由 ``insert_skills`` 内部分配；否则跳过该行。

        2. **Synth reward 日志**（与 ``skill_src/reward_manager.py`` 中
           ``{storage_path}/reward_info/exp_data_step_XXX.jsonl`` 一致）：含 ``skill_info``、
           ``raw_q_info``、``reward``、``traj_prompt_group`` 等；合法 JSON skill 在
           ``skill_info.is_format``（dict）。**同一** ``raw_q_info.question`` 可出现多行
          （多 traj 或来自多 step 文件）；**仅**全局 ``reward`` 最大的一行可入池；其 ``reward`` 还须
           ``>= memory_min_utility``。``utility`` 取 ``reward``；``problem`` 取
           ``raw_q_info.question``；缺 id 且 ``assign_id_if_missing`` 为 False 时使用
           ``synth_s{step}_i{idx}``，为 True 时交 ``insert_skills`` 分配。

        非 synth 行：若 ``utility < memory_min_utility`` 则跳过。

        等价于 ``ingest_skills_from_jsonl_paths([path], ...)``。
        """
        return self.ingest_skills_from_jsonl_paths(
            [path],
            assign_id_if_missing=assign_id_if_missing,
            use_eviction=use_eviction,
            memory_min_utility=memory_min_utility,
        )

    def ingest_skills_from_synth_reward_dir(
        self,
        storage_path: str | Path,
        *,
        assign_id_if_missing: bool = True,
        use_eviction: bool = True,
        memory_min_utility: float = 0.0,
    ) -> list[dict[str, Any]]:
        """读取 ``synth_reward_info_jsonl_paths(storage_path)`` 下全部 ``exp_data_step_*.jsonl`` 并入库。"""
        paths = synth_reward_info_jsonl_paths(storage_path)
        if not paths:
            raise FileNotFoundError(
                f"no exp_data_step_*.jsonl under {Path(storage_path) / 'reward_info'}"
            )
        return self.ingest_skills_from_jsonl_paths(
            paths,
            assign_id_if_missing=assign_id_if_missing,
            use_eviction=use_eviction,
            memory_min_utility=memory_min_utility,
        )

    def ingest_skills_from_synth_reward_step(
        self,
        storage_path: str | Path,
        step: int,
        *,
        assign_id_if_missing: bool = True,
        use_eviction: bool = True,
        memory_min_utility: float = 0.0,
    ) -> list[dict[str, Any]]:
        """读取 ``synth_reward_info_jsonl_path(storage_path, step)`` 并 ``ingest_skills_from_jsonl``。"""
        p = synth_reward_info_jsonl_path(storage_path, step)
        if not p.is_file():
            raise FileNotFoundError(f"synth reward jsonl not found: {p}")
        return self.ingest_skills_from_jsonl(
            p,
            assign_id_if_missing=assign_id_if_missing,
            use_eviction=use_eviction,
            memory_min_utility=memory_min_utility,
        )

    def update_utilities_from_rewards_jsonl(
        self,
        path: str | Path,
        *,
        persist: bool = True,
    ) -> list[dict[str, Any]]:
        """从 jsonl 按行读取并更新各 skill 的 utility（委托 ``SkillManager.update_utilities_from_rewards``）。

        每行支持两种结构：

        - **旧格式**：``id``、``is_success``、``reward``（缺一不可）。
        - **Solver / reward_info 格式**（与 ``reward_manager`` 写入的条目对齐）：非空
          ``skill_id`` 列表、``group_infos`` 内含非空 ``acc`` 列表；对 ``acc`` 取均值后按
          ``s_t=2*(acc-0.5)`` 与 ``reward`` 参与更新。``reward`` 可省略，默认 ``1.0``。
          行内列出的每个 ``skill_id`` 各计一次更新；多 id 时结果项中会有 ``sub`` 下标。

        若 ``persist`` 为 True，处理完后对 ``manager.persist_path`` 执行 ``save_jsonl``。
        """
        p = Path(path)
        updates: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[SkillController] skip line {line_no}: {e}", file=sys.stderr)
                    continue
                if not isinstance(d, dict):
                    continue
                updates.append(d)

        results = self._manager.update_utilities_from_rewards(updates)
        if persist:
            self._manager.save_jsonl()
        return results

    def retrieve_for_questions_jsonl(
        self,
        path: str | Path,
        *,
        top_k: int | None = None,
        question_key: str = "question",
    ) -> list[dict[str, Any]]:
        """每行一个 JSON，用 ``question_key`` 取问题文本，调用 ``SkillManager.retrieve``。

        返回列表每项含 ``line``、``ok``、``question``、成功时的 ``skills``（``to_json_dict`` 列表）。
        """
        p = Path(path)
        out: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    out.append({
                        "line": line_no,
                        "ok": False,
                        "error": f"json: {e}",
                    })
                    continue
                if not isinstance(d, dict):
                    out.append({"line": line_no, "ok": False, "error": "not a JSON object"})
                    continue
                q = d.get(question_key)
                if not isinstance(q, str) or not q.strip():
                    out.append({
                        "line": line_no,
                        "ok": False,
                        "error": f"missing or empty {question_key!r}",
                    })
                    continue
                try:
                    skills = self._manager.retrieve(q.strip(), top_k=top_k)
                except Exception as e:
                    out.append({
                        "line": line_no,
                        "ok": False,
                        "question": q,
                        "error": str(e),
                    })
                    continue
                out.append({
                    "line": line_no,
                    "ok": True,
                    "question": q,
                    "skills": [s.to_json_dict() for s in skills],
                })
        return out

    @staticmethod
    def _sanitize_extra_info_for_parquet(ex: dict[str, Any]) -> None:
        """将 ``extra_info`` 中除 ``skill_id`` 外的 list 值转为 JSON 字符串，避免 pyarrow
        写 parquet 时在 struct 字段上出现 str / list 混用（如部分样本 ``solution`` 为 str、部分为 list）。"""
        for k, v in list(ex.items()):
            if k == "skill_id":
                continue
            if isinstance(v, list):
                ex[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, dict):
                SkillController._sanitize_extra_info_for_parquet(v)

    @staticmethod
    def _parquet_flatten_prompt_and_extra_info(rec: dict[str, Any]) -> None:
        """整块 ``prompt``（chat 列表）与 ``extra_info`` 写成 JSON 文本列。

        HuggingFace ``datasets.load_dataset('parquet')`` + 嵌套 list/struct 易触发

        ``ArrowNotImplementedError: Nested data conversions not implemented for chunked array``；

        训练侧 ``RLHFDataset`` 在读取后再 ``json.loads`` 还原。"""
        p = rec.get("prompt")
        if isinstance(p, (list, dict)):
            rec["prompt"] = json.dumps(p, ensure_ascii=False)
        ei = rec.get("extra_info")
        if ei is None:
            rec["extra_info"] = "{}"
            return
        if isinstance(ei, dict):
            rec["extra_info"] = json.dumps(ei, ensure_ascii=False)

    @staticmethod
    def _coerce_remaining_nested_to_json_strings(rec: dict[str, Any]) -> None:
        """将其它列里的 dict/list/tuple/set/ndarray 转为 JSON 文本，避免 parquet 保留 struct/list Arrow 嵌套。"""
        for k in list(rec.keys()):
            if k in ("prompt", "extra_info"):
                continue
            v = rec[k]
            if isinstance(v, (dict, list, tuple, set)):
                rec[k] = json.dumps(list(v) if isinstance(v, set) else v, ensure_ascii=False)
            else:
                try:
                    import numpy as np

                    if isinstance(v, np.ndarray):
                        rec[k] = json.dumps(v.tolist(), ensure_ascii=False)
                except ImportError:
                    pass

    def prepare_solver_skills(
        self,
        jsonl_path: str | Path,
        out_parquet: str | Path = "train.parquet",
        *,
        top_k: int | None = 3,
        problem_key: str = "problem",
        template_name: str = "skill_use_v1",
    ) -> Path:
        """从 DeepMath 风格 jsonl 读入样本，按 `extra_info[problem_key]` 检索 skill，

        用 ``prompt/skill_use_v1`` 等模板拼好 user 文本；先从 ``prompt`` 列表中去掉
        ``role==system`` 的消息，再将首个 user 消息的 ``content`` 替换为该文本，并在
        ``extra_info`` 中写入 ``skill_id``（检索到的 id 列表），最后写入 ``out_parquet``
        （默认 `train.parquet`，路径可统一调整）。

        ``top_k``：传给 ``SkillManager.retrieve`` / ``retriever_server`` ``/rank``；默认 ``3``。
        若为 ``None``，服务端会对整库排序并返回全部索引（``skill_id`` 可与 memory 容量同量级）。

        无效行、缺字段、检索失败会跳过并打 stderr 日志，不中断整文件处理。
        """
        p = Path(jsonl_path)
        out = Path(out_parquet)
        prompt_dir = Path(__file__).resolve().parent.parent / "prompt"
        template_path = prompt_dir / f"{template_name}.txt"
        with template_path.open(encoding="utf-8") as f:
            use_template = f.read()

        rows: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[SkillController] prepare_solver_skills skip line {line_no}: {e}", file=sys.stderr)
                    continue
                if not isinstance(d, dict):
                    print(
                        f"[SkillController] prepare_solver_skills skip line {line_no}: not a JSON object",
                        file=sys.stderr,
                    )
                    continue
                extra = d.get("extra_info")
                if not isinstance(extra, dict):
                    print(
                        f"[SkillController] prepare_solver_skills skip line {line_no}: missing extra_info",
                        file=sys.stderr,
                    )
                    continue
                q = extra.get(problem_key)
                if not isinstance(q, str) or not q.strip():
                    print(
                        f"[SkillController] prepare_solver_skills skip line {line_no}: "
                        f"missing or empty extra_info[{problem_key!r}]",
                        file=sys.stderr,
                    )
                    continue
                q = q.strip()
                try:
                    skills = self._manager.retrieve(q, top_k=top_k)
                except Exception as e:
                    print(
                        f"[SkillController] prepare_solver_skills skip line {line_no}: retrieve: {e}",
                        file=sys.stderr,
                    )
                    continue

                skill_block = SkillManager.skills_block_for_template(skills)
                try:
                    new_content = use_template.format(skill=skill_block, question=q)
                except Exception as e:
                    print(
                        f"[SkillController] prepare_solver_skills skip line {line_no}: template: {e}",
                        file=sys.stderr,
                    )
                    continue

                rec = copy.deepcopy(d)
                pl = rec.get("prompt")
                if isinstance(pl, list):
                    pl = [
                        m
                        for m in pl
                        if not (isinstance(m, dict) and m.get("role") == "system")
                    ]
                    rec["prompt"] = pl
                if not self._apply_first_user_content(pl, new_content):
                    print(
                        f"[SkillController] prepare_solver_skills skip line {line_no}: no user message in prompt",
                        file=sys.stderr,
                    )
                    continue
                ex = rec.get("extra_info")
                if not isinstance(ex, dict):
                    ex = {}
                    rec["extra_info"] = ex
                ex["skill_id"] = [s.id for s in skills]
                SkillController._sanitize_extra_info_for_parquet(ex)
                SkillController._parquet_flatten_prompt_and_extra_info(rec)
                SkillController._coerce_remaining_nested_to_json_strings(rec)
                rows.append(rec)

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("prepare_solver_skills 需要 pandas：pip install pandas pyarrow") from e
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)
        return out

    @staticmethod
    def _apply_first_user_content(prompt: Any, new_content: str) -> bool:
        """将 ``prompt`` 列表中首个 ``role==user`` 的 ``content`` 替换为 ``new_content``。"""
        if not isinstance(prompt, list):
            return False
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                msg["content"] = new_content
                return True
        return False
