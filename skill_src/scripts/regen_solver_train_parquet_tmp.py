#!/usr/bin/env python3
"""临时代码：按本脚本顶部写死的配置，从磁盘 memory_after_syn + data_cursor + 原始语料切片重写 train_data.parquet，
并（默认同 ``memory_func_after_sync.sh``）整文件重写 ``test_data.parquet``。

不须 export 任何环境变量；改 ``_CFG`` 里路径与数字即可。

须先启动 ``retriever_server``（每条样本会做 ``/rank``）。

注意 ``retrieve_top_k``：不传 ``top_k`` 时 ``retriever_server`` 会对**整库 candidates**排序并返回全部索引，
``extra_info.skill_id`` 会变成上千条；务必设为较小的 Top‑K（与 prompt 里展示的 skill 条数一致）。

若已有 ``Memory/doc_embed_cache``（emb_*.npy），须在**启动 retriever** 时挂上该目录，``/rank`` 才会读盘向量、减少 embed 计算；
本脚本/SkillManager 不读该目录。任选其一：
``MEMORY_PATH_DIR=<实验>/Memory``，或 ``DOC_EMBED_CACHE_DIR=<实验>/Memory/doc_embed_cache``，
或 ``SE_RETRIEVER_DOC_CACHE_DIR``（同上路径）。
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

# =====================================================================
# 仅改这里（与你在 main_o / experiment 里用的产物目录一致）
# =====================================================================
_CFG: dict[str, Any] = {
    # 实验根：<saved>/Skill_Evo/data_<数据集>_model_<基座>_v1/
    "experiment_saved_root": Path(
        "/home/ycy/sdi/skill_saved/Skill_Evo/data_DeepMath-103K_model_Qwen3-4B-Instruct-2507_v1"
    ),
    "exp_version": "V1",
    # 原始全量 DeepMath jsonl（与 SE_DATA_FILE 同源）
    "data_file": Path("/home/ycy/sdi/data/DeepMath-103K.jsonl"),
    # retriever_server
    "retriever_url": "http://127.0.0.1:8766",
    # 与 Solver 一致的 batch × steps（本段切片条数）
    "solver_batch_size": 128,
    "solver_retrain_steps": 40,
    "skill_memory_max_capacity": 8192,
    # 非空则用该路径代替默认 ``Memory/memory_after_syn_vN.jsonl``
    "memory_jsonl_override": "",
    # True：写完 parquet 后推进 data_cursor（与 after-sync 末尾行为一致）；False：只重写 parquet，不动游标
    "advance_cursor": False,
    # True：写 parquet 前先 POST /docs/replace（需 memory 已与 retriever 一致或愿意全量重写缓存）
    "sync_doc_cache": False,
    # 每条 problem 检索写入 extra_info.skill_id 时保留的最 relevant 条数（对应 POST /rank 的 top_k）
    "retrieve_top_k": 3,
    # 可选：emb_*.npy 目录；空则用 Memory/doc_embed_cache（仅用于运行时的终端提示）
    "doc_embed_cache_dir": "",
    # 与 memory_func_after_sync：整文件 jsonl → Solver/<exp>/test_data.parquet；设为空字符串 "" 则跳过
    "test_data_file": Path("/home/ycy/sdi/data/test_new.jsonl"),
}


def _add_skill_src_to_path() -> None:
    skill_src = Path(__file__).resolve().parent.parent
    s = str(skill_src)
    if s not in sys.path:
        sys.path.insert(0, s)


def version_slug(exp_version: str) -> str:
    v = exp_version.strip()
    m = re.match(r"^v?(\d+)$", v, re.I)
    if m:
        return "v" + m.group(1)
    m = re.match(r"^V(\d+)$", v)
    if m:
        return "v" + m.group(1)
    return "v" + re.sub(r"^v", "", v, flags=re.I).lower() or "1"


def main() -> int:
    _add_skill_src_to_path()
    from skill_manager.data_cursor_io import (
        DATA_CURSOR_FILENAME,
        read_data_cursor,
        read_jsonl_slice,
        write_data_cursor,
    )
    from skill_manager.skill_controller import SkillController
    from skill_manager.skill_manager import SkillManager

    root = Path(_CFG["experiment_saved_root"]).resolve()
    exp = str(_CFG["exp_version"])
    solver_path_dir = root / "Solver"
    memory_path_dir = root / "Memory"
    data_file = Path(_CFG["data_file"]).resolve()
    retriever_url = str(_CFG["retriever_url"]).rstrip("/")
    sbs = int(_CFG["solver_batch_size"])
    sol_steps = int(_CFG["solver_retrain_steps"])
    train_n = sbs * sol_steps
    cap = max(1, int(_CFG["skill_memory_max_capacity"]))
    advance_cursor = bool(_CFG["advance_cursor"])
    sync_doc_cache = bool(_CFG["sync_doc_cache"])
    ov = (_CFG.get("memory_jsonl_override") or "").strip()
    rk_raw = _CFG.get("retrieve_top_k")
    retrieve_top_k = int(rk_raw) if rk_raw is not None else 8
    if retrieve_top_k < 1:
        print("[regen] retrieve_top_k must be >= 1", file=sys.stderr)
        return 1

    sol_dir = solver_path_dir / exp
    mem_dir = memory_path_dir
    sl = version_slug(exp)

    mem_syn = Path(ov) if ov else (mem_dir / f"memory_after_syn_{sl}.jsonl")
    if not mem_syn.is_file():
        print(f"[regen] missing memory jsonl: {mem_syn}", file=sys.stderr)
        return 1
    if not data_file.is_file():
        print(f"[regen] missing data file: {data_file}", file=sys.stderr)
        return 1

    doc_cache_cfg = (_CFG.get("doc_embed_cache_dir") or "").strip()
    doc_cache = Path(doc_cache_cfg).resolve() if doc_cache_cfg else (mem_dir / "doc_embed_cache").resolve()
    if doc_cache.is_dir():
        try:
            n_emb = sum(
                1 for name in os.listdir(doc_cache) if name.startswith("emb_") and name.endswith(".npy")
            )
        except OSError:
            n_emb = -1
        print(
            f"[regen] doc_embed_cache: {doc_cache}（emb_*.npy 约 {n_emb} 个）；"
            "请用该路径启动 retriever 才能命中缓存、减少 embedding 计算：",
            file=sys.stderr,
        )
        print(f"  DOC_EMBED_CACHE_DIR={doc_cache} bash skill_src/Zero/start_retriever_server.sh &", file=sys.stderr)
        print(f"  或: MEMORY_PATH_DIR={mem_dir} bash skill_src/Zero/start_retriever_server.sh &", file=sys.stderr)

    manager = SkillManager(persist_path=mem_syn, retriever_url=retriever_url, max_capacity=cap)
    n_loaded = manager.load_jsonl(mem_syn)
    print(f"[regen] loaded {n_loaded} skills from {mem_syn}", file=sys.stderr)
    ctrl = SkillController(manager)

    if sync_doc_cache:
        try:
            manager.sync_retriever_doc_cache()
            print("[regen] sync_retriever_doc_cache ok", file=sys.stderr)
        except Exception as e:
            print(f"[regen] sync_retriever_doc_cache failed: {e}", file=sys.stderr)
            return 1

    cur_path = sol_dir / DATA_CURSOR_FILENAME
    start = read_data_cursor(cur_path)
    try:
        _slice, next_c = read_jsonl_slice(data_file, start, train_n)
    except ValueError as e:
        print(
            f"[regen] slice failed: need {train_n} from cursor={start}, batch={sbs}×steps={sol_steps}: {e}",
            file=sys.stderr,
        )
        return 1

    out_parq = sol_dir / "train_data.parquet"
    out_parq.parent.mkdir(parents=True, exist_ok=True)

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
        ctrl.prepare_solver_skills(tmp_path, out_parquet=out_parq, top_k=retrieve_top_k)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass

    if advance_cursor:
        write_data_cursor(cur_path, next_c)
        print(f"[regen] advanced data_cursor {start} -> {next_c}", file=sys.stderr)
    else:
        print(
            f"[regen] data_cursor unchanged ({start}); slice [{start},{start + len(_slice)})",
            file=sys.stderr,
        )
    print(
        f"[regen] wrote {out_parq} (rows={len(_slice)}, batch={sbs}, steps={sol_steps}, "
        f"retrieve_top_k={retrieve_top_k}, cap={cap})",
        file=sys.stderr,
    )

    _default_test = Path("/home/ycy/sdi/data/test.jsonl")
    tdf_raw = _CFG.get("test_data_file")
    if isinstance(tdf_raw, str) and not tdf_raw.strip():
        print("[regen] skip test_data.parquet (test_data_file is empty)", file=sys.stderr)
    else:
        test_path = Path(tdf_raw).resolve() if tdf_raw is not None else _default_test.resolve()
        if not test_path.is_file():
            print(
                f"[regen] skip test_data.parquet: file not found: {test_path}",
                file=sys.stderr,
            )
        else:
            out_test = sol_dir / "test_data.parquet"
            ctrl.prepare_solver_skills(test_path, out_parquet=out_test, top_k=retrieve_top_k)
            print(
                f"[regen] wrote {out_test} (source={test_path}, retrieve_top_k={retrieve_top_k})",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
