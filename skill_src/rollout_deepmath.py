"""
DeepMath-103K 数据集独立 Rollout 脚本

用法:
    export ROLLOUT_MODEL_PATH=/path/to/model
    python -m skill_src.rollout_deepmath --step 100 --batch-size 128 --output-dir ./results \\
        --num-random-questions 10 --embedding-cache-path /path/to/embedding_cache

功能:
    1. 根据 step 和 batch-size 计算数据范围
    2. 从 DeepMath-103K.jsonl 加载对应范围的数据
    3. 分片到多个 rollout server 并行处理
    4. 分片内 merge + build_merged_rollout_records(num_random=0) 后追加写入（断点可训）
    5. 全部样本就绪后收尾：读回全量 jsonl，再调用 build_merged_rollout_records(K, embedding_cache)
       覆盖写回；可选 ``--random-q-training-step``：对照题仅从 **training 区间** 内选取，
       区间长度 = training_step * batch_size（例如 5*128=640，全量 12800 则共 20 个区间）。
    6. 已有 jsonl 可用 skill_src.tools.backfill_deepmath_random_q 补全对照字段（无需重跑 rollout）

收尾或 backfill 后可用 jq 抽样检查（K=num_random_questions）::

    shuf -n 1 out.jsonl | jq '{raw_random_len: (.reward_model.raw_random_q_acc|length), rq: (.extra_info.random_q_info|{keys: keys, nq: (.questions|length), nacc: (.acc|length)})}'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from skill_src.solver_offline_driver import (
    _env_bool_default,
    _env_int_min1,
    _load_embedding_cache_from_dir,
    build_merged_rollout_records as driver_build_merged_rollout_records,
    merge_shard_jsons,
)


# =============================================================================
# 数据加载
# =============================================================================

def load_records_from_files(data_files: Sequence[str]) -> List[Dict[str, Any]]:
    """将多个 jsonl、json 数组或 parquet 合并为样本列表（顺序与传入路径一致）。"""
    records: List[Dict[str, Any]] = []
    for path in tqdm(data_files, desc="加载数据文件"):
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        lower = path.lower()
        if lower.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        elif lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            else:
                raise ValueError(f"JSON 文件应为数组: {path}")
        elif lower.endswith(".parquet") or lower.endswith(".pq"):
            df = pd.read_parquet(path)
            records.extend(df.to_dict(orient="records"))
        else:
            raise ValueError(
                f"不支持的扩展名（请用 .jsonl / .json / .parquet / .pq）: {path}"
            )
    return records


def load_records_in_range(data_file: str, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
    """加载指定范围的记录 [start_idx, end_idx)。"""
    records = []
    total = end_idx - start_idx
    with open(data_file, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total, desc=f"加载数据 [{start_idx}:{end_idx})")
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
                pbar.update(1)
        pbar.close()
    return records


# =============================================================================
# JSON 序列化辅助
# =============================================================================

def _http_json_sanitize(obj: Any) -> Any:
    """将 numpy 标量 / 数组等转为 json.dumps 可编码类型。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return _http_json_sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(_http_json_sanitize(k))
            out[key] = _http_json_sanitize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_http_json_sanitize(x) for x in obj]
    raise TypeError(
        f"_http_json_sanitize: 无法 JSON 编码的类型 {type(obj).__name__}"
    )


def _rollout_post_json_default(o: Any) -> Any:
    """json.dumps default 兜底：处理残留的 numpy / bytes / pandas 等。"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (bytes, bytearray)):
        return bytes(o).decode("utf-8", errors="replace")
    if isinstance(o, pd.Timestamp):
        if pd.isna(o):
            return None
        return o.isoformat()
    raise TypeError(
        f"Object of type {type(o).__name__} is not JSON serializable"
    )


# =============================================================================
# Rollout Server URL 解析
# =============================================================================

def resolve_rollout_server_urls(
    cli_urls: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    优先级：cli_urls > SE_ROLLOUT_SERVER_URLS > SE_ROLLOUT_BASE_PORT + SE_ROLLOUT_N_SERVERS + SE_ROLLOUT_HOST
    """
    if cli_urls:
        out = [u.strip() for u in cli_urls if u.strip()]
        if out:
            return out
    env_urls = os.environ.get("SE_ROLLOUT_SERVER_URLS", "").strip()
    if env_urls:
        parts = re.split(r"[\s,]+", env_urls)
        return [p for p in parts if p]
    host = (os.environ.get("SE_ROLLOUT_HOST", "127.0.0.1").strip() or "127.0.0.1")
    base_s = os.environ.get(
        "SE_ROLLOUT_BASE_PORT",
        os.environ.get("ROLLOUT_BASE_PORT", "8760"),
    ).strip()
    n_s = os.environ.get("SE_ROLLOUT_N_SERVERS", "").strip()
    if not n_s:
        raise ValueError(
            "未提供 --server-urls，且环境中无 SE_ROLLOUT_SERVER_URLS。"
            "请设置 SE_ROLLOUT_SERVER_URLS，或设置 SE_ROLLOUT_N_SERVERS，"
            "并设置 SE_ROLLOUT_BASE_PORT（或 ROLLOUT_BASE_PORT）。"
        )
    base_port = int(base_s)
    n = int(n_s)
    return [f"http://{host}:{base_port + i}" for i in range(n)]


# =============================================================================
# 分片（负载均衡）
# =============================================================================

def split_sizes(n: int, num_shards: int) -> List[int]:
    """将 n 条样本拆成 num_shards 份，尽量均匀（每份至多相差 1）。"""
    if num_shards <= 0:
        raise ValueError("num_shards 必须为正")
    base = n // num_shards
    rem = n % num_shards
    return [base + (1 if i < rem else 0) for i in range(num_shards)]


# =============================================================================
# HTTP 调用 rollout server
# =============================================================================

def post_rollout(
    server_url: str,
    body: Dict[str, Any],
    timeout: float = 86400.0,
) -> Dict[str, Any]:
    url = server_url.rstrip("/") + "/rollout"
    data = json.dumps(
        _http_json_sanitize(body),
        ensure_ascii=False,
        default=_rollout_post_json_default,
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    n_prompts = body.get("num_questions", 0)
    print(f"[post_rollout] 开始请求: {url} | prompts={n_prompts} | timeout={timeout:.0f}s")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        print(f"[post_rollout] 请求完成: {url} | 耗时={elapsed:.2f}s | results={len(payload.get('results', []))}")
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"[post_rollout] HTTP错误: {url} | 耗时={elapsed:.2f}s | code={e.code}")
        raise RuntimeError(f"HTTP {e.code} {url}: {err_body}") from e
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[post_rollout] 请求失败: {url} | 耗时={elapsed:.2f}s | error={type(e).__name__}: {e}")
        raise
    return payload


# =============================================================================
# 断点续存支持
# =============================================================================

def load_existing_results(out_path: str) -> set:
    """
    加载已有的结果文件，返回已完成的原始数据索引集合。
    与 solver_offline_driver 落盘一致：优先读顶层 ``idx``，其次 ``extra_info.raw_q_info.idx``。
    """
    completed_indices = set()
    if not os.path.exists(out_path):
        return completed_indices
    
    print(f"[rollout] 检测到已有结果文件: {out_path}")
    with open(out_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="加载已有结果"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "idx" in row:
                    completed_indices.add(int(row["idx"]))
                elif "extra_info" in row and isinstance(row["extra_info"], dict):
                    raw_q_info = row["extra_info"].get("raw_q_info", {})
                    if "idx" in raw_q_info:
                        completed_indices.add(int(raw_q_info["idx"]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    print(f"[rollout] 已找到 {len(completed_indices)} 条已完成的数据")
    return completed_indices


def save_rollout_results(
    rows: List[Dict[str, Any]],
    out_path: str,
    append: bool = False,
) -> None:
    """
    保存 rollout 结果为 JSONL 文件。
    
    Args:
        rows: 要保存的数据列表
        out_path: 输出文件路径
        append: 是否追加写入，True=追加，False=覆盖
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    mode = "a" if append else "w"
    desc = "追加结果" if append else "保存结果"
    
    with open(out_path, mode, encoding="utf-8") as f:
        for row in tqdm(rows, desc=desc):
            f.write(
                json.dumps(
                    _http_json_sanitize(row),
                    ensure_ascii=False,
                    default=_rollout_post_json_default,
                )
                + "\n"
            )
    action = "追加" if append else "保存"
    print(f"[rollout] {action}完成: {out_path} (本次 {len(rows)} 条)")


def load_merged_rows_from_jsonl(
    jsonl_path: str,
    expected_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    读取 jsonl，按 idx 排序为 [0..N-1] 的列表。
    expected_n 若给定则要求恰有 N 条且 idx 覆盖 0..N-1。
    """
    by_idx: Dict[int, Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "idx" not in row:
                raise ValueError(f"{jsonl_path} 某行缺少顶层 idx")
            idx = int(row["idx"])
            if idx in by_idx:
                raise ValueError(f"{jsonl_path} 存在重复 idx={idx}")
            by_idx[idx] = row

    if not by_idx:
        raise ValueError(f"{jsonl_path} 无有效样本行")

    if expected_n is not None:
        n = int(expected_n)
        if len(by_idx) != n:
            raise ValueError(
                f"{jsonl_path} 条数 {len(by_idx)} 与 expected_n={n} 不一致"
            )
    else:
        n = max(by_idx.keys()) + 1

    missing = [i for i in range(n) if i not in by_idx]
    if missing:
        raise ValueError(
            f"{jsonl_path} 缺失 idx（共 {len(missing)} 个），示例: {missing[:20]}"
        )
    extra = [i for i in by_idx if i >= n]
    if extra:
        raise ValueError(
            f"{jsonl_path} 存在 idx>={n} 的条目（与 N={n} 不符），示例: {sorted(extra)[:10]}"
        )
    return [by_idx[i] for i in range(n)]


def finalize_rollout_jsonl(
    output_path: str,
    total_samples: int,
    num_random_questions: int,
    skill_type: str,
    embedding_cache_path: Optional[str],
    random_q_training_step: Optional[int] = None,
    batch_size_for_interval: Optional[int] = None,
    prefer_low_acc: bool = True,
    sim_pool_factor: int = 10,
) -> None:
    """
    读回全量输出，在整表 merged 上调用 build_merged_rollout_records，原子覆盖写回。
    与 solver_offline_driver 在 merge_shard_jsons 之后单次 build 的语义一致。

    ``prefer_low_acc`` / ``sim_pool_factor`` 与 driver 一致：开启时约一半对照题为高相似、约一半为低 acc。
    """
    if not os.path.isfile(output_path):
        print(f"[rollout] 收尾跳过：文件不存在 {output_path}")
        return

    emb = (embedding_cache_path or "").strip() or None
    ts = random_q_training_step
    bs_i = batch_size_for_interval
    interval_note = ""
    if ts is not None and bs_i is not None and int(ts) > 0 and int(bs_i) > 0:
        interval_note = f", training_interval_len={int(ts) * int(bs_i)} (step={ts}×bs={bs_i})"
    print(
        f"[rollout] 收尾: 全量 {total_samples} 条上 build_merged_rollout_records "
        f"(num_random={num_random_questions}, embedding_cache={emb!r}{interval_note})"
    )
    merged = load_merged_rows_from_jsonl(output_path, expected_n=total_samples)

    if emb and num_random_questions > 0:
        doc, query = _load_embedding_cache_from_dir(emb)
        max_idx = total_samples - 1
        if doc.shape[0] <= max_idx or query.shape[0] <= max_idx:
            raise ValueError(
                f"embedding 矩阵行数须严格大于 max_idx={max_idx}（0-based），"
                f"当前 doc={doc.shape[0]}, query={query.shape[0]}"
            )

    tr_step: Optional[int] = None
    tr_bs: Optional[int] = None
    if ts is not None and bs_i is not None and int(ts) > 0 and int(bs_i) > 0:
        tr_step, tr_bs = int(ts), int(bs_i)

    driver_build_merged_rollout_records(
        merged,
        num_random_questions,
        skill_type,
        emb,
        training_step=tr_step,
        batch_size_for_interval=tr_bs,
        prefer_low_acc=prefer_low_acc,
        sim_pool_factor=sim_pool_factor,
    )
    tmp_path = output_path + ".tmp_finalize"
    save_rollout_results(merged, tmp_path, append=False)
    os.replace(tmp_path, output_path)
    print(f"[rollout] 收尾完成，已覆盖写入: {output_path}")


# =============================================================================
# 主逻辑
# =============================================================================

DATA_FILE = "/home/ycy/sdi/data/DeepMath-103K.jsonl"


def run_rollout(
    step: int,
    batch_size: int,
    output_dir: str,
    output_name: str,
    server_urls: Optional[List[str]] = None,
    rollout_n: int = 10,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    skill_type: str = "skill_generation_v1",
    request_timeout: float = 86400.0,
    num_random_questions: int = 10,
    embedding_cache_path: Optional[str] = None,
    random_q_training_step: Optional[int] = None,
    prefer_low_acc: bool = True,
    sim_pool_factor: int = 10,
) -> str:
    """
    执行 DeepMath-103K 的 rollout，支持断点续存和动态写入。

    Args:
        step: 总步数，数据总量 = step * batch_size（从索引 0 开始）
        batch_size: 每批次大小
        output_dir: 输出目录
        output_name: 输出文件名
        server_urls: rollout server URLs，None 时从环境变量解析
        rollout_n: 每个问题的 rollout 次数
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: top-p 采样
        top_k: top-k 采样
        skill_type: skill 类型
        request_timeout: HTTP 请求超时时间
        num_random_questions: 收尾阶段对照题数量 K（与 solver_offline_driver 一致；0 表示不采对照）
        embedding_cache_path: 可选；若设且 K>0 则按向量 top-K 选对照（同 driver --embedding-cache-path）
        random_q_training_step: 若为正，则对照题仅从 idx 所在 training 区间选；区间长 = 该值 * batch_size
        prefer_low_acc: 约一半高相似 / 均匀，约一半偏好低 acc（embedding 与 driver 一致）
        sim_pool_factor: embedding 路径相似度短名单倍数；1 等价于旧版纯 top-K 相似度

    Returns:
        输出文件的完整路径
    """
    # 验证数据文件
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(f"数据文件不存在: {DATA_FILE}")

    # 计算数据范围：每次从头开始，总数据量 = step * batch_size
    total_samples = step * batch_size
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    
    # 检查已有结果，计算需要跳过的数据
    completed_indices = load_existing_results(output_path)
    
    # 根据已有结果调整起始索引
    if completed_indices:
        # 找到最大的连续完成索引
        max_completed = max(completed_indices)
        if max_completed >= total_samples - 1:
            print(f"[rollout] 所有 {total_samples} 条数据已完成，无需重新 rollout")
            finalize_rollout_jsonl(
                output_path,
                total_samples,
                num_random_questions,
                skill_type,
                embedding_cache_path,
                random_q_training_step=random_q_training_step,
                batch_size_for_interval=batch_size,
                prefer_low_acc=prefer_low_acc,
                sim_pool_factor=sim_pool_factor,
            )
            return output_path
        # 从断点继续（已完成的跳过）
        start_idx = 0  # 仍然从0开始，但在后续筛选时跳过已完成的
        print(f"[rollout] 将从断点继续，跳过已完成的 {len(completed_indices)} 条数据")
    else:
        start_idx = 0
    
    end_idx = total_samples

    print(f"[rollout] step={step}, batch_size={batch_size}")
    print(f"[rollout] 总数据量: {total_samples} (范围: [{start_idx}, {end_idx}))")
    print(f"[rollout] 数据源: {DATA_FILE}")
    print(f"[rollout] 输出文件: {output_path}")

    # 加载数据
    print(f"[rollout] 加载数据中...")
    records = load_records_in_range(DATA_FILE, start_idx, end_idx)
    if len(records) == 0:
        raise ValueError(f"未加载到数据，请检查数据范围和文件: {DATA_FILE}")
    if len(records) < batch_size:
        print(f"[rollout] 警告: 实际加载 {len(records)} 条，少于请求的 {batch_size} 条")

    print(f"[rollout] 加载完成: {len(records)} 条")

    # 解析 server URLs
    if server_urls is None:
        server_urls = resolve_rollout_server_urls()
    n_servers = len(server_urls)
    if n_servers == 0:
        raise ValueError("未提供 server URLs，请设置环境变量或直接传入")

    print(f"[rollout] rollout servers: n={n_servers} urls={server_urls!r}")

    # 构造请求 body 的公共部分
    common_body = {
        "rollout_n": rollout_n,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "num_random_questions": 0,  # 简化版不采样
        "skill_type": skill_type,
        "num_questions": 0,  # 将由 task 填充
        "suffix": "",  # 将由 task 填充
    }

    def _one(task: Dict[str, Any]) -> Tuple[int, int, List[Dict[str, Any]]]:
        shard_id = task["shard_id"]
        server_url = task["server_url"]
        size = task["size"]
        print(f"[_one] shard {shard_id} 开始处理: server={server_url} | records={size}")
        t0 = time.time()
        try:
            body = {
                **common_body,
                "num_questions": task["size"],
                "suffix": task["suffix"],
                "data_records": task["records"],
            }
            r = post_rollout(server_url, body, timeout=request_timeout)
            elapsed = time.time() - t0
            st = r.get("stats") if isinstance(r, dict) else None
            if isinstance(st, dict) and st.get("wall_time_sec") is not None:
                print(
                    f"[rollout] shard {shard_id} 完成: "
                    f"wall_s={st.get('wall_time_sec'):.2f} "
                    f"prompts/s={st.get('prompts_per_sec'):.2f} "
                    f"completions/s={st.get('completions_per_sec'):.2f}"
                )
            data = r.get("results")
            if not isinstance(data, list):
                raise RuntimeError(f"server 应返回 results 列表: {type(r)} {r.keys() if isinstance(r, dict) else 'N/A'}")
            print(f"[_one] shard {shard_id} 成功: 返回 {len(data)} 条结果 | 总耗时={elapsed:.2f}s")
            return task["global_base"], shard_id, data
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[_one] shard {shard_id} 失败: error={type(e).__name__} | 耗时={elapsed:.2f}s | {e}")
            raise

    # 分片：根据已有结果筛选需要处理的数据
    sizes = split_sizes(len(records), n_servers)
    shard_tasks: List[Dict[str, Any]] = []
    offset = 0
    skipped_records = 0
    
    for shard_id, sz in enumerate(sizes):
        if sz == 0:
            continue
        
        global_base = start_idx + offset
        chunk = records[offset : offset + sz]
        offset += sz
        
        # 检查这个 shard 中是否有未完成的记录
        # 计算该 shard 对应的原始数据索引
        shard_indices = set(range(global_base, global_base + sz))
        remaining_indices = shard_indices - completed_indices
        
        if not remaining_indices:
            # 该 shard 的所有数据都已完成，跳过
            skipped_records += sz
            print(f"[rollout] shard {shard_id}: 全部 {sz} 条已完成，跳过")
            continue
        
        # 如果有部分完成，筛选未完成的记录
        if len(remaining_indices) < sz:
            # 部分完成，筛选未完成的记录（保持顺序）
            filtered_chunk = [
                rec for i, rec in enumerate(chunk)
                if (global_base + i) not in completed_indices
            ]
            print(f"[rollout] shard {shard_id}: {sz - len(remaining_indices)}/{sz} 条已完成，"
                  f"处理剩余 {len(remaining_indices)} 条")
        else:
            filtered_chunk = chunk
        
        task: Dict[str, Any] = {
            "shard_id": shard_id,
            "size": len(filtered_chunk),
            "global_base": global_base,
            "original_base": global_base,  # 用于计算真实索引
            "records": filtered_chunk,
            "server_url": server_urls[shard_id % n_servers],
            "suffix": f"step{step}_shard{shard_id}",
            "is_partial": len(filtered_chunk) != sz,
        }
        shard_tasks.append(task)

    if not shard_tasks:
        print(f"[rollout] 所有 {len(records)} 条数据都已完成，本分片无需 HTTP")
        finalize_rollout_jsonl(
            output_path,
            total_samples,
            num_random_questions,
            skill_type,
            embedding_cache_path,
            random_q_training_step=random_q_training_step,
            batch_size_for_interval=batch_size,
            prefer_low_acc=prefer_low_acc,
            sim_pool_factor=sim_pool_factor,
        )
        return output_path
    
    if skipped_records > 0:
        print(f"[rollout] 已跳过 {skipped_records} 条已完成的数据")
    
    # 计算总待处理样本数（用于进度条）
    total_samples_to_process = sum(t["size"] for t in shard_tasks)
    print(f"[rollout] 分片完成: {len(shard_tasks)} 个任务，共 {total_samples_to_process} 条样本待处理")

    # 并行执行分片任务，动态追加写入结果
    completed_count = len(completed_indices)
    files_already_written = os.path.exists(output_path) and completed_count > 0
    processed_count = 0  # 本次处理的样本计数
    print(f"[rollout] 开始 rollout...")
    
    with ThreadPoolExecutor(max_workers=max(1, len(shard_tasks))) as ex:
        futs = {ex.submit(_one, t): t for t in shard_tasks}
        # 使用样本总数作为进度条总量，强制显示在底部并保留
        with tqdm(total=total_samples_to_process, desc="Rollout 进度", unit="条", 
                  position=0, leave=True, dynamic_ncols=True) as pbar:
            for fut in as_completed(futs):
                task_info = futs[fut]
                try:
                    global_base, shard_id, data = fut.result()
                    n_samples = len(data)
                    pbar.update(n_samples)  # 更新样本数量
                    pbar.set_postfix({
                        "shard": shard_id, 
                        "done": f"{completed_count + processed_count + n_samples}/{completed_count + total_samples_to_process}"
                    })
                    pbar.refresh()  # 强制刷新显示
                    
                    # 与 solver_offline_driver 一致：先 merge_shard_jsons 写顶层 idx / shard_local_idx，
                    # 再 build_merged_rollout_records（num_random=0，无 embedding cache）
                    merged_chunk = merge_shard_jsons([(global_base, data)])
                    driver_build_merged_rollout_records(
                        merged_chunk, 0, skill_type, None
                    )
                    
                    # 追加写入文件
                    # 如果文件已存在且已有数据，则追加；否则覆盖写入（首次写入）
                    append_mode = files_already_written
                    save_rollout_results(merged_chunk, output_path, append=append_mode)
                    files_already_written = True  # 首次写入后后续都追加
                    
                    processed_count += n_samples
                    print(f"\n[rollout] shard {shard_id} 完成 (global_base={global_base}), "
                          f"本批 {n_samples} 条，累计 {completed_count + processed_count}/{completed_count + total_samples_to_process} 条")
                except Exception as e:
                    shard_id = task_info.get("shard_id", "unknown")
                    print(f"\n[rollout] shard {shard_id} 执行异常: {type(e).__name__}: {e}")
                    # 继续处理其他 shard，但这个 shard 的数据会丢失
                    # 你可以选择在这里重试或记录失败的任务
                    raise  # 暂时抛出异常，方便调试

    print(f"[rollout] 分片 rollout 结束，本 run 新写入 {processed_count} 条")
    finalize_rollout_jsonl(
        output_path,
        total_samples,
        num_random_questions,
        skill_type,
        embedding_cache_path,
        random_q_training_step=random_q_training_step,
        batch_size_for_interval=batch_size,
        prefer_low_acc=prefer_low_acc,
        sim_pool_factor=sim_pool_factor,
    )
    print(f"[rollout] 输出文件: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="DeepMath-103K 独立 Rollout 脚本")
    parser.add_argument("--step", type=int, required=True, help="总步数，数据总量 = step * batch_size（每次从索引 0 开始处理）")
    parser.add_argument("--batch-size", type=int, required=True, help="每批次大小")
    parser.add_argument("--output-dir", type=str, default="./rollout_results", help="输出目录")
    parser.add_argument("--output-name", type=str, default="deepmath_rollout.jsonl", help="输出文件名")
    parser.add_argument("--server-urls", nargs="*", default=None, help="rollout server URLs（可省略，从环境变量解析）")
    parser.add_argument("--rollout-n", type=int, default=10, help="每个问题的 rollout 次数")
    parser.add_argument("--max-tokens", type=int, default=4096, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="top-p 采样")
    parser.add_argument("--top-k", type=int, default=50, help="top-k 采样")
    parser.add_argument("--skill-type", type=str, default="skill_generation_v1", help="skill 类型")
    parser.add_argument("--request-timeout", type=float, default=86400.0, help="HTTP 请求超时时间")
    parser.add_argument(
        "--num-random-questions",
        type=int,
        default=int(os.environ.get("SE_NUM_RANDOM_QUESTIONS", "10")),
        help="收尾时对照题数量 K（默认 10 或环境变量 SE_NUM_RANDOM_QUESTIONS；0 表示不采对照）",
    )
    _emb_default = (os.environ.get("SE_EMBEDDING_CACHE_PATH") or "").strip() or None
    parser.add_argument(
        "--embedding-cache-path",
        type=str,
        default=_emb_default,
        help="embedding cache 目录（*.meta.json + *.npz）；K>0 时与 driver 一致用于 top-K 近邻",
    )
    _ts_env = (os.environ.get("SE_RANDOM_Q_TRAINING_STEP") or "").strip()
    _random_q_ts_default = int(_ts_env) if _ts_env else None
    parser.add_argument(
        "--random-q-training-step",
        type=int,
        default=_random_q_ts_default,
        help=(
            "对照题仅从 idx 所在 training 区间选取；区间长度 = 本值 × --batch-size "
            "（如 5×128=640）。不设或 ≤0 则候选为全表。默认可读 SE_RANDOM_Q_TRAINING_STEP"
        ),
    )
    parser.add_argument(
        "--random-q-prefer-low-acc",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_default("SE_RANDOM_Q_PREFER_LOW_ACC", True),
        help="约一半高相似/均匀、一半低 acc；--no-random-q-prefer-low-acc 关闭。环境 SE_RANDOM_Q_PREFER_LOW_ACC",
    )
    parser.add_argument(
        "--random-q-sim-pool-factor",
        type=int,
        default=_env_int_min1("SE_RANDOM_Q_SIM_POOL_FACTOR", 10),
        help="embedding：低 acc 半侧从 top-(K×因子) 短名单挑选。环境 SE_RANDOM_Q_SIM_POOL_FACTOR",
    )

    args = parser.parse_args()

    # 检查模型路径环境变量（仅用于记录，实际模型由 rollout server 加载）
    model_path = os.environ.get("ROLLOUT_MODEL_PATH", "")
    if model_path:
        print(f"[rollout] 环境变量 ROLLOUT_MODEL_PATH={model_path} (仅记录，实际由 server 加载)")

    run_rollout(
        step=args.step,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        output_name=args.output_name,
        server_urls=args.server_urls,
        rollout_n=args.rollout_n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        skill_type=args.skill_type,
        request_timeout=args.request_timeout,
        num_random_questions=args.num_random_questions,
        embedding_cache_path=(
            (args.embedding_cache_path or "").strip() or None
        ),
        random_q_training_step=(
            args.random_q_training_step
            if args.random_q_training_step is not None
            and int(args.random_q_training_step) > 0
            else None
        ),
        prefer_low_acc=bool(args.random_q_prefer_low_acc),
        sim_pool_factor=max(1, int(args.random_q_sim_pool_factor)),
    )


if __name__ == "__main__":
    main()
