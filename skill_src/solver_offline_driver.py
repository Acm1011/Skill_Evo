"""
离线 rollout 客户端：训练数据游标、分片到多卡 HTTP server、合并结果。

前置：与 GPU 数量相同的 solver_offline_rollout_server 进程（见 skill_src/Zero/run_with_gpus.sh 导出的环境变量）。

示例：
  export ROLLOUT_SERVER_MODEL=/path/to/Qwen3-4B-Base
  # 与 run_with_gpus.sh 配套时可省略 --server-urls，使用 SE_ROLLOUT_SERVER_URLS 或 SE_ROLLOUT_BASE_PORT+SE_ROLLOUT_N_SERVERS

  python -m skill_src.solver_offline_driver run \\
    --data-files /data/train_part1.jsonl /data/train_part2.jsonl \\
    --steps 100 --batch-size 8 \\
    --work-dir /tmp/skill_rollout_work \\
    --merge-output-dir /data/merged_round
  # 游标状态默认写入 {work_dir}/train_cursor_state.json；也可用 --state-path 指定

  默认将本分片样本以 JSON（data_records）POST 到各 server，结果从响应体收集，不落临时分片盘。
  合并后 ``enrich_merged_rows_from_dataset`` + ``build_merged_rollout_records``：
  回填 ``topic``/``difficulty`` 后做 random / embedding 对照题并拼好 merged 格式（顶层 skill prompt 等）。
  ``--rollout-n`` 原样写入 HTTP 请求体，server 上 vLLM ``SamplingParams.n`` 与之相同。全错样本丢弃，
  有效条数不足时自动多波 rollout，游标按本轮实际消费的原始样本总数推进。
  每波用 ``ThreadPoolExecutor`` 并发 HTTP（``--rollout-max-workers`` 默认同上，上限常 512）。
  默认 ``--rollout-http-chunk-size`` 为 0：**仅按 GPU 数均分**，每 GPU 一单大包，并发最多为 GPU 台数；
  设为正整数则按连续样本切成小块并轮询各 server（类似 reward ``_solver_use_skill``
  ``min(N_requests, workers)``），并发为 ``min(ceil(wave/chunk_size), workers)``。
  若需旧版「写 shard jsonl + 读返回路径」可加 --shard-via-disk。

  若 server 由 shell 提前启动，可在成功后释放一半 GPU 供后续 RL：
  --shutdown-servers-after-run first-half|second-half
  （依赖 Linux 的 fuser，来自 psmisc 包）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from skill_src.skill_manager.data_cursor_io import DATA_CURSOR_FILENAME, write_data_cursor


def _http_json_sanitize(obj: Any) -> Any:
    """
    将 numpy 标量 / 数组等转为 ``json.dumps`` 可编码类型。
    parquet / DataProto 里读出的 ``question``、``gt`` 常为 ``numpy`` 类型，直接 ``json.dumps(body)``
    会在 POST /rollout 时报错。
    """
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
    """
    ``json.dumps(..., default=...)`` 兜底：处理预清洗后仍残留的 numpy / bytes / pandas 等。
    """
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


# ---------------------------------------------------------------------------
# Rollout server URL（与 Zero/run_with_gpus.sh 环境变量对齐）
# ---------------------------------------------------------------------------


def resolve_rollout_server_urls(
    cli_urls: Optional[Sequence[str]],
) -> List[str]:
    """
    优先级：--server-urls > SE_ROLLOUT_SERVER_URLS > SE_ROLLOUT_BASE_PORT（或 ROLLOUT_BASE_PORT）+ SE_ROLLOUT_N_SERVERS + SE_ROLLOUT_HOST。
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
            "使用 skill_src/Zero/run_with_gpus.sh 时会自动导出上述变量。"
        )
    base_port = int(base_s)
    n = int(n_s)
    return [f"http://{host}:{base_port + i}" for i in range(n)]


# ---------------------------------------------------------------------------
# 数据加载与状态
# ---------------------------------------------------------------------------


def load_records_from_files(data_files: Sequence[str]) -> List[Dict[str, Any]]:
    """将多个 jsonl、json 数组或 parquet 合并为样本列表（顺序与传入路径一致）。"""
    records: List[Dict[str, Any]] = []
    for path in data_files:
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


def _state_fingerprint(data_files: Sequence[str]) -> str:
    parts = []
    for p in sorted(os.path.abspath(x) for x in data_files):
        st = os.stat(p)
        parts.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(parts)


@dataclass
class TrainCursorState:
    version: int = 1
    data_fingerprint: str = ""
    total_n: int = 0
    cursor: int = 0
    data_files: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "data_fingerprint": self.data_fingerprint,
            "total_n": self.total_n,
            "cursor": self.cursor,
            "data_files": list(self.data_files or []),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainCursorState":
        return TrainCursorState(
            version=int(d.get("version", 1)),
            data_fingerprint=str(d.get("data_fingerprint", "")),
            total_n=int(d.get("total_n", 0)),
            cursor=int(d.get("cursor", 0)),
            data_files=list(d.get("data_files", [])),
        )


def load_or_init_state(
    state_path: str,
    data_files: Sequence[str],
    total_n: int,
    reset: bool,
) -> TrainCursorState:
    fp = _state_fingerprint(data_files)
    if not reset and os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            st = TrainCursorState.from_dict(json.load(f))
        if st.data_fingerprint != fp or st.total_n != total_n:
            raise ValueError(
                "状态文件与当前 data_files 不一致（指纹或条数变化）。"
                "请使用 --reset-state 或更换 --state-path。"
            )
        return st
    return TrainCursorState(
        data_fingerprint=fp,
        total_n=total_n,
        cursor=0,
        data_files=[os.path.abspath(x) for x in data_files],
    )


def save_state(state_path: str, state: TrainCursorState) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(state_path)) or ".", exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)


def resolve_state_path(state_path: Optional[str], work_dir: str) -> str:
    """
    训练游标文件路径。未指定 state_path 时默认为 {work_dir}/train_cursor_state.json（首次运行会创建）。
    """
    sp = (state_path or "").strip()
    if sp:
        return os.path.abspath(sp)
    return os.path.join(os.path.abspath(work_dir), "train_cursor_state.json")


def allocate_this_round(
    cursor: int,
    steps: int,
    batch_size: int,
    total_n: int,
) -> Tuple[List[int], int]:
    """本回合使用的全局下标 [cursor, cursor + need)；返回 (indices, new_cursor)。"""
    need = steps * batch_size
    if need <= 0:
        raise ValueError("steps * batch_size 必须为正")
    if cursor + need > total_n:
        raise ValueError(
            f"剩余样本不足：cursor={cursor}, need={need}, total_n={total_n}。"
            "请减小 steps/batch_size 或换一批 data_files。"
        )
    indices = list(range(cursor, cursor + need))
    return indices, cursor + need


# ---------------------------------------------------------------------------
# 分片（负载均衡）
# ---------------------------------------------------------------------------


def split_sizes(n: int, num_shards: int) -> List[int]:
    """将 n 条样本拆成 num_shards 份，尽量均匀（每份至多相差 1）。"""
    if num_shards <= 0:
        raise ValueError("num_shards 必须为正")
    base = n // num_shards
    rem = n % num_shards
    return [base + (1 if i < rem else 0) for i in range(num_shards)]


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    _http_json_sanitize(row),
                    ensure_ascii=False,
                    default=_rollout_post_json_default,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# HTTP 调用 rollout server
# ---------------------------------------------------------------------------


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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {err_body}") from e
    return payload


def _tcp_port_from_url(url: str) -> int:
    p = urllib.parse.urlparse(url.strip())
    if p.port is not None:
        return int(p.port)
    if p.scheme == "http":
        return 80
    if p.scheme == "https":
        return 443
    raise ValueError(f"无法从 URL 解析端口: {url}")


def _shutdown_server_indices(n_servers: int, mode: str) -> List[int]:
    """mode: none | first-half | second-half。奇数台时 first 为前 floor(n/2) 台。"""
    if mode == "none" or n_servers <= 0:
        return []
    half = n_servers // 2
    if half == 0:
        return []
    if mode == "first-half":
        return list(range(0, half))
    if mode == "second-half":
        return list(range(half, n_servers))
    raise ValueError(f"未知 shutdown 模式: {mode}")


def kill_tcp_processes_on_ports(ports: List[int]) -> None:
    """对占用各 TCP 端口的进程执行 fuser -k（需系统安装 fuser/psmisc）。"""
    if not ports:
        return
    fuser = "fuser"
    for port in ports:
        try:
            r = subprocess.run(
                [fuser, "-k", f"{port}/tcp"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "未找到 fuser，无法按端口结束 rollout 进程。"
                "请安装 psmisc（如: apt install psmisc）或改为手动停掉一半 server。"
            ) from e
        if r.returncode == 0:
            print(f"[driver] 已结束占用端口 {port} 的进程 (fuser)")
        else:
            # 无进程监听时 fuser 常返回 1
            err = (r.stderr or r.stdout or "").strip()
            print(f"[driver] 端口 {port}: fuser 退出码 {r.returncode} {err or '(可能已无监听)'}")


def shutdown_half_rollout_servers(
    server_urls: List[str],
    mode: str,
) -> None:
    """在 run 成功后调用：按 URL 顺序关闭前一半或后一半 server 对应的 TCP 监听进程。"""
    if mode == "none":
        return
    n = len(server_urls)
    idxs = _shutdown_server_indices(n, mode)
    if not idxs:
        print(f"[driver] shutdown-servers-after-run={mode}：server 数为 {n}，无需关闭")
        return
    ports = [_tcp_port_from_url(server_urls[i]) for i in idxs]
    which = "前" if mode == "first-half" else "后"
    print(
        f"[driver] 将关闭 {which}一半 rollout server（共 {len(ports)} 个），端口: {ports}"
    )
    kill_tcp_processes_on_ports(ports)


# ---------------------------------------------------------------------------
# 合并结果与导出训练格式
# ---------------------------------------------------------------------------


def merge_shard_jsons(
    shard_outputs: List[Tuple[int, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """shard_outputs: (global_base_index, results list from one shard json)。"""
    merged: List[Dict[str, Any]] = []
    for base, items in sorted(shard_outputs, key=lambda x: x[0]):
        for li, local in enumerate(items):
            row = dict(local)
            local_idx = int(row["idx"]) if "idx" in row else li
            row["idx"] = base + local_idx
            row["shard_local_idx"] = local_idx
            merged.append(row)
    return merged


def enrich_merged_rows_from_dataset(
    merged: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> None:
    """
    Rollout 结果不含原始 ``extra_info``；按 ``row[\"idx\"]`` 从数据集记录回填
    ``topic`` / ``difficulty`` 等，供 DeepMath 风格 random-q 池化使用。
    """
    n_rec = len(records)
    for row in merged:
        try:
            idx = int(row["idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= n_rec:
            continue
        src_ei = records[idx].get("extra_info")
        if not isinstance(src_ei, dict):
            continue
        ei = row.setdefault("extra_info", {})
        for key in ("topic", "difficulty", "split", "problem"):
            if key in src_ei:
                ei[key] = src_ei[key]


def _ensure_rollout_row_normalized(row: Dict[str, Any]) -> None:
    """将单条 rollout 扁平字段（question, responses, …）整理为 extra_info.raw_q_info 等。"""
    ei = row.setdefault("extra_info", {})
    if "raw_q_info" not in ei:
        ei["raw_q_info"] = {
            "question": row["question"],
            "responses": row["responses"],
            "answers": row["answers"],
            "gt": row["gt"],
            "is_right": row["is_right"],
            "acc": row["acc"],
        }
    row["raw_question"] = row["question"]
    row.setdefault("reward_model", {})["raw_q_acc"] = row["acc"]


def _load_embedding_cache_from_dir(
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 cache 目录（如 /path/to/embedding_cache）自动发现 ``*.meta.json``，
    加载并返回 ``(doc_emb, query_emb)``。跳过文件指纹校验（cache 与 data_file
    可能在不同机器上，路径不同），只做 meta/npz 完整性检查。
    """
    cache_path = Path(cache_dir).expanduser().resolve()
    if not cache_path.is_dir():
        raise FileNotFoundError(f"embedding_cache_path 不是目录: {cache_path}")
    metas = sorted(cache_path.glob("*.meta.json"))
    if not metas:
        raise FileNotFoundError(f"embedding_cache_path 目录中无 *.meta.json: {cache_path}")

    # 依次尝试，取第一个能成功加载的
    for meta_fp in metas:
        suf = ".meta.json"
        prefix = meta_fp.with_name(meta_fp.name[: -len(suf)])
        npz_fp = Path(str(prefix) + ".npz")
        if not npz_fp.is_file():
            continue
        try:
            with meta_fp.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        try:
            loaded = np.load(npz_fp)
            doc = loaded["doc_embeddings"]
            query = loaded["query_embeddings"]
        except (OSError, KeyError, ValueError):
            continue
        if doc.shape[0] != query.shape[0] or doc.shape[0] == 0:
            continue
        num_problems = int(meta.get("num_problems", doc.shape[0]))
        if num_problems != doc.shape[0]:
            continue
        print(
            f"[driver] 已加载 embedding cache: {meta_fp.name}  "
            f"num_problems={doc.shape[0]}  dim={doc.shape[1]}"
        )
        return doc.astype(np.float32, copy=False), query.astype(np.float32, copy=False)

    raise RuntimeError(
        f"embedding_cache_path 目录中未找到可用的 (*.meta.json + *.npz) 对: {cache_path}"
    )


def _row_rollout_acc(row: Dict[str, Any]) -> float:
    """单条 rollout 的 ``acc``（多 completion 正确率，0~1）；缺省回退 1.0（不当作难题）。"""
    v: Any = row.get("acc")
    if v is None:
        ei = row.get("extra_info") or {}
        rqi = ei.get("raw_q_info") or {}
        v = rqi.get("acc")
    if v is None:
        return 1.0
    try:
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        return float(v)
    except (TypeError, ValueError):
        return 1.0


def _weighted_sample_pool_indices(
    pool: List[int],
    k_take: int,
    merged: List[Dict[str, Any]],
    prefer_low_acc: bool,
) -> List[int]:
    """
    从 ``pool`` 无放回抽 ``k_take`` 个 merged 下标。

    ``prefer_low_acc`` 时：至少 ``ceil(k_take/2)`` 条按权重 ∝ (1-acc)+eps 抽取（低 acc），
    其余在剩余池中均匀随机。
    """
    if k_take <= 0 or not pool:
        return []
    k_take = min(k_take, len(pool))
    if not prefer_low_acc:
        return random.sample(pool, k_take)
    k_low = (k_take + 1) // 2
    k_rand = k_take - k_low
    eps = 1e-6
    remaining = list(pool)
    low_pick: List[int] = []
    for _ in range(k_low):
        if not remaining:
            break
        weights = [
            max(0.0, (1.0 - _row_rollout_acc(merged[j])) + eps) for j in remaining
        ]
        s = float(sum(weights))
        if s <= 0:
            pick_j = random.choice(remaining)
        else:
            pick_j = random.choices(
                remaining, weights=[w / s for w in weights], k=1
            )[0]
        low_pick.append(pick_j)
        remaining.remove(pick_j)
    rand_pick: List[int] = []
    if k_rand > 0 and remaining:
        rand_pick = random.sample(remaining, min(k_rand, len(remaining)))
    return low_pick + rand_pick


_TOPIC_SEP = " -> "


def _topic_parent(topic: Optional[str]) -> Optional[str]:
    if not topic or not isinstance(topic, str):
        return None
    parts = [p.strip() for p in topic.split(_TOPIC_SEP)]
    if len(parts) <= 1:
        return None
    return _TOPIC_SEP.join(parts[:-1])


def _topic_matches_parent(
    anchor_topic: str, cand_topic: Optional[str], parent: str
) -> bool:
    if not cand_topic or not isinstance(cand_topic, str):
        return False
    if cand_topic == parent:
        return True
    return cand_topic.startswith(parent + _TOPIC_SEP)


def _same_difficulty(a: Any, b: Any) -> bool:
    try:
        return round(float(a), 6) == round(float(b), 6)
    except (TypeError, ValueError):
        return False


def _difficulty_neighbor_ok(anchor_d: Any, cand_d: Any) -> bool:
    try:
        a = float(anchor_d)
    except (TypeError, ValueError):
        return False
    return _same_difficulty(cand_d, a - 1.0) or _same_difficulty(cand_d, a + 1.0)


def _row_dataset_meta(row: Dict[str, Any]) -> Tuple[Optional[str], Any]:
    ei = row.get("extra_info") or {}
    t = ei.get("topic")
    d = ei.get("difficulty")
    topic_s: Optional[str] = None
    if isinstance(t, str):
        topic_s = t.strip() or None
    return topic_s, d


def _expand_random_q_pool_deepmath(
    scored: List[Tuple[float, int]],
    merged: List[Dict[str, Any]],
    pos: int,
) -> List[int]:
    """
    按 DeepMath extra_info：先同难度+同 topic，再父 topic+同难度，再 topic 规则不变且难度 ±1，
    最后按相似度顺序补全至全表候选。
    """
    anchor_topic, anchor_d = _row_dataset_meta(merged[pos])
    if anchor_topic is None or anchor_d is None:
        return [j for _, j in scored]

    in_pool: Set[int] = set()
    pool_order: List[int] = []
    parent = _topic_parent(anchor_topic)

    def _add_tier(filter_fn) -> None:
        for _sim, j in scored:
            if j in in_pool:
                continue
            ct, cd = _row_dataset_meta(merged[j])
            if filter_fn(ct, cd):
                in_pool.add(j)
                pool_order.append(j)

    _add_tier(
        lambda ct, cd: ct == anchor_topic and _same_difficulty(cd, anchor_d)
    )
    if parent is not None:
        _add_tier(
            lambda ct, cd: _same_difficulty(cd, anchor_d)
            and _topic_matches_parent(anchor_topic, ct, parent)
        )

    def _topic_ok_c(ct: Optional[str]) -> bool:
        if parent is not None:
            return _topic_matches_parent(anchor_topic, ct, parent)
        return ct == anchor_topic

    _add_tier(lambda ct, cd: _topic_ok_c(ct) and _difficulty_neighbor_ok(anchor_d, cd))

    for _sim, j in scored:
        if j not in in_pool:
            in_pool.add(j)
            pool_order.append(j)
    return pool_order


def _pick_from_similarity_pool(
    pool_order: List[int],
    sim_by_j: Dict[int, float],
    merged: List[Dict[str, Any]],
    k: int,
    prefer_low_acc: bool,
    sim_pool_factor: int,
) -> List[int]:
    """在已定池内按相似度与 acc 策略选 ``k`` 个 merged 下标。"""
    if k <= 0 or not pool_order:
        return []
    pool_sorted = sorted(pool_order, key=lambda j: -sim_by_j.get(j, 0.0))
    k = min(k, len(pool_sorted))
    if not prefer_low_acc:
        return pool_sorted[:k]

    spf = max(1, int(sim_pool_factor))
    k_low = (k + 1) // 2
    k_rand = k - k_low
    m_take = min(len(pool_sorted), k * spf)
    shortlist_js = pool_sorted[:m_take]

    picked: Set[int] = set()
    low_out: List[int] = []

    def _low_acc_fill(need: int) -> None:
        nonlocal low_out
        cands = [(sim_by_j[j], j) for j in shortlist_js if j not in picked]
        cands.sort(key=lambda sj: (_row_rollout_acc(merged[sj[1]]), -sj[0]))
        for _s, j in cands:
            if len(low_out) >= need:
                break
            if j not in picked:
                low_out.append(j)
                picked.add(j)
        if len(low_out) < need:
            extra = [
                (sim_by_j[j], j)
                for j in pool_sorted
                if j not in picked
            ]
            extra.sort(
                key=lambda sj: (_row_rollout_acc(merged[sj[1]]), -sj[0])
            )
            for _s, j in extra:
                if len(low_out) >= need:
                    break
                low_out.append(j)
                picked.add(j)

    _low_acc_fill(k_low)
    rand_pool = [j for j in pool_sorted if j not in picked]
    n_rand = min(k_rand, len(rand_pool))
    rand_out = random.sample(rand_pool, n_rand) if n_rand > 0 else []
    return low_out + rand_out


def _topk_by_embedding(
    pos: int,
    merged: List[Dict[str, Any]],
    k: int,
    doc_emb: np.ndarray,
    query_emb: np.ndarray,
    interval_len: Optional[int] = None,
    prefer_low_acc: bool = True,
    sim_pool_factor: int = 10,
) -> List[int]:
    """
    在本 round merged 列表中（排除 ``pos`` 自身），用 ``merged[pos]["idx"]`` 对应的
    ``query_emb`` 与其余 ``doc_emb`` 算余弦相似度。

    若 ``merged`` 已含 ``extra_info.topic`` / ``difficulty``（见 ``enrich_merged_rows_from_dataset``），
    则候选池按 DeepMath 规则扩展：同难度+同 topic → 同难度+父 topic → 难度 ±1 + topic 规则，
    再并入全表（仍受 interval 限制）按相似度排序。

    ``prefer_low_acc`` 时：至少 ``ceil(k/2)`` 条在短名单内按 acc 升序、相似度降序选取；
    其余在池内均匀随机。``prefer_low_acc=False`` 时池内按相似度 top-``k``。

    ``interval_len`` 若为正，则只考虑与 ``pos`` 同属一个 training 区间的候选。
    """
    if k <= 0:
        return []
    n = len(merged)
    n_emb = doc_emb.shape[0]
    try:
        ci = int(merged[pos]["idx"])
    except (KeyError, TypeError, ValueError):
        return []
    if ci < 0 or ci >= n_emb or ci >= query_emb.shape[0]:
        return []
    lo: Optional[int] = None
    hi: Optional[int] = None
    if interval_len is not None and int(interval_len) > 0:
        il = int(interval_len)
        lo = (ci // il) * il
        hi = lo + il
    q_vec = query_emb[ci].astype(np.float64)
    qn = float(np.linalg.norm(q_vec))
    if qn < 1e-12:
        return []
    q_unit = q_vec / qn

    scored: List[Tuple[float, int]] = []
    for j in range(n):
        if j == pos:
            continue
        try:
            cj = int(merged[j]["idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if cj < 0 or cj >= n_emb:
            continue
        if lo is not None and hi is not None and not (lo <= cj < hi):
            continue
        d_vec = doc_emb[cj].astype(np.float64)
        dn = float(np.linalg.norm(d_vec))
        if dn < 1e-12:
            continue
        scored.append((float(np.dot(d_vec / dn, q_unit)), j))

    scored.sort(key=lambda x: -x[0])
    if not scored:
        return []

    sim_by_j = {j: s for s, j in scored}
    pool_order = _expand_random_q_pool_deepmath(scored, merged, pos)
    return _pick_from_similarity_pool(
        pool_order,
        sim_by_j,
        merged,
        k,
        prefer_low_acc=prefer_low_acc,
        sim_pool_factor=sim_pool_factor,
    )


def build_merged_rollout_records(
    merged: List[Dict[str, Any]],
    num_random: int,
    skill_type: str,
    embedding_cache_path: Optional[str] = None,
    training_step: Optional[int] = None,
    batch_size_for_interval: Optional[int] = None,
    prefer_low_acc: bool = True,
    sim_pool_factor: int = 10,
) -> None:
    """
    对 ``merge_shard_jsons`` 的结果原地整理为最终 merged 格式：

    - 调用方应在 merge 之后对本列表执行 ``enrich_merged_rows_from_dataset(merged, records)``
      （``run`` 已做），以便 ``extra_info.topic`` / ``difficulty`` 与数据集 ``idx`` 对齐。
    - ``num_random_questions > 0`` 且本 round 至少 2 条：从合并列表中随机抽其它样本，
      写入 ``random_questions`` / ``random_q_indices`` / ``reward_model.raw_random_q_acc`` /
      ``extra_info.random_q_info``（与原先 rollout 导出一致，但不含 ``responses`` 以减小体积）。
    - 若提供 ``embedding_cache_path``：在 ``merged`` 已含 topic/difficulty 时按 DeepMath 规则
      扩候选池（同难度+同 topic → 父 topic+同难度 → 难度 ±1 + 同 topic 规则 → 全表），
      再在池内按余弦相似度；``prefer_low_acc`` 时至少 ``ceil(K/2)`` 条在短名单（``sim_pool_factor``）
      内按低 ``acc`` 优先选取，其余在池内均匀随机。缺 topic/difficulty 时退化为区间内的全局相似度池。
    - 无 embedding 时随机对照：``prefer_low_acc`` 为真则至少一半 ``(1-acc)`` 加权、其余均匀随机。
    - ``reward_model``：``raw_q_acc``、``raw_random_q_acc``（列表，无对照时为 ``[]``）、
      ``style``、``skill_type``（与历史 merged 一致）。
    - 每条写顶层 skill ``prompt``（get_skill_prompt）；并在 ``extra_info.skill_traj_prompt_group`` 写入
      ``success_only`` / ``mixed_sf`` / ``unclassified``（由 ``raw_q_info.is_right`` 判定，与模板中
      ``[SUCCESS]``/``[FAIL]`` 一致）。全错行在收尾时从 ``merged`` 中移除。
    - ``cmd_run`` 多波凑数时：仅在最终有效列表上调用本函数一次，故此处不再重复过滤入参。

    若同时传入正整数 ``training_step`` 与 ``batch_size_for_interval``，则 random / embedding
    对照题仅从 **同一 training 区间** 内选取；区间长度 ``= training_step * batch_size_for_interval``
   （例如每区间 5 个 batch、每 batch 128 条 → 640 条样本共享候选池）。不传则候选为全表。
    """
    from skill_src.utils import get_skill_prompt, skill_traj_prompt_group_from_is_right

    n = len(merged)
    path_s = (embedding_cache_path or "").strip()
    use_emb_cache = bool(path_s)
    doc_emb: Optional[np.ndarray] = None
    query_emb: Optional[np.ndarray] = None
    if use_emb_cache:
        doc_emb, query_emb = _load_embedding_cache_from_dir(path_s)

    interval_len: Optional[int] = None
    if training_step is not None and batch_size_for_interval is not None:
        ts = int(training_step)
        bs_i = int(batch_size_for_interval)
        if ts > 0 and bs_i > 0:
            interval_len = ts * bs_i

    k = 0 if use_emb_cache else (min(num_random, n - 1) if num_random > 0 and n > 1 else 0)

    for pos, row in enumerate(merged):
        _ensure_rollout_row_normalized(row)

        pick: List[int] = []
        ei = row["extra_info"]
        if use_emb_cache and doc_emb is not None and query_emb is not None and num_random > 0:
            k_eff = min(num_random, n - 1) if n > 1 else 0
            pick = _topk_by_embedding(
                pos,
                merged,
                k_eff,
                doc_emb,
                query_emb,
                interval_len=interval_len,
                prefer_low_acc=prefer_low_acc,
                sim_pool_factor=sim_pool_factor,
            )
            if pick:
                for j in pick:
                    _ensure_rollout_row_normalized(merged[j])

                dataset_idxs = [merged[j]["idx"] for j in pick]
                row["random_questions"] = [merged[j]["question"] for j in pick]
                row["random_q_indices"] = dataset_idxs

                rqi: Dict[str, Any] = {
                    "indices": dataset_idxs,
                    "questions": [merged[j]["question"] for j in pick],
                    "answers": [],
                    "gt": [],
                    "is_right": [],
                    "acc": [],
                }
                for j in pick:
                    rq = merged[j]["extra_info"]["raw_q_info"]
                    rqi["answers"].append(rq["answers"])
                    rqi["gt"].append(rq["gt"])
                    rqi["is_right"].append(rq["is_right"])
                    rqi["acc"].append(rq["acc"])
                ei["random_q_info"] = rqi
        elif k > 0:
            if interval_len is not None:
                try:
                    ci = int(merged[pos]["idx"])
                except (KeyError, TypeError, ValueError):
                    ci = -1
                if ci < 0:
                    pool = []
                else:
                    lo = (ci // interval_len) * interval_len
                    hi = lo + interval_len
                    pool = []
                    for j in range(n):
                        if j == pos:
                            continue
                        try:
                            cj = int(merged[j]["idx"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if lo <= cj < hi:
                            pool.append(j)
            else:
                pool = [j for j in range(n) if j != pos]
            if pool:
                k_take = min(k, len(pool))
                pick = _weighted_sample_pool_indices(
                    pool, k_take, merged, prefer_low_acc
                )
            else:
                pick = []
            for j in pick:
                _ensure_rollout_row_normalized(merged[j])

            dataset_idxs = [merged[j]["idx"] for j in pick]
            row["random_questions"] = [merged[j]["question"] for j in pick]
            row["random_q_indices"] = dataset_idxs

            rqi = {
                "indices": dataset_idxs,
                "questions": [merged[j]["question"] for j in pick],
                "answers": [],
                "gt": [],
                "is_right": [],
                "acc": [],
            }
            for j in pick:
                rq = merged[j]["extra_info"]["raw_q_info"]
                rqi["answers"].append(rq["answers"])
                rqi["gt"].append(rq["gt"])
                rqi["is_right"].append(rq["is_right"])
                rqi["acc"].append(rq["acc"])
            ei["random_q_info"] = rqi

        # num_random=0 或未命中 embedding 对照时不会进入上面分支，须与带对照的 merged 结构一致，
        # 否则 SynthesizerDataset / SynthsizerRewardManager 读 cache 或 jsonl 会缺 random_q_info。
        if "random_q_info" not in ei:
            ei["random_q_info"] = {
                "indices": [],
                "questions": [],
                "answers": [],
                "gt": [],
                "is_right": [],
                "acc": [],
            }

        rq = ei["raw_q_info"]
        ei["skill_traj_prompt_group"] = skill_traj_prompt_group_from_is_right(
            rq.get("is_right")
        )
        if any(rq["is_right"]):
            row["prompt"] = [
                {
                    "role": "user",
                    "content": get_skill_prompt(
                        row["raw_question"],
                        rq["responses"],
                        rq["is_right"],
                        skill_type,
                    ),
                }
            ]
        else:
            row["prompt"] = [{"role": "user", "content": ""}]
        rm = row.setdefault("reward_model", {})
        rm["raw_q_acc"] = rq["acc"]
        rm["raw_random_q_acc"] = (
            [merged[j]["reward_model"]["raw_q_acc"] for j in pick] if pick else []
        )
        rm.setdefault("style", "rule")
        rm.setdefault("skill_type", skill_type)

    merged[:] = [
        r
        for r in merged
        if any(r["extra_info"]["raw_q_info"]["is_right"])
    ]


def save_train_artifacts(
    train_rows: List[Dict[str, Any]],
    out_dir: str,
    prefix: str = "train_data",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"{prefix}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(
                json.dumps(
                    _http_json_sanitize(row),
                    ensure_ascii=False,
                    default=_rollout_post_json_default,
                )
                + "\n"
            )
    df = pd.DataFrame(train_rows)
    df.to_parquet(os.path.join(out_dir, f"{prefix}.parquet"))


# ---------------------------------------------------------------------------
# 主流程 run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    data_files: List[str] = [os.path.abspath(x) for x in args.data_files]
    records = load_records_from_files(data_files)
    total_n = len(records)
    if total_n == 0:
        raise ValueError("data_files 中没有样本")

    need = args.steps * args.batch_size
    print(f"[driver] need={need} samples (有效条数目标，全错过滤后可能多波 rollout 补足)")
    if total_n == 0:
        raise ValueError("data_files 中没有样本")
    if total_n < need:
        print(
            f"[driver] WARN: total_n={total_n} < need={need}，将按剩余样本多波 rollout，"
            "输出条数可能仍不足 need"
        )
    emb_cache_opt = (getattr(args, "embedding_cache_path", "") or "").strip()
    if (
        args.num_random_questions > 0
        and need < args.num_random_questions + 1
        and not emb_cache_opt
    ):
        raise ValueError(
            f"num_random_questions={args.num_random_questions} 需要本 round 样本数 "
            f"steps*batch_size >= num_random+1，当前 need={need}"
        )

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    state_path_resolved = resolve_state_path(args.state_path, work_dir)

    state = load_or_init_state(
        state_path_resolved,
        data_files,
        total_n,
        reset=args.reset_state,
    )

    initial_cursor = state.cursor
    num_random = args.num_random_questions
    server_urls = resolve_rollout_server_urls(args.server_urls)
    n_servers = len(server_urls)
    if n_servers == 0:
        raise ValueError("请提供 --server-urls 或配置 SE_ROLLOUT_SERVER_URLS / SE_ROLLOUT_N_SERVERS")

    valid_accum: List[Dict[str, Any]] = []
    cursor_pos = state.cursor
    rolled_total = 0
    wave_id = 0

    common_body = {
        "rollout_n": max(1, int(args.rollout_n)),
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "gpu_utilization": args.gpu_utilization,
        "temperature": args.temperature,
        "num_random_questions": 0,
        "skill_type": args.skill_type,
        "storage_path": work_dir,
    }

    mp = (getattr(args, "model_path", "") or "").strip()
    dfs_display = data_files if len(data_files) <= 3 else data_files[:3] + [f"... (+{len(data_files) - 3} more)"]
    extra_mp = f"\n  --model-path（仅记录，推理在 server）: {mp}" if mp else ""
    print(
        "[driver] 即将向 rollout HTTP server 发送请求（以下为实际入参与 common_body 对齐）：\n"
        f"  data_files ({len(data_files)}): {dfs_display}\n"
        f"  --steps {args.steps}  --batch-size {args.batch_size}  need={need} (steps×batch，有效条数目标)\n"
        f"  --rollout-n {args.rollout_n}（请求体与 server vLLM SamplingParams.n 一致）\n"
        f"  --num-random-questions {args.num_random_questions}（server 侧本条为 0；random 在 build_merged 阶段）\n"
        f"  ThreadPoolExecutor 上限={args.rollout_max_workers}；"
        f"--rollout-http-chunk-size={getattr(args, 'rollout_http_chunk_size', 0)} "
        f"（0=每 GPU 一整包并行数≈GPU 台数；>0 时每波拆成多块 POST，并发=min(HTTP 任务数, 上限)，"
        f"对齐 reward 的多连接；ENV SE_OFFLINE_ROLLOUT_HTTP_CHUNK_SIZE）\n"
        f"  servers: n={n_servers}  urls={server_urls!r}"
        f"{extra_mp}",
        flush=True,
    )

    def _one(task: Dict[str, Any]) -> Tuple[int, int, List[Dict[str, Any]]]:
        body = {
            **common_body,
            "num_questions": task["size"],
            "suffix": task["suffix"],
            "storage_path": work_dir,
        }
        if args.shard_via_disk:
            body["data_file"] = task["data_file"]
        else:
            body["data_file"] = ""
            body["data_records"] = task["records"]
        r = post_rollout(task["server_url"], body, timeout=args.request_timeout)
        st = r.get("stats") if isinstance(r, dict) else None
        if isinstance(st, dict) and st.get("wall_time_sec") is not None:
            print(
                f"[driver] shard {task['shard_id']} server_stats: "
                f"wall_s={st.get('wall_time_sec')} "
                f"prompts/s={st.get('prompts_per_sec')} "
                f"completions/s={st.get('completions_per_sec')}"
            )
        if args.shard_via_disk:
            out_path = r.get("output_path")
            if not out_path or not os.path.isfile(out_path):
                raise RuntimeError(f"server 返回无效 output_path: {r}")
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = r.get("results")
            if not isinstance(data, list):
                raise RuntimeError(f"server 应返回 results 列表: {r}")
        if not isinstance(data, list):
            raise RuntimeError("rollout 输出应为 list")
        return task["global_base"], task["shard_id"], data

    while len(valid_accum) < need and cursor_pos < total_n:
        wave_n = min(need - len(valid_accum), total_n - cursor_pos)
        if wave_n <= 0:
            break
        indices = list(range(cursor_pos, cursor_pos + wave_n))
        this_round = [records[i] for i in indices]
        print(
            f"[driver] rollout wave {wave_id}: wave_n={wave_n} "
            f"idx=[{indices[0]}, {indices[-1]}] 有效累计 {len(valid_accum)}/{need}；"
            f"servers={n_servers} urls={server_urls!r}"
        )

        chunk_sz_cfg = max(0, int(getattr(args, "rollout_http_chunk_size", 0) or 0))
        shard_tasks: List[Dict[str, Any]] = []
        if chunk_sz_cfg <= 0:
            sizes = split_sizes(len(this_round), n_servers)
            offset = 0
            for shard_id, sz in enumerate(sizes):
                if sz == 0:
                    continue
                chunk_rows = this_round[offset : offset + sz]
                global_base = indices[offset]
                offset += sz
                task: Dict[str, Any] = {
                    "shard_id": shard_id,
                    "size": sz,
                    "global_base": global_base,
                    "records": chunk_rows,
                    "server_url": server_urls[shard_id % n_servers],
                    "suffix": f"r{initial_cursor}_w{wave_id}_s{shard_id}",
                }
                if args.shard_via_disk:
                    shard_path = os.path.join(
                        work_dir, f"round_w{wave_id}_shard_{shard_id}.jsonl"
                    )
                    write_jsonl(shard_path, chunk_rows)
                    task["data_file"] = shard_path
                shard_tasks.append(task)
        else:
            chunk_sz = max(1, chunk_sz_cfg)
            task_id = 0
            for start_off in range(0, wave_n, chunk_sz):
                end_off = min(start_off + chunk_sz, wave_n)
                chunk_rows = this_round[start_off:end_off]
                global_base = indices[start_off]
                sz = len(chunk_rows)
                task = {
                    "shard_id": task_id,
                    "size": sz,
                    "global_base": global_base,
                    "records": chunk_rows,
                    "server_url": server_urls[task_id % n_servers],
                    "suffix": f"r{initial_cursor}_w{wave_id}_c{task_id}",
                }
                if args.shard_via_disk:
                    shard_path = os.path.join(
                        work_dir,
                        f"round_w{wave_id}_chunk_{task_id}.jsonl",
                    )
                    write_jsonl(shard_path, chunk_rows)
                    task["data_file"] = shard_path
                shard_tasks.append(task)
                task_id += 1

        shard_outputs: List[Tuple[int, List[Dict[str, Any]]]] = []
        n_tasks = len(shard_tasks)
        _pool = max(1, min(int(args.rollout_max_workers), n_tasks))
        mode = (
            "chunked_http"
            if chunk_sz_cfg > 0
            else "single_request_per_gpu"
        )
        print(
            f"[driver] wave {wave_id} HTTP_tasks={n_tasks} mode={mode} chunk_size cfg={chunk_sz_cfg} "
            f"ThreadPoolExecutor max_workers={_pool}（cap={args.rollout_max_workers}）start rollout",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=_pool) as ex:
            futs = [ex.submit(_one, t) for t in shard_tasks]
            for fut in as_completed(futs):
                global_base, shard_id, data = fut.result()
                print(
                    f"[driver] wave {wave_id} shard {shard_id} 完成, "
                    f"global_base={global_base}, n={len(data)}"
                )
                shard_outputs.append((global_base, data))

        merged_wave = merge_shard_jsons(shard_outputs)
        enrich_merged_rows_from_dataset(merged_wave, records)
        for row in merged_wave:
            _ensure_rollout_row_normalized(row)
            if any(row["extra_info"]["raw_q_info"]["is_right"]):
                valid_accum.append(row)

        cursor_pos += wave_n
        rolled_total += wave_n
        print(
            f"[driver] wave {wave_id} 结束: 有效样本 {len(valid_accum)}/{need}, "
            f"cursor_pos={cursor_pos}"
        )
        wave_id += 1

    merged = valid_accum[:need]
    if len(merged) < need:
        print(
            f"[driver] WARNING: 全错过滤或多波用尽后仅得到 {len(merged)}/{need} 条有效样本"
        )

    final_cursor = initial_cursor + rolled_total

    build_merged_rollout_records(
        merged,
        num_random,
        args.skill_type,
        embedding_cache_path=emb_cache_opt or None,
        prefer_low_acc=bool(args.random_q_prefer_low_acc),
        sim_pool_factor=int(args.random_q_sim_pool_factor),
    )
    merge_json = os.path.join(work_dir, f"merged_r{initial_cursor}_{final_cursor}.json")
    with open(merge_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    if args.merge_output_dir:
        save_train_artifacts(
            merged,
            os.path.abspath(args.merge_output_dir),
            prefix=args.merge_prefix,
        )

    state.cursor = final_cursor
    save_state(state_path_resolved, state)
    write_data_cursor(Path(work_dir) / DATA_CURSOR_FILENAME, state.cursor)
    remaining = total_n - state.cursor
    print(
        f"[driver] 本轮共 rollout 原始样本 {rolled_total} 条，输出有效 {len(merged)} 条；"
        f"合并文件: {merge_json}\n"
        f"         cursor {initial_cursor} -> {final_cursor}, 剩余未消费样本: {remaining}"
    )

    if args.shutdown_servers_after_run != "none":
        shutdown_half_rollout_servers(server_urls, args.shutdown_servers_after_run)


def cmd_status(args: argparse.Namespace) -> None:
    data_files = [os.path.abspath(x) for x in args.data_files]
    records = load_records_from_files(data_files)
    total_n = len(records)
    wp = (args.work_dir or "").strip()
    sp = (args.state_path or "").strip()
    if not sp and not wp:
        raise ValueError("请提供 --state-path 或 --work-dir（与 run 时一致即可）。")
    state_path_resolved = resolve_state_path(sp if sp else None, wp if wp else os.getcwd())
    st = load_or_init_state(
        state_path_resolved,
        data_files,
        total_n,
        reset=False,
    )
    print(
        json.dumps(
            {
                "total_n": total_n,
                "cursor": st.cursor,
                "remaining": total_n - st.cursor,
                "data_files": st.data_files,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _env_bool_default(env_name: str, default: bool) -> bool:
    v = (os.environ.get(env_name) or "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


def _env_int_min1(env_name: str, default: int) -> int:
    v = (os.environ.get(env_name) or "").strip()
    if not v:
        return max(1, default)
    try:
        return max(1, int(v))
    except ValueError:
        return max(1, default)


def _offline_rollout_max_workers_default() -> int:
    """与 Synthesizer reward 一致：优先 ``SYNTH_SOLVER_ROLLOUT_MAX_WORKERS``，否则 offline 专用变量。"""
    for env_name in ("SYNTH_SOLVER_ROLLOUT_MAX_WORKERS", "SE_OFFLINE_DRIVER_ROLLOUT_MAX_WORKERS"):
        v = (os.environ.get(env_name) or "").strip()
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                pass
    return 512


def _env_rollout_http_chunk_size_default() -> int:
    """单次 POST ``data_records`` 最大条数；0 表示不切片（每 GPU 一整包）。``SE_OFFLINE_ROLLOUT_HTTP_CHUNK_SIZE``。"""
    v = (os.environ.get("SE_OFFLINE_ROLLOUT_HTTP_CHUNK_SIZE") or "").strip()
    if not v:
        return 0
    try:
        return max(0, int(v))
    except ValueError:
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="solver_offline_driver")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="消费一步数据并分发 rollout")
    run.add_argument(
        "--data-files",
        nargs="+",
        required=True,
        help="训练用 jsonl / json / parquet(.pq) 路径列表，顺序拼接成全局样本池",
    )
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--batch-size", type=int, required=True)
    run.add_argument(
        "--server-urls",
        nargs="*",
        default=None,
        help=(
            "rollout server 的 base URL 列表；可省略，此时使用环境变量 "
            "SE_ROLLOUT_SERVER_URLS 或 SE_ROLLOUT_BASE_PORT+SE_ROLLOUT_N_SERVERS（run_with_gpus.sh 已导出）"
        ),
    )
    run.add_argument("--work-dir", type=str, required=True, help="分片 jsonl、各 shard 输出与合并 json")
    run.add_argument(
        "--state-path",
        type=str,
        default="",
        help="训练游标状态文件；省略则使用 {work_dir}/train_cursor_state.json（首次运行会自动创建）",
    )
    run.add_argument("--reset-state", action="store_true", help="忽略已有状态，从 cursor=0 重算指纹")
    run.add_argument(
        "--merge-output-dir",
        type=str,
        default="",
        help="若设置，则将合并结果再转为 train_data.jsonl/parquet",
    )
    run.add_argument("--exp-name", type=str, default="offline", help="写入 data_source 名")
    run.add_argument("--merge-prefix", type=str, default="train_data")
    run.add_argument("--skill-type", type=str, default="skill_generation_v1")
    run.add_argument("--rollout-n", type=int, default=10)
    run.add_argument(
        "--rollout-max-workers",
        type=int,
        default=_offline_rollout_max_workers_default(),
        help=(
            "每波 rollout 的 ThreadPoolExecutor 并发上限。"
            "--rollout-http-chunk-size=0 时每 GPU 一单，并发通常≤GPU 台数；"
            "为正时 HTTP 请求数=min(ceil(本波样本数/chunk_size), 本上限)。"
            "未传参时优先 SYNTH_SOLVER_ROLLOUT_MAX_WORKERS，其次 SE_OFFLINE_DRIVER_ROLLOUT_MAX_WORKERS。"
        ),
    )
    run.add_argument(
        "--rollout-http-chunk-size",
        type=int,
        default=_env_rollout_http_chunk_size_default(),
        help=(
            "单次 POST body 中所含顶层样本上限；0（默认除非设 SE_OFFLINE_ROLLOUT_HTTP_CHUNK_SIZE）"
            "= 整条 wave 只按 GPU 均分几大包。"
            ">0 时对连续索引切片成多单并轮询各 server URL，可提高客户端并发。"
        ),
    )
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--top-k", type=int, default=50)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--gpu-utilization", type=float, default=0.95)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--num-random-questions", type=int, default=10)
    run.add_argument(
        "--embedding-cache-path",
        type=str,
        default="",
        help=(
            "可选：embedding cache 目录（含 *.meta.json + *.npz）；"
            "用 merged[i][\"idx\"] 查 doc/query 向量，按余弦相似度 top-num_random 替代本 round 随机采样"
        ),
    )
    run.add_argument(
        "--random-q-prefer-low-acc",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_default("SE_RANDOM_Q_PREFER_LOW_ACC", True),
        help=(
            "对照题：至少 ceil(K/2) 条偏好低 acc（embedding：DeepMath 池内短名单按 acc；"
            "无 embedding：(1-acc) 加权），其余在候选池内均匀随机。可由 SE_RANDOM_Q_PREFER_LOW_ACC=0 关闭"
        ),
    )
    run.add_argument(
        "--random-q-sim-pool-factor",
        type=int,
        default=_env_int_min1("SE_RANDOM_Q_SIM_POOL_FACTOR", 10),
        help=(
            "embedding：低 acc 侧主要从相似度 top-(K×因子) 短名单挑选（池为 DeepMath topic/difficulty 扩池后）。"
            "环境变量 SE_RANDOM_Q_SIM_POOL_FACTOR"
        ),
    )
    run.add_argument("--request-timeout", type=float, default=86400.0)
    run.add_argument(
        "--model-path",
        type=str,
        default="",
        help="仅作记录；实际推理在 server 端模型",
    )
    run.add_argument(
        "--shard-via-disk",
        action="store_true",
        help="分片写入 jsonl 并由 server 读盘（旧行为）；默认用 data_records 走内存/网络",
    )
    run.add_argument(
        "--shutdown-servers-after-run",
        choices=["none", "first-half", "second-half"],
        default="none",
        help=(
            "run 全部成功后，按端口结束一半由 shell 启动的 rollout 进程以释放 GPU（"
            "first-half=按 --server-urls 顺序前半，second-half=后半；需 Linux fuser/psmisc）"
        ),
    )

    st = sub.add_parser("status", help="查看游标与剩余样本数")
    st.add_argument("--data-files", nargs="+", required=True)
    st.add_argument(
        "--state-path",
        type=str,
        default="",
        help="状态文件路径；若省略则需同时提供 --work-dir（默认读 {work_dir}/train_cursor_state.json）",
    )
    st.add_argument(
        "--work-dir",
        type=str,
        default="",
        help="与 run 时 --work-dir 一致；未指定 --state-path 时用于解析默认状态文件",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
