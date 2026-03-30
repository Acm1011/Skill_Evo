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
  合并后 ``build_merged_rollout_records``：对本 round 全量列表做 random 采样并拼好 merged 格式（顶层 skill prompt 等）。
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# HTTP 调用 rollout server
# ---------------------------------------------------------------------------


def post_rollout(
    server_url: str,
    body: Dict[str, Any],
    timeout: float = 86400.0,
) -> Dict[str, Any]:
    url = server_url.rstrip("/") + "/rollout"
    data = json.dumps(body).encode("utf-8")
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


def _ensure_rollout_row_normalized(row: Dict[str, Any]) -> None:
    """将单条 rollout 扁平字段（q, responses, …）整理为 extra_info.raw_q_info 等。"""
    ei = row.setdefault("extra_info", {})
    if "raw_q_info" not in ei:
        ei["raw_q_info"] = {
            "question": row["q"],
            "responses": row["responses"],
            "answers": row["answers"],
            "gt": row["gt"],
            "is_right": row["is_right"],
            "acc": row["acc"],
        }
    row["raw_question"] = row["q"]
    row.setdefault("reward_model", {})["raw_q_acc"] = row["acc"]


def build_merged_rollout_records(
    merged: List[Dict[str, Any]],
    num_random: int,
    skill_type: str,
) -> None:
    """
    对 ``merge_shard_jsons`` 的结果原地整理为最终 merged 格式：

    - ``num_random_questions > 0`` 且本 round 至少 2 条：从合并列表中随机抽其它样本，
      写入 ``random_questions`` / ``random_q_indices`` / ``reward_model.raw_random_q_acc`` /
      ``extra_info.random_q_info``（与原先 rollout 导出一致）。
    - ``reward_model``：``raw_q_acc``、``raw_random_q_acc``（列表，无对照时为 ``[]``）、
      ``style``、``skill_type``（与历史 merged 一致）。
    - 每条写顶层 skill ``prompt``（get_skill_prompt）。
    """
    from skill_src.utils import get_skill_prompt

    n = len(merged)
    k = min(num_random, n - 1) if num_random > 0 and n > 1 else 0

    for pos, row in enumerate(merged):
        _ensure_rollout_row_normalized(row)

        pick: List[int] = []
        ei = row["extra_info"]
        if k > 0:
            pool = [j for j in range(n) if j != pos]
            pick = random.sample(pool, k)
            for j in pick:
                _ensure_rollout_row_normalized(merged[j])

            dataset_idxs = [merged[j]["idx"] for j in pick]
            row["random_questions"] = [merged[j]["q"] for j in pick]
            row["random_q_indices"] = dataset_idxs

            rqi: Dict[str, Any] = {
                "indices": dataset_idxs,
                "questions": [merged[j]["q"] for j in pick],
                "responses": [],
                "answers": [],
                "gt": [],
                "is_right": [],
                "acc": [],
            }
            for j in pick:
                rq = merged[j]["extra_info"]["raw_q_info"]
                rqi["responses"].append(rq["responses"])
                rqi["answers"].append(rq["answers"])
                rqi["gt"].append(rq["gt"])
                rqi["is_right"].append(rq["is_right"])
                rqi["acc"].append(rq["acc"])
            ei["random_q_info"] = rqi

        rq = ei["raw_q_info"]
        row["prompt"] = [
            {
                "role": "user",
                "content": get_skill_prompt(
                    row["raw_question"], rq["responses"], rq["is_right"], skill_type
                ),
            }
        ]
        rm = row.setdefault("reward_model", {})
        rm["raw_q_acc"] = rq["acc"]
        rm["raw_random_q_acc"] = (
            [merged[j]["reward_model"]["raw_q_acc"] for j in pick] if k > 0 else []
        )
        rm.setdefault("style", "rule")
        rm.setdefault("skill_type", skill_type)



def save_train_artifacts(
    train_rows: List[Dict[str, Any]],
    out_dir: str,
    prefix: str = "train_data",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"{prefix}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
    if total_n < need:
        raise ValueError(
            f"需要 len(data) >= steps*batch_size，当前 total_n={total_n}, need={need}"
        )
    if args.num_random_questions > 0 and need < args.num_random_questions + 1:
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

    indices, new_cursor = allocate_this_round(
        state.cursor, args.steps, args.batch_size, total_n
    )
    this_round = [records[i] for i in indices]

    num_random = args.num_random_questions
    server_urls = resolve_rollout_server_urls(args.server_urls)
    n_servers = len(server_urls)
    if n_servers == 0:
        raise ValueError("请提供 --server-urls 或配置 SE_ROLLOUT_SERVER_URLS / SE_ROLLOUT_N_SERVERS")

    sizes = split_sizes(len(this_round), n_servers)

    # 分片任务：默认内存列表 + 网络；可选写 shard jsonl（--shard-via-disk）
    shard_tasks: List[Dict[str, Any]] = []
    offset = 0
    for shard_id, sz in enumerate(sizes):
        if sz == 0:
            continue
        chunk = this_round[offset : offset + sz]
        global_base = indices[offset]
        offset += sz
        task: Dict[str, Any] = {
            "shard_id": shard_id,
            "size": sz,
            "global_base": global_base,
            "records": chunk,
            "server_url": server_urls[shard_id % n_servers],
            "suffix": f"r{state.cursor}_{new_cursor}_s{shard_id}",
        }
        if args.shard_via_disk:
            shard_path = os.path.join(work_dir, f"round_shard_{shard_id}.jsonl")
            write_jsonl(shard_path, chunk)
            task["data_file"] = shard_path
        shard_tasks.append(task)

    # server 只做本分片推理；merge 后 build_merged_rollout_records 做跨 shard random 与 skill prompt
    common_body = {
        "rollout_n": args.rollout_n,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "gpu_utilization": args.gpu_utilization,
        "temperature": args.temperature,
        "num_random_questions": 0,
        "skill_type": args.skill_type,
        "storage_path": work_dir,
    }

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
            raise RuntimeError(f"rollout 输出应为 list")
        return task["global_base"], task["shard_id"], data

    shard_outputs: List[Tuple[int, List[Dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(shard_tasks))) as ex:
        futs = [ex.submit(_one, t) for t in shard_tasks]
        for fut in as_completed(futs):
            global_base, shard_id, data = fut.result()
            print(f"[driver] shard {shard_id} 完成, global_base={global_base}, n={len(data)}")
            shard_outputs.append((global_base, data))

    merged = merge_shard_jsons(shard_outputs)
    build_merged_rollout_records(merged, num_random, args.skill_type)
    merge_json = os.path.join(work_dir, f"merged_r{state.cursor}_{new_cursor}.json")
    with open(merge_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    if args.merge_output_dir:
        save_train_artifacts(
            merged,
            os.path.abspath(args.merge_output_dir),
            prefix=args.merge_prefix,
        )

    state.cursor = new_cursor
    save_state(state_path_resolved, state)
    remaining = total_n - state.cursor
    print(
        f"[driver] 本轮使用下标 [{indices[0]}, {indices[-1]}], 已写入合并文件: {merge_json}\n"
        f"         cursor {state.cursor - need} -> {state.cursor}, 剩余未消费样本: {remaining}"
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
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--top-k", type=int, default=50)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--gpu-utilization", type=float, default=0.95)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--num-random-questions", type=int, default=10)
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
