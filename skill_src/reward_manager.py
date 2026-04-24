from collections import defaultdict,Counter

import torch
import re
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from skill_src.solver_offline_driver import post_rollout, resolve_rollout_server_urls
from skill_src.utils import INSTRUCTIONS
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import socket
import time
import random
import urllib.error
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def _retryable_solver_rollout_http_error(e: BaseException) -> bool:
    """超时或短暂网络错误时可重试；HTTP 业务错误（``RuntimeError`` 等）不重试。"""
    if isinstance(e, (TimeoutError, socket.timeout)):
        return True
    if isinstance(e, urllib.error.URLError):
        r = e.reason
        if isinstance(r, (TimeoutError, socket.timeout, OSError, ConnectionError)):
            return True
        return True
    if isinstance(e, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
        return True
    return False


def _synth_aggregate_reward_details(
    reward_infos: List[Dict[str, Any]],
    traj_groups: List[str],
) -> Dict[str, Any]:
    """batch 级聚合；空子集对应键为 ``None``（避免 json 非法 nan）。"""
    n = len(reward_infos)
    assert len(traj_groups) == n

    def _mean(vals: List[float]) -> Optional[float]:
        return (float(sum(vals) / len(vals)) if vals else None)

    raw_deltas: List[float] = []
    rand_deltas: List[float] = []
    eq_num = 0
    eq_den = 0

    for i, ri in enumerate(reward_infos):
        rinf = ri["reward_info"]
        if not rinf["skill_skipped"] and rinf["raw_q_acc_delta"] is not None:
            raw_deltas.append(float(rinf["raw_q_acc_delta"]))
        if not rinf["skill_skipped"] and rinf["random_q_acc_delta"] is not None:
            rand_deltas.append(float(rinf["random_q_acc_delta"]))
        if not rinf["skill_skipped"] and rinf["skill_raw_q_acc"] is not None:
            eq_den += 1
            if float(rinf["raw_q_acc"]) == float(rinf["skill_raw_q_acc"]):
                eq_num += 1

    def _group_reward_and_skip_rate(group: str) -> Tuple[Optional[float], Optional[float], int]:
        idxs = [i for i in range(n) if traj_groups[i] == group]
        if not idxs:
            return None, None, 0
        rews = [float(reward_infos[i]["reward"]) for i in idxs]
        skips = [bool(reward_infos[i]["reward_info"]["skill_skipped"]) for i in idxs]
        return _mean(rews), float(sum(skips) / len(skips)), len(idxs)

    r_succ, skip_succ, n_succ = _group_reward_and_skip_rate("success_only")
    r_mix, skip_mix, n_mix = _group_reward_and_skip_rate("mixed_sf")
    n_uncls = sum(1 for g in traj_groups if g == "unclassified")

    return {
        "raw_q_acc_delta_mean": _mean(raw_deltas),
        "random_q_acc_delta_mean": _mean(rand_deltas),
        "reward_mean_success_only_traj": r_succ,
        "reward_mean_mixed_sf_traj": r_mix,
        "skill_skipped_rate_success_only_traj": skip_succ,
        "skill_skipped_rate_mixed_sf_traj": skip_mix,
        "frac_raw_q_acc_eq_skill_raw_q_acc": (float(eq_num) / float(eq_den)) if eq_den else None,
        "n_success_only_traj": n_succ,
        "n_mixed_sf_traj": n_mix,
        "n_prompt_unclassified": n_uncls,
        "batch_size": n,
    }


def _to_json_serializable(obj: Any) -> Any:
    """把 reward 日志里的 numpy/torch 等转为 json 可序列化类型（仅用于落盘，不改变训练逻辑）。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        if t.numel() == 1:
            return t.item()
        return _to_json_serializable(t.numpy())
    if isinstance(obj, np.ndarray):
        # object 数组或嵌套时 tolist() 里仍可能有 ndarray / np 标量，需继续递归
        return _to_json_serializable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(_to_json_serializable(k))
            out[key] = _to_json_serializable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    raise TypeError(
        f"_to_json_serializable: 未支持的类型 {type(obj).__name__}（reward 日志落盘）"
    )


def custom_extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return None


def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
                # sentence_bleu may return float or list[float], ensure we get a float
                score = score[0] if isinstance(score, list) else score
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def generate_temp_filename(storage_path:str, prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    os.makedirs(f"{storage_path}/temp_results", exist_ok=True)
    return f"{storage_path}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=2):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def get_reward_server_config():
    """从环境变量获取 Reward Server 配置"""
    # 获取端口列表
    ports_str = os.environ.get("SE_REWARD_PORTS", "5000,5001")
    ports = [int(p.strip()) for p in ports_str.split(",") if p.strip()]
    
    # 获取服务器数量
    n_servers = int(os.environ.get("SE_N_REWARD_SERVERS", len(ports)))
    
    # 如果端口数量不足，使用基础端口生成
    base_port = int(os.environ.get("SE_REWARD_BASE_PORT", 5000))
    while len(ports) < n_servers:
        ports.append(base_port + len(ports))
    
    return ports[:n_servers]

def fetch(port, filepath, question_reward):
    """向指定端口的 Reward Server 发送请求"""
    response = requests.get(f"http://0.0.0.0:{port}/hello?name={filepath}&question_reward={question_reward}")
    print(f"[fetch] port={port}, response={response}")
    return True

def generate_results(data, storage_path: str, question_reward: str):
    """将数据分发到多个 Reward Server 并收集结果"""
    # 从环境变量获取端口配置
    ports = get_reward_server_config()
    n_servers = len(ports)
    
    print(f"[generate_results] 使用 {n_servers} 个 Reward Server, 端口: {ports}")
    
    # 将数据分成 n_servers 份
    datas = split_list(data, n_servers)
    random_names = [generate_temp_filename(storage_path=storage_path, prefix=f"temp_{i}") for i in range(n_servers)]
    
    # 保存数据到临时文件
    for i in range(n_servers):
        with open(random_names[i], 'w') as f:
            json.dump(datas[i], f, indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=n_servers) as executor:
        # 使用端口列表而不是索引偏移
        futures = [executor.submit(fetch, ports[i], random_names[i], question_reward) for i in range(n_servers)]

        for future in as_completed(futures):
            print(future.result())

    for i in range(n_servers):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
    for i in range(n_servers):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results


@register("synthesizer")
class SynthsizerRewardManager(AbstractRewardManager):
    """The reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='synthesizer',
        storage_path:str="",
        rollout_server_urls: Optional[List[str]] = None,
        rollout_request_timeout: float = 600.0,
        rollout_http_max_attempts: int = 3,
        use_skill_type: str = "skill_use_v1",
        random_q_coef: float = 0.5,
        solver_rollout_max_workers: int = 512,
    ) -> None:
        assert storage_path is not None, "storage_path must be provided"
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        self.rollout_server_urls = rollout_server_urls
        self.rollout_request_timeout = rollout_request_timeout
        self.use_skill_type = use_skill_type
        self.random_q_coef = random_q_coef
        _a = os.environ.get("SYNTH_ROLLOUT_HTTP_MAX_ATTEMPTS", "").strip()
        if _a:
            try:
                self.rollout_http_max_attempts = max(1, int(_a))
            except ValueError:
                self.rollout_http_max_attempts = max(1, int(rollout_http_max_attempts))
        else:
            self.rollout_http_max_attempts = max(1, int(rollout_http_max_attempts))
        _w = os.environ.get("SYNTH_SOLVER_ROLLOUT_MAX_WORKERS", "").strip()
        if _w:
            try:
                self.solver_rollout_max_workers = max(1, int(_w))
            except ValueError:
                self.solver_rollout_max_workers = max(1, int(solver_rollout_max_workers))
        else:
            self.solver_rollout_max_workers = max(1, int(solver_rollout_max_workers))
        os.makedirs(self.storage_path, exist_ok=True)
        
    _SKILL_JSON_KEYS = ("skill name", "problem type", "key insight", "method")

    def check_skill_format(
        self, skill: str
    ) -> Tuple[bool, Union[Dict[str, str], str]]:
        s = skill.strip()
        if s.startswith("```"):
            parts = s.split("\n", 1)
            s = parts[1] if len(parts) > 1 else ""
            if "```" in s:
                s = s.rsplit("```", 1)[0]
            s = s.strip()
        try:
            return json.loads(s), None
        except json.JSONDecodeError as e:
            return None, str(e)

    def _solver_use_skill(
        self,
        reward_info: List[Dict[str, Any]],
        storage_path: str,
        step: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        仅当 ``skill_info.is_format`` 为真时，将该样本打成 ``data_records`` 并异步请求
        ``solver_offline_rollout_server`` ``/rollout``。

        返回列表长度与 ``reward_info`` 一致、与 batch 下标对齐：未通过格式校验的条目为
        ``skipped=True``，不访问 server。

        环境变量与 ``solver_offline_driver`` 一致：``SE_ROLLOUT_SERVER_URLS`` 或
        ``SE_ROLLOUT_N_SERVERS`` + ``SE_ROLLOUT_BASE_PORT`` + ``SE_ROLLOUT_HOST``；
        或在构造 ``SynthsizerRewardManager`` 时传入 ``rollout_server_urls``。

        说明：``solver_offline_rollout_server`` 使用 ``AsyncLLMEngine``，多连接并发由 vLLM 内部队列
        调度；与旧版相比 HTTP 响应及 ``results`` 条目的字段保持一致。

        客户端侧通过 ``ThreadPoolExecutor`` 并发发起 HTTP 请求（I/O 密集，无需多进程）。
        并发上限为 ``min(合法样本数, solver_rollout_max_workers)``，默认 ``512``，环境变量
        ``SYNTH_SOLVER_ROLLOUT_MAX_WORKERS`` 可覆盖；也可在 ``reward_model.reward_kwargs`` 中传入
        ``solver_rollout_max_workers``。

        单次 ``post_rollout`` 在超时或短暂网络失败时会在 ``rollout_http_max_attempts`` 内重试
        （默认 3 次尝试，含首次），环境变量 ``SYNTH_ROLLOUT_HTTP_MAX_ATTEMPTS`` 可覆盖。
        """
        urls = self.rollout_server_urls
        if not urls:
            urls = resolve_rollout_server_urls(None)
        if not urls:
            raise ValueError("rollout_server_urls 为空且环境变量未配置 rollout server URL")

        os.makedirs(storage_path, exist_ok=True)

        def _rollout_prompt_str(question_text: str,skill) -> str:
            """供 rollout server 使用的完整输入串（与训练侧 chat 格式一致）。"""
            with open(os.path.join(os.path.dirname(__file__), "prompt", f"{self.use_skill_type}.txt"), "r", encoding="utf-8") as f:
                use_skill_template = f.read()
            
            messages = [
                {"role": "user", "content": use_skill_template.format(skill=skill,question=question_text)},
            ]
            kw: Dict[str, Any] = {}
            name = getattr(self.tokenizer, "name_or_path", "") or ""
            if "qwen3" in str(name).lower():
                kw["enable_thinking"] = False
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True,
                **kw,
            )

        def _pack_data_records(item: Dict[str, Any]) -> List[Dict[str, Any]]:
            raw = item["raw_q_info"]
            rand = item["random_q_info"]
            skill = item["skill_info"]["skill"]
            rows: List[Dict[str, Any]] = [
                {
                    "prompt": _rollout_prompt_str(raw["question"],skill),
                    "question": raw["question"],
                    "gt": raw["gt"],
                    "data_source": "synth_reward",
                }
            ]
            random_questions = rand["questions"]
            random_gts = rand["gt"]
            if len(random_questions) != len(random_gts):
                raise ValueError(
                    f"random_q_info questions/gt 长度不一致: {len(random_questions)} vs {len(random_gts)}"
                )
            for q, gt in zip(random_questions, random_gts):
                rows.append(
                    {
                        "prompt": _rollout_prompt_str(q, skill),
                        "question": q,
                        "gt": gt,
                        "data_source": "synth_reward",
                    }
                )
            return rows

        def _one(job_idx: int, batch_i: int, item: Dict[str, Any]) -> Dict[str, Any]:
            url = urls[job_idx % len(urls)]
            try:
                records = _pack_data_records(item)
                nrec = len(records)
                suffix = f"synth_r{step}_{item.get('idx', batch_i)}_{uuid.uuid4().hex[:10]}"
                body: Dict[str, Any] = {
                    "data_file": "",
                    "data_records": records,
                    "num_questions": nrec,
                    "suffix": suffix,
                    "storage_path": storage_path,
                    "skill_type": "skill_generation_v1",
                    "rollout_n": int(os.environ.get("SYNTH_ROLLOUT_N", "4")),
                    "max_tokens": int(os.environ.get("SYNTH_ROLLOUT_MAX_TOKENS", "4096")),
                    "top_k": int(os.environ.get("SYNTH_ROLLOUT_TOP_K", "50")),
                    "top_p": float(os.environ.get("SYNTH_ROLLOUT_TOP_P", "0.95")),
                    "gpu_utilization": float(
                        os.environ.get("SYNTH_ROLLOUT_GPU_UTIL", "0.9")
                    ),
                    "temperature": float(os.environ.get("SYNTH_ROLLOUT_TEMPERATURE", "1.0")),
                    "num_random_questions": 0,
                }
            except Exception as e:
                return {
                    "idx": item.get("idx", batch_i),
                    "step": step,
                    "skipped": True,
                    "reason": "pack_rollout_body_error",
                    "server_url": url,
                    "reward_input": item,
                    "rollout_response": None,
                    "error": str(e),
                }

            max_tries = max(1, int(self.rollout_http_max_attempts))
            last_err: Optional[BaseException] = None
            for attempt in range(max_tries):
                try:
                    payload = post_rollout(
                        url, body, timeout=self.rollout_request_timeout
                    )
                    return {
                        "idx": item.get("idx", batch_i),
                        "step": step,
                        "skipped": False,
                        "reason": None,
                        "server_url": url,
                        "reward_input": item,
                        "rollout_response": payload,
                        "error": None,
                    }
                except Exception as e:
                    last_err = e
                    if not _retryable_solver_rollout_http_error(e):
                        break
                    if attempt + 1 >= max_tries:
                        break
                    time.sleep(min(2.0, 0.25 * (2**attempt)))

            return {
                "idx": item.get("idx", batch_i),
                "step": step,
                "skipped": True,
                "reason": "http server error_occurred",
                "server_url": url,
                "reward_input": item,
                "rollout_response": None,
                "error": str(last_err) if last_err is not None else "unknown",
            }

        n = len(reward_info)
        if n == 0:
            return []

        out: List[Optional[Dict[str, Any]]] = [None] * n
        for i, item in enumerate(reward_info):
            if not item["skill_info"]["is_format"]:
                out[i] = {
                    "idx": item.get("idx", i),
                    "step": step,
                    "skipped": True,
                    "reason": "skill_format_invalid",
                    "server_url": None,
                    "reward_input": item,
                    "rollout_response": None,
                    "error": None,
                }

        eligible: List[Tuple[int, Dict[str, Any]]] = [
            (i, item)
            for i, item in enumerate(reward_info)
            if item["skill_info"]["is_format"]
        ]
        if eligible:
            max_workers = min(len(eligible), self.solver_rollout_max_workers)
            futs: Dict = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for job_idx, (i, item) in enumerate(eligible):
                    fut = ex.submit(_one, job_idx, i, item)
                    futs[fut] = i
                for fut in as_completed(futs):
                    i = futs[fut]
                    out[i] = fut.result()
        # out_path = os.path.join(
        #     storage_path, f"synth_rollout_step_{str(step).zfill(6)}.json"
        # )
        # try:
        #     with open(out_path, "w", encoding="utf-8") as f:
        #         json.dump(out, f, indent=2, ensure_ascii=False)
        # except OSError:
        #     pass
        assert all(x is not None for x in out)
        return cast(List[Dict[str, Any]], out)

    def random_q_f(self, random_acc_list: List[Any]) -> float:
        """对若干条 accuracy 取平均；含非数值则显式报错。

        ``random_q_info["acc"]`` 等可能来自 parquet/numpy，须避免 ``if not ndarray`` 的真值歧义。
        """
        if random_acc_list is None:
            return 0.0
        if isinstance(random_acc_list, np.ndarray):
            random_acc_list = random_acc_list.ravel().tolist()
        elif isinstance(random_acc_list, (int, float, np.integer, np.floating)):
            return float(random_acc_list)
        elif isinstance(random_acc_list, tuple):
            random_acc_list = list(random_acc_list)
        elif not isinstance(random_acc_list, list):
            raise TypeError(
                f"random_q_f: 须为 list/tuple/ndarray/标量，实为 {type(random_acc_list).__name__}: {random_acc_list!r}"
            )
        if len(random_acc_list) == 0:
            return 0.0
        vals: List[float] = []
        for j, x in enumerate(random_acc_list):
            if isinstance(x, (np.integer, np.floating)):
                x = float(x)
            if not isinstance(x, (int, float)):
                raise TypeError(
                    f"random_q_f: 第 {j} 项须为 int/float，实为 {type(x).__name__}: {x!r}"
                )
            vals.append(float(x))
        return sum(vals) / len(vals)

    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # RayPPOTrainer 使用 verl.compute_reward 时不传 step；trainer 已在 batch.meta_info 写入 global_steps
        meta = getattr(data, "meta_info", None) or {}
        gs = meta.get("global_steps")
        if gs is not None:
            step = int(gs)

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        core_reward_info=[]
        valid_response_lengths=[]
        traj_prompt_groups: List[str] = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            extra_info = data_item.non_tensor_batch.get("extra_info")
            if isinstance(extra_info, np.ndarray) and extra_info.size:
                extra_info = (
                    extra_info.item()
                    if extra_info.ndim == 0
                    else extra_info.flat[0]
                )
            if not isinstance(extra_info, dict):
                raise KeyError(
                    f"SynthsizerRewardManager: batch[{i}] extra_info 须为 dict，"
                    f"实为 {type(extra_info).__name__}"
                )

            stored = extra_info.get("skill_traj_prompt_group")
            if stored not in ("success_only", "mixed_sf", "unclassified"):
                raise KeyError(
                    f"SynthsizerRewardManager: batch[{i}] extra_info 缺少合法 "
                    f"skill_traj_prompt_group（须为 success_only | mixed_sf | unclassified），"
                    f"实为 {stored!r}"
                )
            traj_prompt_groups.append(str(stored))

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_lengths.append(valid_response_length)

            # decode
            #prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            skill_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if skill_str.endswith(eos_token):
                skill_str = skill_str[: -len(eos_token)]
            is_skill_format, skill_or_err = self.check_skill_format(skill_str)
            raw_q_info = extra_info["raw_q_info"]
            random_q_info = extra_info["random_q_info"]
            core_reward_info.append({
                "idx": i,
                "step": step,
                "raw_q_info": {
                    "question": raw_q_info["question"],
                    "gt": raw_q_info["gt"],
                    "acc": raw_q_info["acc"],
                },
                "random_q_info": {
                    "questions": random_q_info["questions"],
                    "gt": random_q_info["gt"],
                    "acc": random_q_info["acc"],
                },
                "skill_info": {
                    "skill_type": self.use_skill_type,
                    "is_format": is_skill_format,
                    "skill": skill_or_err,
                    "raw_skill_str": skill_str,
                },
                "traj_prompt_group": traj_prompt_groups[-1],
            })
        rollout_results = self._solver_use_skill(
            core_reward_info, storage_path=self.storage_path, step=step
        )
        assert len(rollout_results) == len(core_reward_info), "rollout_results and core_reward_info must have the same length"
        reward_infos=[]
        for i, rollout_result in enumerate(rollout_results):
            reward = -1.0
            if not isinstance(rollout_result, dict):
                raise TypeError(
                    f"SynthsizerRewardManager: batch[{i}] rollout_result 须为 dict，"
                    f"实为 {type(rollout_result).__name__}"
                )
            if "skipped" not in rollout_result:
                raise KeyError(
                    f"SynthsizerRewardManager: batch[{i}] rollout_result 缺少 'skipped'，"
                    f"keys={sorted(rollout_result.keys())!r}"
                )
            raw_q_acc = core_reward_info[i]["raw_q_info"]["acc"]
            raw_random_q_acc = core_reward_info[i]["random_q_info"]["acc"]
            skill_raw_q_acc = None
            n_rand = len(core_reward_info[i]["random_q_info"]["questions"])
            skill_random_q_acc = [None] * max(0, n_rand - 1)
            raw_q_acc_delta = None
            random_q_acc_delta = None
            skipped = bool(rollout_result["skipped"])
            if not skipped:
                payload = rollout_result["rollout_response"]
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"batch[{i}] rollout_response 须为 dict，实为 {type(payload).__name__}"
                    )
                res_list = payload["results"]
                if not isinstance(res_list, list) or len(res_list) == 0:
                    raise RuntimeError(
                        f"rollout server 返回无效（缺 results）: {payload!r}"
                    )
                assert (
                    core_reward_info[i]["raw_q_info"]["question"]
                    == res_list[0]["question"]
                ), "raw_q_info and rollout_response must have the same question"
                skill_raw_q_acc = res_list[0]["acc"]
                skill_random_q_acc = [row["acc"] for row in res_list[1:]]
                raw_q_acc_delta = skill_raw_q_acc - raw_q_acc
                random_q_acc_delta = self.random_q_f(skill_random_q_acc) - self.random_q_f(
                    raw_random_q_acc
                )
                reward = raw_q_acc_delta + self.random_q_coef * random_q_acc_delta
            reward_tensor[i, valid_response_lengths[i] - 1] = reward
            reward_infos.append({
                "idx": i,
                "step": step,
                "raw_q_info": core_reward_info[i]["raw_q_info"],
                "random_q_info": core_reward_info[i]["random_q_info"],
                "skill_info": core_reward_info[i]["skill_info"],
                "traj_prompt_group": core_reward_info[i]["traj_prompt_group"],
                "rollout_result": rollout_result,
                "reward": reward,
                "reward_info": {
                    "raw_q_acc": raw_q_acc,
                    "raw_random_q_acc": raw_random_q_acc,
                    "skill_raw_q_acc": skill_raw_q_acc,
                    "skill_random_q_acc": skill_random_q_acc,
                    "skill_skipped": rollout_result["skipped"],
                    "raw_q_acc_delta": raw_q_acc_delta,
                    "random_q_acc_delta": random_q_acc_delta,
                    "reward": reward,
                },
            })
        reward_details = _synth_aggregate_reward_details(
            reward_infos, [core_reward_info[i]["traj_prompt_group"] for i in range(len(core_reward_info))]
        )
        bs = len(reward_infos)
        reward_extra_info["reward_details"] = [reward_details] * bs

        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        os.makedirs(reward_info_path_dir, exist_ok=True)
        with open(os.path.join(reward_info_path_dir, f"exp_data_step_{str(step).zfill(3)}.jsonl"), "w", encoding="utf-8") as f:
            for reward_info in reward_infos:
                row = dict(reward_info)
                row["reward_details"] = reward_details
                f.write(
                    json.dumps(
                        _to_json_serializable(row),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor



@register("solver")
class SolverRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        storage_path:str="",
       
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        
    def compute_score(
        self,
        solution_str: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        """Compute the reward score for a solution.

        Args:
            solution_str: The solution string
            ground_truth: The ground truth answer
        
        Returns:
            Reward score (1.0 for correct, -1.0 for incorrect)
        """
        # Limit solution length for efficiency
        solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

        # Verify the solution
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        correct = False
        pred = custom_extract_boxed_content(solution_str)
        for gt in ground_truth:
            if pred is None:
                continue
            correct = grade_answer(str(pred), str(gt))
            if correct:
                break

        reward = 1.0 if correct  else 0.0
        acc = reward

        return {
            "score": reward,
            "acc": acc,
            "pred": pred if pred is not None else 'None',
        }
 
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        #topics = data.non_tensor_batch["topic"] if self.num_examine == 0 else data.non_tensor_batch["data_source"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        uids = data.non_tensor_batch["uid"]
        uid2idx=defaultdict(list)
        
        prompts = []
        responses = []
        responses_length = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            responses_length.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            
            

            
            prompts.append(prompt_str)
            responses.append(response_str)
        
       
        reward_infos = []
        uid2group_acc = defaultdict(list)
        for i in range(len(data)):
            uid2idx[uids[i]].append(i)
            result = self.compute_score(responses[i], data[i].non_tensor_batch['reward_model']['ground_truth'])
            score: float
            valid_response_length = responses_length[i]
            if isinstance(result, dict):
                score = result["score"]
                uid2group_acc[uids[i]].append(result["acc"])
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                uid2group_acc[uids[i]].append(score)
                reward_extra_info["acc"].append(score)
            reward = score
            reward_tensor[i, valid_response_length - 1] = reward
            
        for (i, (uid, group_acc)) in enumerate(uid2group_acc.items()):
            
            reward_infos.append(
                {
                    'idx':uid2idx[uid],
                    "i":i,
                    "uid":uid,
                    "step": step,
                    "skill_id":list(data[uid2idx[uid][0]].non_tensor_batch['extra_info'].get("skill_id",[])),
                    "group_infos":{
                        "problem": str(data[uid2idx[uid][0]].non_tensor_batch['extra_info'].get("problem","")),
                        "prompt": str(prompts[uid2idx[uid][0]]),
                        "response": [str(responses[idx]) for idx in uid2idx[uid]],
                        "acc": list(group_acc),
                    }
                    
                }
            )
            
 
        

        reward_extra_info['reward_infos'] = reward_infos
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        step_str = str(step).zfill(3)
        if self.num_examine > 0:
            os.makedirs(f"{reward_info_path_dir}/valdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/valdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        else:
            os.makedirs(f"{reward_info_path_dir}/expdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/expdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

