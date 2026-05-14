#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from skill_src.reward_manager import SynthsizerRewardManager, _to_json_serializable


def _load_rows(train_file: str) -> list[dict[str, Any]]:
    p = Path(train_file)
    if not p.is_file():
        raise FileNotFoundError(f"train file not found: {train_file}")
    if p.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p)
        return df.to_dict(orient="records")
    if p.suffix == ".jsonl":
        df = pd.read_json(p, lines=True)
        return df.to_dict(orient="records")
    raise ValueError(f"unsupported train file: {train_file}")


def _maybe_json_loads(obj: Any) -> Any:
    if not isinstance(obj, str):
        return obj
    s = obj.strip()
    if not s:
        return obj
    if s[0] not in "[{":
        return obj
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return obj


def _normalize_messages(prompt_obj: Any) -> list[dict[str, str]]:
    prompt_obj = _maybe_json_loads(prompt_obj)
    if isinstance(prompt_obj, np.ndarray):
        prompt_obj = prompt_obj.tolist()
    elif isinstance(prompt_obj, tuple):
        prompt_obj = list(prompt_obj)
    if isinstance(prompt_obj, str):
        return [{"role": "user", "content": prompt_obj}]
    if isinstance(prompt_obj, list):
        return prompt_obj
    raise TypeError(f"prompt must be str or list, got {type(prompt_obj).__name__}")


def _normalize_extra_info(extra_info_obj: Any) -> dict[str, Any]:
    extra_info_obj = _maybe_json_loads(extra_info_obj)
    if isinstance(extra_info_obj, dict):
        return extra_info_obj
    if extra_info_obj is None:
        return {}
    raise TypeError(f"extra_info must be dict, got {type(extra_info_obj).__name__}")


def _get_env_int(name: str, default: int) -> int:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def run(args: argparse.Namespace) -> int:
    rows = _load_rows(args.train_file)
    if not rows:
        raise ValueError(f"no rows in train file: {args.train_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.synth_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.synth_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    reward_mgr = SynthsizerRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        storage_path=args.storage_path,
        use_skill_type=args.use_skill_type,
        random_q_coef=args.random_q_coef,
    )

    max_prompt_length = args.max_prompt_length
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k

    core_reward_info: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        messages = _normalize_messages(row.get("prompt"))
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"][:, -max_prompt_length:]
        attention_mask = encoded["attention_mask"][:, -max_prompt_length:]
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if top_k and top_k > 0:
                gen_kwargs["top_k"] = top_k
            out = model.generate(
                **gen_kwargs,
            )
        gen_ids = out[0, input_ids.shape[-1] :]
        skill_str = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        is_skill_format, skill_or_err = reward_mgr.check_skill_format(skill_str)

        extra_info = _normalize_extra_info(row.get("extra_info"))
        raw_q_info = extra_info.get("raw_q_info")
        random_q_info = extra_info.get("random_q_info")
        if not isinstance(raw_q_info, dict) or not isinstance(random_q_info, dict):
            raise KeyError(f"row[{i}] missing extra_info.raw_q_info/random_q_info")
        traj_prompt_group = extra_info.get("skill_traj_prompt_group", "unclassified")

        core_reward_info.append(
            {
                "idx": i,
                "step": args.step,
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
                    "skill_type": args.use_skill_type,
                    "is_format": is_skill_format,
                    "skill": skill_or_err,
                    "raw_skill_str": skill_str,
                },
                "traj_prompt_group": traj_prompt_group,
            }
        )

    rollout_results = reward_mgr._solver_use_skill(
        core_reward_info,
        storage_path=args.storage_path,
        step=args.step,
    )

    reward_infos: list[dict[str, Any]] = []
    for i, rollout_result in enumerate(rollout_results):
        reward = -1.0
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
            res_list = payload["results"]
            skill_raw_q_acc = res_list[0]["acc"]
            skill_random_q_acc = [x["acc"] for x in res_list[1:]]
            raw_q_acc_delta = skill_raw_q_acc - raw_q_acc
            random_q_acc_delta = reward_mgr.random_q_f(skill_random_q_acc) - reward_mgr.random_q_f(raw_random_q_acc)
            reward = raw_q_acc_delta + args.random_q_coef * random_q_acc_delta

        reward_infos.append(
            {
                "idx": i,
                "step": args.step,
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
                    "skill_skipped": skipped,
                    "raw_q_acc_delta": raw_q_acc_delta,
                    "random_q_acc_delta": random_q_acc_delta,
                    "reward": reward,
                },
            }
        )

    reward_dir = Path(args.storage_path) / "reward_info"
    reward_dir.mkdir(parents=True, exist_ok=True)
    out_file = reward_dir / f"exp_data_step_{str(args.step).zfill(3)}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for item in reward_infos:
            f.write(json.dumps(_to_json_serializable(item), ensure_ascii=False) + "\n")

    print(f"[synth_reward_only_driver] wrote {out_file} rows={len(reward_infos)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--synth-model-path", required=True)
    ap.add_argument("--storage-path", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--use-skill-type", default=os.environ.get("SYNTH_USE_SKILL_TYPE", "skill_use_v1"))
    ap.add_argument("--random-q-coef", type=float, default=_get_env_float("SYNTH_RANDOM_Q_COEF", 0.5))
    ap.add_argument("--max-prompt-length", type=int, default=_get_env_int("SYNTH_MAX_PROMPT_LENGTH", 8192))
    ap.add_argument("--max-new-tokens", type=int, default=_get_env_int("SYNTH_MAX_RESPONSE_LENGTH", 512))
    ap.add_argument("--temperature", type=float, default=_get_env_float("SYNTH_QUERY_TEMPERATURE", 0.7))
    ap.add_argument("--top-p", type=float, default=_get_env_float("SYNTH_QUERY_TOP_P", 0.99))
    ap.add_argument("--top-k", type=int, default=_get_env_int("SYNTH_QUERY_TOP_K", -1))
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
