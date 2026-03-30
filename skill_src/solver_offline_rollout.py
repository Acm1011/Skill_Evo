import multiprocessing

multiprocessing.set_start_method("spawn", force=True)


from typing import Any, Dict, List

import vllm
from transformers import AutoConfig, AutoTokenizer
from vllm.outputs import RequestOutput
from tqdm import tqdm
import regex as re

from skill_src.utils import custom_grade_answer, extract_boxed_content


def is_qwen3_post_trained(model_path):
    """判断是否是后训练的 Qwen3 模型（非 Base 模型）

    通过 model_type 判断是否是 Qwen3，通过 _name_or_path 判断是否是 Base 模型
    Qwen3 Base 模型的命名格式是 Qwen3-*B-Base，如 Qwen/Qwen3-4B-Base
    """
    try:
        config = AutoConfig.from_pretrained(model_path)
        model_type = getattr(config, "model_type", "")
        name_or_path = getattr(config, "_name_or_path", model_path)

        is_qwen3 = "qwen3" in model_type.lower()
        is_base = bool(re.search(r"qwen3-[^/-]+-base\b", name_or_path.lower()))

        return is_qwen3 and not is_base
    except Exception as e:
        print(f"Warning: 无法读取模型配置: {e}")
        return False


def _gt_from_row(row: Dict[str, Any]) -> Any:
    g = row.get("gt")
    if g is None and isinstance(row.get("reward_model"), dict):
        g = row["reward_model"].get("ground_truth")
    if g is None:
        raise ValueError("每条记录需要 gt 或 reward_model.ground_truth")
    return g


def _prompt_strings(
    data_records: List[Dict[str, Any]],
    tokenizer,
    qwen3_post_trained: bool,
) -> List[str]:
    """prompt 为已渲染 str 则直接用；为 messages 列表则 apply_chat_template。"""
    out: List[str] = []
    chat_kw: Dict[str, Any] = {}
    if qwen3_post_trained:
        chat_kw["enable_thinking"] = False

    for row in data_records:
        p = row["prompt"]
        if isinstance(p, str) and p.strip():
            out.append(p.strip())
        elif isinstance(p, (list, tuple)) and len(p) > 0:
            out.append(
                tokenizer.apply_chat_template(
                    list(p),
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=True,
                    **chat_kw,
                )
            )
        else:
            raise ValueError(
                "每条记录的 prompt 须为非空 str，或非空 messages 列表"
            )
    return out


def run_rollout(
    args,
    model=None,
    tokenizer=None,
    qwen3_post_trained=None,
    data_records: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    入参：``args`` 为采样与模型路径等配置；``data_records`` 每条含 ``prompt`` 与 ``gt`` /
    ``reward_model.ground_truth``。

    出参：每条仅含本样本 rollout 与判分结果（无嵌套 ``extra_info`` 等）；driver 合并前会归一化。
    """
    if not data_records:
        raise ValueError("data_records 不能为空")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if qwen3_post_trained is None:
        qwen3_post_trained = is_qwen3_post_trained(args.model)
    if qwen3_post_trained:
        print("检测到后训练的 Qwen3 模型，将使用 enable_thinking=False")

    if model is None:
        seed = getattr(args, "seed", None)
        if seed is None:
            try:
                seed = int(args.suffix)
            except (TypeError, ValueError):
                seed = abs(hash(str(args.suffix))) % (2**31)
        model = vllm.LLM(
            model=args.model,
            tokenizer=args.model,
            gpu_memory_utilization=args.gpu_utilization,
            seed=seed,
        )
    sample_params = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.rollout_n,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    prompt_texts = _prompt_strings(data_records, tokenizer, qwen3_post_trained)
    completions: List[RequestOutput] = model.generate(
        prompt_texts, sampling_params=sample_params, use_tqdm=False
    )

    n = len(completions)
    assert n == len(data_records), (
        f"vLLM 返回条数 {n} 与 data_records {len(data_records)} 不一致"
    )

    results: List[Dict[str, Any]] = []
    for i in tqdm(range(n), total=n, desc="Rollout results"):
        completion = completions[i]
        row = data_records[i]
        responses = [output.text for output in completion.outputs]
        answers = [extract_boxed_content(response) for response in responses]
        gt = _gt_from_row(row)
        is_right = [custom_grade_answer(rsp, gt) for rsp in responses]
        raw_q_acc = sum(is_right) / len(is_right) if len(is_right) > 0 else 0.0
        assert len(responses) == len(is_right), (
            f"responses and is_right must have the same length, but got {len(responses)} and {len(is_right)}"
        )
        results.append(
            {
                "q": row['q'],
                "responses": responses,
                "answers": answers,
                "gt": gt,
                "is_right": is_right,
                "acc": raw_q_acc,
            }
        )

    return results
