import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams


def get_max_position_embeddings(model_path):
    try:
        config = AutoConfig.from_pretrained(model_path)
        return getattr(config, "max_position_embeddings", None)
    except Exception as e:
        print(f"Warning: 无法读取模型配置: {e}")
        return None


def is_qwen3_post_trained(model_path):
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


def extract_step_from_path(model_path):
    match = re.search(r"global_step_(\d+)", model_path)
    if match:
        return int(match.group(1))
    return None


def extract_step_from_name(model_name):
    match = re.search(r"-step(\d+)$", model_name)
    if match:
        return int(match.group(1))
    return None


def extract_last_boxed(text):
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text):
    pattern1 = r"Final Answer:((?:[^<]|<[^<])*?)\n"
    pattern2 = r"The answer is:((?:[^<]|<[^<])*?)\n"
    matches1 = list(re.finditer(pattern1, text))
    matches2 = list(re.finditer(pattern2, text))
    if matches1:
        return matches1[-1].group(1)
    if matches2:
        return matches2[-1].group(1)
    return None


def extract_solution(solution_str):
    if "<|im_start|>user" in solution_str:
        model_output = re.sub(
            r"^.*?<\|im_start\|>assistant",
            "<|im_start|>assistant",
            solution_str,
            flags=re.DOTALL,
            count=1,
        )
    elif "Assistant:" in solution_str:
        model_output = solution_str.split("Assistant:")[-1].strip()
    else:
        model_output = solution_str

    for stop_word in ["</s>", "<|im_end|>", "<|endoftext|>"]:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    boxed = extract_last_boxed(model_output)
    if boxed:
        return boxed
    return extract_last_final_answer(model_output)


def form_options(options):
    option_str = "Options are:\n"
    for opt, letter in zip(options, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]):
        option_str += f"({letter}): {opt}\n"
    return option_str


def get_prediction(output):
    solution = extract_solution(output)
    if solution is None:
        return random.choice(["A", "B", "C", "D"])
    for option in ["A", "B", "C", "D"]:
        if option in solution:
            return option
    return random.choice(["A", "B", "C", "D"])


def build_examples(dataset):
    examples = []
    for row in dataset:
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        random.shuffle(options)
        answer_letter = chr(65 + options.index(row["Correct Answer"]))
        examples.append(
            {
                "question": row["Question"],
                "options": options,
                "answer_letter": answer_letter,
            }
        )
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPQA 评测脚本（支持按 step 保存）")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--save_path_dir", type=str, default="/home/ycy/sdi/saved_results/evaluation")
    parser.add_argument("--data_path_dir", type=str, default="/home/ycy/data1/data")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=0.1)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    step = args.step
    if step is None:
        step = extract_step_from_path(args.model_path)
    if step is None:
        step = extract_step_from_name(args.model_name)

    if step is not None:
        save_dir = os.path.join(args.save_path_dir, f"step_{step}")
        display_name = f"{args.model_name} (step {step})"
    else:
        save_dir = os.path.join(args.save_path_dir, args.model_name)
        display_name = args.model_name
    os.makedirs(save_dir, exist_ok=True)

    final_results_file = os.path.join(save_dir, "gpqa_final_results.json")
    if os.path.exists(final_results_file):
        print(f"结果已存在，跳过: {final_results_file}")
        raise SystemExit(0)

    print("=" * 60)
    print("GPQA 评测")
    print(f"  模型: {display_name}")
    print(f"  模型路径: {args.model_path}")
    print(f"  Step: {step if step is not None else 'N/A'}")
    print(f"  保存目录: {save_dir}")
    print(f"  采样比例: {args.sample_ratio}")
    print("=" * 60)

    if args.output_file is None:
        args.output_file = os.path.join(save_dir, "gpqa_outputs.json")

    local_candidates = [
        os.path.join(args.data_path_dir, "gpqa"),
        os.path.join(args.data_path_dir, "gpqa_diamond"),
        os.path.join(args.data_path_dir, "GPQA"),
    ]
    data_path = None
    for candidate in local_candidates:
        if os.path.exists(candidate):
            data_path = candidate
            break
    if data_path is None:
        data_path = ("Idavidrein/gpqa", "gpqa_diamond")
        print("Warning: 本地数据集不存在，从 HuggingFace 加载: Idavidrein/gpqa/gpqa_diamond")
    else:
        print(f"从本地加载: {data_path}")

    if isinstance(data_path, tuple):
        dataset = datasets.load_dataset(data_path[0], data_path[1], split="train")
    else:
        try:
            dataset = datasets.load_dataset(data_path, "gpqa_diamond", split="train")
        except Exception:
            dataset = datasets.load_dataset(data_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    max_pos_emb = get_max_position_embeddings(args.model_path)
    qwen3_post_trained = is_qwen3_post_trained(args.model_path)
    if qwen3_post_trained:
        print("检测到后训练的 Qwen3 模型，将使用 enable_thinking=False")

    if max_pos_emb is not None and max_pos_emb <= 4096:
        FILTER_PROMPT = True
        MAX_PROMPT_LEN = 2048
        MAX_TOKENS = 2048
    elif max_pos_emb is not None and max_pos_emb <= 8192:
        FILTER_PROMPT = True
        MAX_PROMPT_LEN = 2048
        MAX_TOKENS = 6144
    else:
        FILTER_PROMPT = False
        MAX_PROMPT_LEN = None
        MAX_TOKENS = 8192
    print(
        f"模型 max_position_embeddings={max_pos_emb}，"
        f"{'启用 prompt 过滤' if FILTER_PROMPT else '不过滤 prompt'}"
    )

    llm = LLM(model=args.model_path, tensor_parallel_size=1, gpu_memory_utilization=0.85)

    print("----------------- 开始评测 -------------------")
    random.seed(42)
    sample_ratio = max(0.0, min(1.0, args.sample_ratio))
    examples = build_examples(dataset)
    total_raw_samples = len(examples)
    target_samples = (
        max(1, int(total_raw_samples * sample_ratio))
        if 0.0 < sample_ratio < 1.0 and total_raw_samples > 0
        else total_raw_samples
    )

    valid_examples = []
    valid_prompts = []
    filtered_count = 0

    for example in examples:
        query = example["question"] + "\n" + form_options(example["options"]) + "\n"
        messages = [
            {
                "role": "user",
                "content": query
                + "\nPlease reason step by step, and put your final answer option within \\boxed{}. "
                + "Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer.",
            }
        ]
        if tokenizer.chat_template:
            if qwen3_post_trained:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = (
                "user: "
                + query
                + "\nPlease reason step by step, and put your final answer option within \\boxed{}. "
                + "Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer."
            )

        if FILTER_PROMPT:
            prompt_len = len(tokenizer.encode(prompt))
            if prompt_len > MAX_PROMPT_LEN:
                filtered_count += 1
                continue

        valid_examples.append(example)
        valid_prompts.append(prompt)

    if target_samples < len(valid_examples):
        indices = random.sample(range(len(valid_examples)), target_samples)
        eval_examples = [valid_examples[i] for i in indices]
        eval_prompts = [valid_prompts[i] for i in indices]
    else:
        eval_examples = valid_examples
        eval_prompts = valid_prompts

    print(f"原始样本数: {total_raw_samples}")
    print(f"目标采样数: {target_samples}")
    print(f"过滤长 prompt: {filtered_count}")
    print(f"实际评测数: {len(eval_examples)}")

    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=MAX_TOKENS)
    outputs = llm.generate(eval_prompts, sampling_params)

    success = 0
    fail = 0
    results = []

    def process_entry(entry_output_pair):
        entry, output = entry_output_pair
        answer = output.outputs[0].text
        prediction = get_prediction(answer)
        is_correct = entry["answer_letter"] == prediction
        return {
            "question": entry["question"],
            "options": entry["options"],
            "gold": entry["answer_letter"],
            "prediction": prediction,
            "solution": answer,
            "correct": is_correct,
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_entry, pair) for pair in zip(eval_examples, outputs)]
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            if item["correct"]:
                success += 1
            else:
                fail += 1

    overall_accuracy = success / (success + fail) if (success + fail) > 0 else 0.0
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(final_results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "gpqa",
                "model": args.model_name,
                "model_path": args.model_path,
                "step": step,
                "accuracy": round(overall_accuracy * 100, 2),
                "success": success,
                "fail": fail,
                "total": success + fail,
                "sample_ratio": args.sample_ratio,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"结果已保存: {final_results_file}")
