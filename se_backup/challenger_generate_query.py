import vllm
import torch
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
import os, sys
import random
import json
import regex as re
"""
PROMPT_DICT = {
        "algebra": {
            "topic": "代数",
            "description": "线性代数、多项式、方程求解、函数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert algebra problem setter specializing in advanced algebraic concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial algebra problem. "
                        "The problem should involve concepts like polynomial manipulation, equation solving, function analysis, "
                        "linear algebra, or advanced algebraic structures. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging algebra question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "geometry": {
            "topic": "几何",
            "description": "平面几何、立体几何、解析几何、三角学等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert geometry problem setter specializing in advanced geometric concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial geometry problem. "
                        "The problem should involve concepts like plane geometry, solid geometry, coordinate geometry, "
                        "trigonometry, or advanced geometric constructions. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging geometry question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "number_theory": {
            "topic": "数论",
            "description": "整数性质、素数、同余、数论函数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert number theory problem setter specializing in advanced number-theoretic concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial number theory problem. "
                        "The problem should involve concepts like divisibility, prime numbers, modular arithmetic, "
                        "Diophantine equations, or advanced number-theoretic functions. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging number theory question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "combinatorics": {
            "topic": "组合数学",
            "description": "排列组合、图论、计数原理、生成函数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert combinatorics problem setter specializing in advanced combinatorial concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial combinatorics problem. "
                        "The problem should involve concepts like permutations, combinations, graph theory, "
                        "counting principles, generating functions, or advanced combinatorial structures. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging combinatorics question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "probability": {
            "topic": "概率论",
            "description": "概率计算、条件概率、随机变量、概率分布等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert probability problem setter specializing in advanced probabilistic concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial probability problem. "
                        "The problem should involve concepts like probability calculations, conditional probability, "
                        "random variables, probability distributions, or advanced probabilistic reasoning. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging probability question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "statistics": {
            "topic": "统计学",
            "description": "描述统计、推断统计、假设检验、回归分析等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert statistics problem setter specializing in advanced statistical concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial statistics problem. "
                        "The problem should involve concepts like descriptive statistics, inferential statistics, "
                        "hypothesis testing, regression analysis, or advanced statistical methods. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging statistics question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "calculus": {
            "topic": "微积分",
            "description": "极限、导数、积分、微分方程等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert calculus problem setter specializing in advanced calculus concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial calculus problem. "
                        "The problem should involve concepts like limits, derivatives, integrals, "
                        "differential equations, or advanced calculus applications. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging calculus question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "trigonometry": {
            "topic": "三角学",
            "description": "三角函数、三角恒等式、三角方程、反三角函数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert trigonometry problem setter specializing in advanced trigonometric concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial trigonometry problem. "
                        "The problem should involve concepts like trigonometric functions, trigonometric identities, "
                        "trigonometric equations, inverse trigonometric functions, or advanced trigonometric applications. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging trigonometry question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "complex_numbers": {
            "topic": "复数",
            "description": "复数运算、复平面、复数方程、欧拉公式等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert complex numbers problem setter specializing in advanced complex analysis concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial complex numbers problem. "
                        "The problem should involve concepts like complex arithmetic, complex plane geometry, "
                        "complex equations, Euler's formula, or advanced complex analysis. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging complex numbers question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        },
        
        "sequences_series": {
            "topic": "数列与级数",
            "description": "等差数列、等比数列、数列极限、无穷级数等",
            "chat": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert sequences and series problem setter specializing in advanced sequence and series concepts.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial sequences and series problem. "
                        "The problem should involve concepts like arithmetic sequences, geometric sequences, "
                        "sequence limits, infinite series, convergence tests, or advanced series analysis. "
                        "Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging sequences and series question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        }
    }
"""
PROMPT_DICT={
    "R-Zero": {
            "topic": "raw_math",
            "description": "R-Zero原始的prompt",
            "chat": [
                {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter.\n"
                "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                "Avoid re-using textbook clichés or famous contest problems.\n"
                "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                "<question>\n"
                "{The full problem statement on one or more lines}\n"
                "</question>\n\n"
                r"\boxed{final_answer}"
                "\n\n"
                "Do NOT output anything else—no explanations, no extra markup."
            )
        },
        {
            "role": "user",
            "content": (
                "Generate one new, challenging reasoning question now. "
                "Remember to format the output exactly as instructed."
            )
        }
    ]
}
}

def extract_boxed(text):
    results, i = [], 0
    prefix = r'\boxed{'
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break   # no more \boxed{…}

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        results.append(text[start + plen : j - 1])
        i = j

    return results

def get_response_mask(response_ids, eos_token_id, dtype):
    batch_size, seq_len = response_ids.shape
    mask = torch.ones((batch_size, seq_len), dtype=dtype)
    for i in range(batch_size):
        for j in range(seq_len):
            if response_ids[i][j] == eos_token_id:
                mask[i][j:] = 0
                break
    return mask


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.95,
        seed=int(args.suffix),
    )
    
    prompts = []
    topic_mapping = []  # 记录每个prompt对应的topic
    
    for topic in PROMPT_DICT.keys():
        chat = PROMPT_DICT[topic]['chat']
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                chat, 
                tokenize=False,
                add_generation_prompt=True, 
                add_special_tokens=True
            )
        else:
            prompt = "system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"]
        prompts.append(prompt)
        topic_mapping.append(topic)
        
        
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    
    # 为每个样本重复所有topic的prompts
    repeated_prompts = prompts * args.num_samples
    repeated_topics = topic_mapping * args.num_samples
    
    completions: List[RequestOutput] = model.generate(repeated_prompts, sampling_params=sample_params,use_tqdm=False)
    results=[]
    
    for idx, completion in enumerate(completions):
        response = completion.outputs[0].text
        topic = repeated_topics[idx]  # 获取对应的topic
        
        try:
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            answers = extract_boxed(response)

            if questions and answers:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({
                    "idx": idx, 
                    "topic": topic,
                    "question": question, 
                    "answer": answer, 
                    "score": 0
                })
            else:
                results.append({
                    "idx": idx, 
                    "topic": topic,
                    "question": response, 
                    "answer": "", 
                    "score": -1
                })
        except:
            results.append({
                "idx": idx, 
                "topic": topic,
                "question": response, 
                "answer": "", 
                "score": -1
            })
    random.shuffle(results)
    os.makedirs(args.storage_path, exist_ok=True)
    with open(f"{args.storage_path}/{args.suffix}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_samples", type=int, default=1250, help="Number of samples to generate")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output file")
    parser.add_argument("--storage_path", type=str, default="/root/users/ycy/Self-evolving-Agent/saved_results/Solver/Qwen3-4B-Base-V1", help="")
    #parser.add_argument("--save_name", type=str, default="challenger_generated_question", help="")
    args = parser.parse_args()

    main(args) 