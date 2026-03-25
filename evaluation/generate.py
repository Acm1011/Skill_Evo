import vllm
import argparse
from se.evaluation.datasets_loader import get_dataset_handler
from transformers import AutoTokenizer
import json
import os
from math_verify import parse, verify
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#STORAGE_PATH = os.getenv("STORAGE_PATH")
#/users/ycy/saved_results'

def main(args):
    save_path=os.path.join(args.save_path_dir, args.model_name)
    os.makedirs(save_path, exist_ok=True)
    print(f'eval model: {args.model_name} performance on the math dataset: {args.dataset}, eval results will be saved in {save_path}')
    print(args.model_name, args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=0.85
    )
    if "aime" in args.dataset or 'amc' in args.dataset:
        sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    else:
        sample_params = vllm.SamplingParams(
            max_tokens=4096,
            temperature=0.0,
            stop_token_ids=[tokenizer.eos_token_id],
        )
    handler = get_dataset_handler(args.dataset,args.name)
    questions, answers = handler.load_data()
    chats=[[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": question}] for question in questions]
    if tokenizer.chat_template:
        prompts = [tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=True, add_special_tokens=True) for chat in chats]
    else:
        prompts = ["system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] + '\nPlease reason step by step, and put your final answer within \\boxed{}.' for chat in chats]
    responses = model.generate(prompts, sampling_params=sample_params,use_tqdm=False)
    responses = [response.outputs[0].text for response in responses]
    scores, preds,average_score = handler.get_score(responses, answers)
    results = [
        {"question": question,"prompt": prompt, "answer": answer, "response": response, "score": score, "pred": pred} 
        for question, answer, response, score, prompt, pred 
        in zip(questions, answers, responses, scores,prompts, preds)
    ]
    print(f"Average score: {average_score}")
    results.append({"average_score": average_score})

    with open(f"{save_path}/{args.dataset}_rule_based_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--model_name", type=str, default="Qwen3-4B-Base")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--save_path_dir", type=str, default="/root/users/ycy/saved_results/evaluation")
    args = parser.parse_args()
    main(args)