#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import multiprocessing
import os
import signal
import threading
import time

multiprocessing.set_start_method("spawn", force=True)

import vllm
from flask import Flask, jsonify, request
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="5000")
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Base")
parser.add_argument(
    "--gpu_mem_util",
    type=float,
    default=0.95,
    help="Maximum GPU memory utilization fraction for vLLM.",
)
parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default=1,
    help="Tensor parallel size; must match the number of GPUs vLLM sees "
    "(set CUDA_VISIBLE_DEVICES before launch to pick physical GPUs).",
)
parser.add_argument(
    "--no_enforce_eager",
    action="store_true",
    help="Allow vLLM CUDA graphs (faster). Default uses enforce_eager=True to avoid "
    "memory-profiling AssertionError on some multi-GPU / dirty-GPU setups.",
)
parser.add_argument(
    "--shutdown_token",
    type=str,
    default="",
    help="If set, POST /shutdown must send header X-Shutdown-Token with this value.",
)
args = parser.parse_args()

tokenizer = None
model = None


def initialize_model():
    global tokenizer, model
    print("[init] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    enforce_eager = not args.no_enforce_eager
    if enforce_eager:
        print("[init] enforce_eager=True (disable with --no_enforce_eager)")
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=enforce_eager,
    )
    print("[init] Model loaded.")


def _build_sampling_params(body):
    kwargs = {
        "max_tokens": int(body.get("max_tokens", 4096)),
        "temperature": float(body.get("temperature", 1.0)),
        "top_p": float(body.get("top_p", 1.0)),
        "top_k": int(body.get("top_k", 40)),
        "n": int(body.get("n", 1)),
    }
    if tokenizer and tokenizer.eos_token_id is not None:
        kwargs["stop_token_ids"] = [tokenizer.eos_token_id]
    return vllm.SamplingParams(**kwargs)


def _chat_enable_thinking(chat):
    for m in reversed(chat):
        if m.get("role") == "user":
            return "/think" in (m.get("content") or "")
    return False


def _messages_to_prompt(messages):
    assert tokenizer is not None
    chat = [{"role": m["role"], "content": m["content"]} for m in messages]
    if tokenizer.chat_template:
        kw = dict(
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
        try:
            return tokenizer.apply_chat_template(
                chat, enable_thinking=_chat_enable_thinking(chat), **kw
            )
        except TypeError:
            return tokenizer.apply_chat_template(chat, **kw)
    parts = []
    for m in chat:
        parts.append(f"{m['role']}: {m['content']}")
    return "\n".join(parts)


app = Flask(__name__)


def _schedule_process_exit():
    """HTTP 返回后再结束进程，避免客户端收不到响应。"""

    def _run():
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=_run, daemon=True).start()


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """供 rollout 结束后调用，结束本进程（含 vLLM worker）。"""
    if args.shutdown_token:
        sent = request.headers.get("X-Shutdown-Token", "")
        if sent != args.shutdown_token:
            return jsonify({"error": "invalid or missing X-Shutdown-Token"}), 403
    print("[server] shutdown requested via POST /shutdown")
    _schedule_process_exit()
    return jsonify({"status": "shutting_down"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate():
    if model is None or tokenizer is None:
        return jsonify({"error": "model not loaded"}), 503

    body = request.get_json(silent=True) or {}
    sampling = _build_sampling_params(body)

    prompts = []
    if "messages" in body:
        prompts.append(_messages_to_prompt(body["messages"]))
    elif "prompt" in body:
        prompts.append(str(body["prompt"]))
    elif "prompts" in body:
        prompts = [str(p) for p in body["prompts"]]
    else:
        return (
            jsonify(
                {
                    "error": "provide one of: messages (chat), prompt (str), prompts (list[str])"
                }
            ),
            400,
        )

    try:
        outputs = model.generate(prompts, sampling_params=sampling, use_tqdm=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    def serialize_output(out):
        texts = [c.text for c in out.outputs]
        if sampling.n == 1:
            return texts[0] if len(texts) == 1 else texts
        return texts

    if len(outputs) == 1:
        return jsonify({"text": serialize_output(outputs[0])})
    return jsonify({"texts": [serialize_output(o) for o in outputs]})


if __name__ == "__main__":
    initialize_model()
    app.run(host="127.0.0.1", port=int(args.port), threaded=True)
