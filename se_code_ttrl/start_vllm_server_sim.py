#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Refactored Version: This script employs the 'stopit' library to apply fine-grained, thread-safe
timeout control directly to the `grade_answer` function. This approach is more robust than a
global timeout and avoids the 'signal only works in main thread' error common in multi-threaded
Flask applications. The comparison logic is optimized to perform cheap checks first.

Setup Instructions:
    # 1. Install the required library (note the change from previous versions)
    pip install stopit

    # 2. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

import os
from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import numpy as np
import threading
import time
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch.nn.functional as F
from se_code.utils import format_check, extract_boxed_content, entropy_from_logits, sigmoid
from collections import defaultdict,Counter
from mathruler.grader import grade_answer
import stopit  # 1. Import the thread-safe 'stopit' library
import math
# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.6,
                    help='The maximum GPU memory utilization fraction for vLLM.')
args = parser.parse_args()

# ------------------------- Global Variables ------------------------ #
# Initialize these in main to avoid multiprocessing issues
tokenizer = None
model = None
sample_params = None

# ---------------------- GPU Idle Utilization Thread ---------------------- #
# Initialize threading events first
stop_event = threading.Event()    # Event to stop the thread globally
pause_event = threading.Event()   # Event to pause the thread during requests


    
def initialize_model():
    """Initialize vLLM model and related components"""
    global tokenizer, model, sample_params,entropy_model
    print('[init] Loading model...')
    
    # Pause GPU idle worker during model initialization to avoid CUDA graph conflicts
    print('[init] Pausing GPU idle worker for model initialization...')
    pause_event.set()
    torch.cuda.synchronize()
    time.sleep(1)  # Give the idle worker time to pause
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = vllm.LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            gpu_memory_utilization=args.gpu_mem_util,
        )
        entropy_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        entropy_model.eval()
        print('[init] Entropy model loaded successfully!')
        
        sample_params = vllm.SamplingParams(
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            top_k=40,
            stop_token_ids=[tokenizer.eos_token_id],
            n=10, # Generate 10 candidate answers for each question
        )
        print('[init] Model loaded successfully!')
    finally:
        # Resume GPU idle worker after model initialization
        print('[init] Resuming GPU idle worker...')
        pause_event.clear()

def gpu_idle_worker():
    '''
    This worker occupies the GPU with a continuous matrix multiplication loop when idle,
    preventing potential performance drops from GPU power state changes.
    '''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1) # Sleep briefly while paused
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            # A simple but effective way to keep the GPU busy
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

# Initialize the idle thread but don't start it yet
idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)



# ------------------------ Timeout Utility (Refactored) --------------------------- #
# 2. Use the 'stopit.threading_timeoutable' decorator for thread-safe timeouts.
#    It returns a default value on timeout instead of raising an exception.
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    This wrapper applies a timeout to each individual `grade_answer` call.
    If the function's execution exceeds the specified timeout, it will return 'TIMED_OUT'.
    The timeout duration is passed as a keyword argument during the function call.
    """
    return grade_answer(res1, res2)

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()
    
    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    # ---------- Load Data ----------
    try:
        if not os.path.exists(name):
            print(f'[server] ERROR: Task file {name} does not exist')
            return jsonify({'error': f'Task file {name} does not exist'}), 404
        
        # 等待文件完全写入（避免读取不完整的文件）
        import time
        max_wait = 5  # 最多等待5秒
        wait_time = 0
        while wait_time < max_wait:
            try:
                with open(name, 'r') as f:
                    data = json.load(f)
                break
            except (json.JSONDecodeError, IOError) as e:
                if wait_time < max_wait - 1:
                    print(f'[server] File {name} not ready, waiting... ({wait_time+1}s)')
                    time.sleep(1)
                    wait_time += 1
                else:
                    raise e
        
        if not data:
            print(f'[server] ERROR: Task file {name} is empty')
            return jsonify({'error': f'Task file {name} is empty'}), 400
            
    except Exception as e:
        print(f'[server] ERROR: Failed to load task file {name}: {e}')
        return jsonify({'error': f'Failed to load task file: {str(e)}'}), 500
    
    questions = [item.get('question', '') for item in data]

    # (Data preparation logic remains unchanged)
    valid_indices, valid_questions, valid_chats = [], [], []
    instructions = 'Please reason step by step, and put your final answer within \\boxed{}.'
    for i,q in enumerate(questions):
        if q:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_chats.append([
                {'role': 'system',   'content': instructions},
                {'role': 'user',   'content': q}
            ])
    print('[server] Valid chat prompts have been prepared.')

    # ---------- vLLM Generation ----------
    # (vLLM generation logic remains unchanged)
    
    if valid_chats:
        print(f'[server] Generating responses for {len(valid_chats)} valid chats')
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in valid_chats
            ]
        else:
            prompts = [
                'system: You are a helpful assistant.' + '\n' + 'user: ' + chat[0]['content']
                for chat in valid_chats
            ]
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=False)
        print(f'[server] Generated {len(responses)} responses')
    else:
        responses = []
        print('[server] No valid chats, responses list is empty')
    print('[server] Generation completed.')

    # ---------- Results Post-Processing (Core Refactoring & Optimization Here) ----------
    def process_single(idx,question, response):
        '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
        
        results = []
        all_labels = []
        responses_str=[]
        repetition_rates=[]
        repetition_spans=[]
        repetion_penalty=[]
        entropies_per_rollout=[]
        rsp_lengths=[]
        valid_labels=[]
        boxed_format=[]

        for out in response.outputs:
            responses_str.append(out.text)
            """
            result = {
                    "has_boxed": len(all_boxed) > 0,
                    "boxed_count": len(all_boxed),
                    "all_boxed_answers": all_boxed,
                    "final_answer": all_boxed[-1] if all_boxed else None,
                    "repetition_rate": repetition_rate,
                    "repetition_spans": rep_spans
                }
            """

            format_result = format_check(out.text)
            if format_result['final_answer'] is not None:
                valid_labels.append(format_result['final_answer'])
                all_labels.append(format_result['final_answer'])
            else:
                all_labels.append(None)
            rsp_length = len(out.token_ids)
            repetition_rates.append(format_result['repetition_rate'])
            repetition_spans.append(format_result['repetition_spans'])
            repetion_penalty.append(format_result['repetition_rate'] >= 0.1)
            boxed_format.append(format_result['boxed_count'] > 0 and format_result['boxed_count'] < 3)

            rsp_lengths.append(rsp_length)
            
            entropy_val = None
            try:
                # 1) 复原用于生成的 prompt 文本（与上面生成逻辑保持一致）
                if tokenizer.chat_template:
                    chat_for_this_q = [
                        {'role': 'system', 'content': instructions},
                        {'role': 'user',   'content': question},
                    ]
                    prompt_text = tokenizer.apply_chat_template(
                        chat_for_this_q,
                        tokenize=False,
                        add_generation_prompt=True,
                        add_special_tokens=True
                    )
                else:
                    # 注意：这里严格复用你上面非 chat_template 分支的写法
                    # （即把 system 指令当作 user 内容的那一版，保持一致性）
                    prompt_text = 'system: You are a helpful assistant.\nuser: ' + instructions

                # 2) 分别拿到 prompt 的 token ids 和 生成的 token ids
                prompt_ids = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids[0].to('cuda')
                gen_ids = torch.tensor(out.token_ids, dtype=torch.long, device='cuda')
                if gen_ids.numel() == 0:
                    raise ValueError("empty generated token ids")

                # 3) 构造用于预测“每个生成 token”的输入：
                #    第 k 个生成 token 的分布，输入应为 [prompt + 前 k 个生成 token]
                #    等价做法：一次性前向，把输入设为 [prompt + 生成的前 (L-1) 个 token]，
                #    这样 logits 的切片就依次对应每个生成 token 的预测分布。
                #    若只有 1 个 token，则输入就是 prompt（不拼生成前缀）。
                input_ids_full = torch.cat([prompt_ids, gen_ids[:-1]], dim=0) if gen_ids.size(0) > 1 else prompt_ids

                with torch.no_grad():
                    outputs = entropy_model(input_ids_full.unsqueeze(0))   # [1, T_in, V]
                    logits_full = outputs.logits[0]                        # [T_in, V]

                # 4) 取出用于预测生成 token 的那段 logits 切片：
                #    第一个生成 token 对应的分布在位置 (prompt_len - 1)，
                #    一共需要 gen_len 个步的分布。
                prompt_len = prompt_ids.size(0)
                gen_len = gen_ids.size(0)
                start = max(0, prompt_len - 1)
                end = start + gen_len
                logits_steps = logits_full[start:end]                     # [gen_len, V]

                # 5) 用你现有的 entropy_from_logits 计算整段平均熵
                entropy_val = entropy_from_logits(logits_steps).mean().item()
            except Exception as e:
                print(f"[entropy] Failed to compute entropy for rollout: {e}")
                entropy_val = None
            entropies_per_rollout.append(entropy_val)
        answer_counts = {}
        for res in list(valid_labels):
            if not res: continue # Skip empty results
            matched = False
            
            for exist_ans in list(set(answer_counts.keys())):
                # 3. OPTIMIZATION: Perform cheap comparisons first to avoid expensive calls.
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                    answer_counts[exist_ans] += 1
                    matched = True
                    break # Match found, break from the inner loop over exist_ans
                
                # 4. If cheap checks fail, proceed to the expensive, timed grade_answer calls.
                try:
                    is_match = False
                    # First direction: res vs exist_ans
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                    if match_result_1 == 'TIMED_OUT':
                        print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                    elif match_result_1:
                        is_match = True

                    # Second direction (only if first failed): exist_ans vs res
                    if not is_match:
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                        if match_result_2 == 'TIMED_OUT':
                             # Log timeout for the second direction as well
                            print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                        elif match_result_2:
                            is_match = True
                    
                    if is_match:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break # Match found, break from the inner loop

                except Exception as e:
                    # Catch any other potential errors from the grader function itself.
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                    continue # Continue to the next comparison in the inner loop
            
            if not matched:
                answer_counts[res] = 1
        assert len(rsp_lengths) == len(all_labels) == len(repetition_rates) == len(repetition_spans) == len(entropies_per_rollout), "length mismatch"
        total = sum(answer_counts.values())
        if total != len(valid_labels):
            print(f"[WARN] Mismatch: total({total}) vs valid_labels({len(valid_labels)})")
        
        # 答案分布（经验概率）
        if total > 0:
            eps = 1e-12
            p_answers = {ans: c / total for ans, c in answer_counts.items()}
            # 答案信息熵（预测熵）
            H_pred = -sum(p * math.log(p + eps) for p in p_answers.values())
            K = max(1, len(answer_counts))
            H_pred_norm = H_pred / (math.log(K) + eps)
            # 2️⃣ 内部熵：每个 solution 的平均句子熵
            entropies_valid = [e for i, e in enumerate(entropies_per_rollout) if e is not None and all_labels[i] is not None]
            H_intra = float(sum(entropies_valid) / len(entropies_valid)) if len(entropies_valid) > 0 else 0.0
            H_intra_norm = np.tanh(H_intra)

            # 3️⃣ BALD = 预测熵 - 内部熵
            #BALD = sigmoid(H_pred - H_intra)
            BALD = H_pred_norm * (1 - H_intra_norm)

            
        else:
            H_pred=0.0
            H_intra=0.0
            BALD=0.0
        rep_ratio = sum(repetion_penalty) / len(repetion_penalty)
        repetition_format_penalty = max(0.1, (1 - rep_ratio)*0.5)
        boxed_ratio = sum(boxed_format) / len(boxed_format)
        boxed_format_penalty = max(0.1, boxed_ratio)
        format_penalty = repetition_format_penalty * boxed_format_penalty
        
        avg_rsp_length = sum(rsp_lengths) / len(rsp_lengths)
        length_norm = avg_rsp_length / (4096 / 2)
        length_lambda=0.5
        length_reward = length_lambda*length_norm # 越长越好

        reward = (BALD + length_reward) * format_penalty # 熵越大越好，长度越长越好，格式越规范越好
        
        return {
            'idx':idx,
            'question': question,
            'responses': responses_str,
            'reward_info':{
                'rep_ratio':rep_ratio,
                'boxed_ratio':boxed_ratio,
                'format_penalty':format_penalty,
                'format_penalty':format_penalty,
                'length_norm':length_norm,
                'length_reward':length_reward,
                'entropy_lst': entropies_per_rollout,
                'H_pred_norm': H_pred_norm,
                'H_intra_norm': H_intra_norm,
                'BALD':BALD,
                'reward':reward,
            },
            'answer_counts': answer_counts,
            'all_labels': all_labels,
            'valid_labels': list(answer_counts.keys()),
            'valid_labels_cnt': sum(list(answer_counts.values())),
            'all_labels_cnt': len(all_labels),
            'reward':reward,
            'error':None,
        }

    results_all = []
    response_idx = 0
    print(f'[server] Processing {len(valid_chats)} questions with {len(responses)} responses')
    
    for idx, q in enumerate(questions):
        try:
            if q:
                item = process_single(idx, q,responses[response_idx])
                results_all.append(item)
                response_idx += 1
            else:
                results_all.append({
                    'idx':idx,
                    'question': '',
                    'responses': '',
                    'reward_info':{},
                    'answer_counts': 0,
                    'all_labels': [],
                    'valid_labels': [],
                    'valid_labels_cnt': 0,
                    'all_labels_cnt': 0,
                    'reward':0.0,
                    'error': f'no question:{q=}, skipping...'
                }
            )
        except Exception as e:
            # Catch any other unexpected exceptions from within process_single.
            print(f'[server] CRITICAL: An unhandled error occurred in files:{name} while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                    'idx':idx,
                    'question': '',
                    'responses': '',
                    'reward_info':{},
                    'answer_counts': 0,
                    'all_labels': [],
                    'valid_labels': [],
                    'valid_labels_cnt': 0,
                    'all_labels_cnt': 0,
                    'reward':0.0,
                    'error': f'[server] CRITICAL: An unhandled error occurred in files:{name} while processing question: {q}'
                }
                )
    print('[server] All results have been processed.')

    # 保存结果文件
    out_path = name.replace('.json', '_results.json')
    try:
        with open(out_path, 'w') as f:
            json.dump(results_all, f, indent=4)
        print(f'[server] Results saved to {out_path}')
        
        # 只有在成功保存结果后才删除原始任务文件
        if os.path.exists(name):
            os.remove(name)
            print(f'[server] Cleaned up task file: {name}')
    except Exception as e:
        print(f'[server] ERROR: Failed to save results to {out_path}: {e}')
        return jsonify({'error': f'Failed to save results: {str(e)}'}), 500

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    torch.cuda.synchronize()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'status': 'success', 'results_count': len(results_all)})

# ------------------------- Main Application Entrypoint --------------------------- #
if __name__ == '__main__':
    try:
        # Start the GPU idle worker thread
        idle_thread.start()
        print('[main] GPU idle worker thread started.')
        
        # Initialize model before starting the server
        initialize_model()
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')