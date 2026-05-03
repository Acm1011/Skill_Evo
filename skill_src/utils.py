import os
import torch
import re
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math
from math_verify import verify
import random
from mathruler.grader import grade_answer
from mathruler.math_normalize import normalize_answer, _strip_string
import stopit
import difflib
import pandas as pd

INSTRUCTIONS = "Please reason step by step, and put your final answer within the \\boxed{}."
def approx_entropy_from_logprobs(out):
    """
    近似整段 response 的熵：
    - vLLM 的 out.logprobs 是一个 list[dict]，每步 token 的 top-k logprob。
    - 我们用这些 top-k 概率计算每步的部分熵，并加上一个“其余概率桶”近似。
    返回：该回答的平均 token 熵（float）或 None
    """
    if not hasattr(out, "logprobs") or out.logprobs is None:
        return None

    token_entropies = []
    for lp_dict in out.logprobs:
        if not lp_dict:
            continue
        # 已知 top-k 概率
        ps = [math.exp(lp) for lp in lp_dict.values()]
        p_known = sum(ps)
        # 计算已知部分的熵
        h_known = -sum(p * math.log(p) for p in ps if p > 0.0)
        # 把剩余概率聚合成一个桶（下界近似）
        p_rest = max(0.0, 1.0 - p_known)
        h_rest = -(p_rest * math.log(p_rest)) if p_rest > 0.0 else 0.0
        token_entropies.append(h_known + h_rest)

    if not token_entropies:
        return None
    # 取所有 token 熵的平均值，代表整段回答的不确定度
    return float(sum(token_entropies) / len(token_entropies))

def extract_all_boxed_content(text: str) -> List[str]:
    """
    提取文本中所有的 \\boxed{} 内容
    返回一个列表，包含所有找到的boxed内容
    
    注意：使用括号深度计数而不是正则表达式，因为：
    1. 正则表达式无法正确处理嵌套的花括号
    2. 数学表达式中经常有嵌套，如 \\boxed{\\frac{1}{2}} 或 \\boxed{\\{a,b\\}}
    
    示例：
        text = r"Answer is \\boxed{\\frac{1}{2}}"
        # 正则 r'\\\\boxed\\{([^}]+)\\}' 只会匹配到 "\\frac{1" (错误!)
        # 本函数会正确匹配到 "\\frac{1}{2}"
    """
    boxed_contents = []
    search_start = 0
    
    while True:
        # 从当前位置开始查找下一个\boxed{
        start_pos = text.find(r"\boxed{", search_start)
        if start_pos == -1:
            break
            
        # 从\boxed{之后开始匹配括号
        depth = 0
        content = text[start_pos + len(r"\boxed{"):]
        end_pos = -1
        
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                
            if depth == -1:  # 找到匹配的右括号
                end_pos = i
                break
        
        if end_pos != -1:
            boxed_contents.append(content[:end_pos].strip())
            # 继续从当前位置之后搜索
            search_start = start_pos + len(r"\boxed{") + end_pos + 1
        else:
            # 没有找到匹配的右括号，跳过这个\boxed{
            search_start = start_pos + len(r"\boxed{")
    
    return boxed_contents

def extract_and_validate(text):

   """
   优先尝试 JSON 提取，如果失败则 fallback 正则提取
   """
   # 1. 尝试 JSON 提取
   m = re.search(r"(\{[\s\S]*\})\s*$", text)
   if m:
      try:
         obj = json.loads(m.group(1))
         rw = obj["real_world_question"]
         sy = obj["synthetic_question"]
         conf = float(obj["confidence_score"])
         # 校验字段合法性
         if rw in ("A","B") and sy in ("A","B") and rw != sy and 0.0 <= conf <= 1.0:
               return {"real": rw, "synthetic": sy, "confidence": round(conf,4)}
      except:
         pass  # 如果 JSON 解析失败，进入 fallback

   # 2. fallback 正则
   # 尝试匹配 "Real-World Question: A" 等格式
   rw_match = re.search(r"real[-_\s]*world[-_\s]*question['\":\- ]+([AB])", text, re.IGNORECASE)
   sy_match = re.search(r"synthetic[-_\s]*question['\":\- ]+([AB])", text, re.IGNORECASE)
   conf_match = re.search(r"confidence[_\s\-:]*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
   
   rw = rw_match.group(1).upper() if rw_match else None
   sy = sy_match.group(1).upper() if sy_match else None
   conf = float(conf_match.group(1)) if conf_match else 0.0  # fallback 默认值

   # 保证 A/B 和不重复
   if rw not in ("A","B") or sy not in ("A","B") or rw==sy:
      rw, sy = None, None
   
   conf = max(0.0, min(1.0, conf))  # 确保置信度在 0~1

   return {"real": rw, "synthetic": sy, "confidence": round(conf,4),'response':text}

def custom_grade_answer(solution_str: str, gt) -> bool:
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution（gt 可为 str 或与 MATH 数据一致的 list）
    ground_truth = gt if isinstance(gt, list) else [gt]
    correct = False
    pred = extract_boxed_content(solution_str)
    for g in ground_truth:
        if pred is None:
            continue
        correct = grade_answer(str(pred), str(g))
        if correct:
            break

    return correct
 
def order_responses_for_skill(
    rsps: List[str], is_right: List[bool]
) -> Optional[Tuple[List[str], List[bool]]]:
    """
    为 skill 模板重排轨迹：至少需一条正确；全错返回 ``None``。
    同时含对与错时，优先将「第一个 [SUCCESS]」与「第一个 [FAIL]」置于列表前两位，
    并优先保证一对一错；最终仅保留约半数轨迹（``max(1, len(rsps)//2)``）。
    若混合对错且半数不足 2 条，则保留 2 条以满足一对一错。
    """
    n = len(rsps)
    if n == 0 or n != len(is_right):
        return None
    if not any(is_right):
        return None
    keep_n = max(1, n // 2)
    fail_i = next((i for i in range(n) if not is_right[i]), None)
    if fail_i is None:
        order = list(range(n))
    else:
        succ_i = next(i for i in range(n) if is_right[i])
        head = [succ_i, fail_i]
        rest = [i for i in range(n) if i not in (succ_i, fail_i)]
        order = head + rest
        keep_n = max(2, keep_n)
    order = order[:keep_n]
    return [rsps[i] for i in order], [is_right[i] for i in order]


def skill_traj_prompt_group_from_is_right(is_right: Any) -> str:
    """
    与 ``get_skill_prompt`` 中 ``[SUCCESS]`` / ``[FAIL]`` 前缀一致：由 raw 轨迹 ``is_right`` 判定。
    返回值供 ``extra_info`` 与 ``SynthsizerRewardManager`` 使用；与字符串分类 ``success_only`` /
    ``mixed_sf`` / ``unclassified`` 对齐。
    """
    if is_right is None:
        return "unclassified"
    if isinstance(is_right, np.ndarray):
        is_right = is_right.ravel().tolist()
    elif not isinstance(is_right, (list, tuple)):
        return "unclassified"
    if len(is_right) == 0:
        return "unclassified"
    rights = [bool(x) for x in is_right]
    if not any(rights):
        return "unclassified"
    if all(rights):
        return "success_only"
    return "mixed_sf"


# TODO: add more skill types
def get_skill_prompt(
    q: str, rsps: List[str], is_right: List[bool], skill_type: str
) -> str:
    """
    input:
        q: str, the question
        rsps: List[str], the responses
        is_right: List[bool], the is_right of the responses
        skill_type: str, the type of the skill
    return: str, the prompt for the skill synthesizer model

    全错时抛出 ``ValueError``（调用方应先过滤或勿调用）。
    """
    ordered = order_responses_for_skill(rsps, is_right)
    if ordered is None:
        raise ValueError("get_skill_prompt 需要至少一条正确轨迹（不可全为 FAIL）")
    rsps_o, right_o = ordered
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt", f"{skill_type}.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        skills_prompt_base = f.read()
    trajectories = []
    for rsp, ir in zip(rsps_o, right_o):
        if ir:
            trajectories.append(f"[SUCCESS] {rsp}")
        else:
            trajectories.append(f"[FAIL] {rsp}")
    skills_prompt_base = skills_prompt_base.format(
        question=q, trajectories="\n".join(trajectories)
    )
    return skills_prompt_base



def _ensure_prompt_field(record: Dict[str, Any], instructions: str) -> Dict[str, Any]:
    """
    保证每条样本含 `prompt`，且为 apply_chat_template 可用的 message 列表。
    若已有非空 list/tuple，原样保留；若为字符串则包成 system+user；否则从常见列推断。
    """
    assert isinstance(instructions, str), f"instructions must be a string, but got {instructions}"
    p = record.get("prompt")
    if p is not None:
        if isinstance(p, str) and p.strip():
            record["prompt"] = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": p.strip()},
            ]
            return record
        if isinstance(p, list) and len(p) > 0:
            return record
    if record.get("messages"):
        record["prompt"] = list(record["messages"])
        return record
    for key in ("question", "problem", "query", "content"):
        if key not in record or record[key] is None:
            continue
        text = str(record[key]).strip()
        if text:
            record["prompt"] = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": text},
            ]
            return record
    raise ValueError(
        "Each row must have a usable 'prompt' (non-empty str/list), or 'messages', "
        f"or one of question/problem/query/content. Got keys: {list(record.keys())}"
    )


def read_files(data_file: str, num_questions: Optional[int] = None):
    """num_questions 为 None 时表示使用文件中的全部行。"""
    if data_file.endswith(".jsonl"):
        df = pd.read_json(data_file, lines=True)
    elif data_file.endswith(".parquet"):
        df = pd.read_parquet(data_file)
    else:
        raise ValueError(f"Unsupported file type: {data_file}")

    n_file = len(df)
    if num_questions is None:
        num_questions = n_file
    assert num_questions > 0, (
        f"num_questions must be greater than 0, but got {num_questions}"
    )
    assert n_file >= num_questions, (
        "num_questions must be less than or equal to the number of questions in the file, "
        f"but got {num_questions} and {n_file}"
    )
    rows = df.to_dict(orient="records")[:num_questions]
    return [_ensure_prompt_field(dict(r), INSTRUCTIONS) for r in rows]


def read_records_from_list(
    rows: List[Dict[str, Any]], num_questions: Optional[int] = None
) -> List[Dict[str, Any]]:
    """与 read_files 相同的前处理，但输入为内存中的记录列表（供 HTTP server 免落盘）。
    num_questions 为 None 时表示使用列表中的全部行。"""
    n_rows = len(rows)
    if num_questions is None:
        num_questions = n_rows
    assert num_questions > 0, (
        f"num_questions must be greater than 0, but got {num_questions}"
    )
    assert n_rows >= num_questions, (
        "num_questions must be less than or equal to len(rows), "
        f"but got {num_questions} and {n_rows}"
    )
    slice_rows = rows[:num_questions]
    return [_ensure_prompt_field(dict(r), INSTRUCTIONS) for r in slice_rows]


@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    This wrapper applies a timeout to each individual `grade_answer` call.
    If the function's execution exceeds the specified timeout, it will return 'TIMED_OUT'.
    The timeout duration is passed as a keyword argument during the function call.
    """
    return grade_answer(res1, res2)

def process_single_R_Zero(idx, question, answer, response):
    '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
    
    results = [str(extract_boxed_content(out.text)) for out in response.outputs]

    answer_counts = {}
    for res in list(results):
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

    if not answer_counts:
        majority_ans, max_count = '', 0
    else:
        majority_ans = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_ans]

    if 'none' == majority_ans.lower():
        score = 0.0
    else:
        score = max_count / len(results) if results else 0.0
        #print(f'[process_single] Question {idx}: No valid labels found')
    uncertainty_reward = 1 - 2 * abs(score - 0.5)
    reward_info={
        'majority_accuracy': score,
        'all_labels': results,
        'answer_counts':answer_counts,
        'majority_accuracy': score,
        'majority_ans': majority_ans,
        'reward':uncertainty_reward,
    }
    if random.randint(0, 64) == 0:
        print(f'Question: {question} \n Answer: {answer} \n results:{results} \n answer_counts: {answer_counts} \n Answer: {majority_ans}\n Score: {score} \n Reward: {uncertainty_reward}')
    return {
        'idx':idx,
        'question': question,
        'answer': answer,
        'reward_info': reward_info,
        'reward':uncertainty_reward
    }
def process_single_q_eval(idx, question, response):
    """
    从模型响应中提取 JSON 分数。
    根据新的 prompt (q_eval_system.txt 和 q_eval_user.txt)，模型应该只输出 JSON 格式：{"score": <integer from 0 to 10>}
    """
    prompt = response.prompt
    assert question in prompt, f"question {question=} not in prompt {prompt=}"
    
    scores = []
    responses_str = []
    
    for out in response.outputs:
        text = out.text.strip()
        responses_str.append(text)
        
        # 尝试从文本中提取 JSON 分数
        score = None
        
        # 方法1: 直接解析整个响应文本为 JSON（根据新 prompt，整个响应应该就是 JSON）
        try:
            obj = json.loads(text)
            if 'score' in obj and isinstance(obj['score'], (int, float)):
                score = float(obj['score'])
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # 方法2: 如果直接解析失败，尝试清理文本后解析（移除可能的尾随空白、标点等）
        if score is None:
            # 移除可能的尾随标点符号和空白
            cleaned_text = text.rstrip('.,;!?\n\r\t ')
            try:
                obj = json.loads(cleaned_text)
                if 'score' in obj and isinstance(obj['score'], (int, float)):
                    score = float(obj['score'])
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # 方法3: 如果还是失败，尝试使用正则表达式提取（作为 fallback）
        if score is None:
            # 查找 JSON 格式的 score 字段，支持整数和浮点数
            # 使用 findall 找到所有匹配，取最后一个（通常最后一个可能是最终答案）
            json_matches = re.findall(r'\{[\s]*"score"[\s]*:[\s]*(\d+(?:\.\d+)?)[\s]*\}', text)
            if json_matches:
                try:
                    # 取最后一个匹配的分数
                    score = float(json_matches[-1])
                except (ValueError, IndexError):
                    pass
        
        # 验证分数范围 (0-10)
        if score is not None:
            score = max(0.0, min(10.0, score))
        else:
            # 如果无法提取分数，设置为 0.0（表示解析失败）
            score = 0.0
        
        scores.append(score)
    
    # 计算统计信息
    
    avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0.0
    reward = avg_score / 10.0
    return {
        'idx': idx,
        'question': question,
        'reward_info': {
            'scores': scores,
            'responses_str': responses_str,
            'avg_score': avg_score,
        },
        'reward': reward,  # 将分数归一化到 0-1 范围
    }


def calculate_bell_reward(score, sharpness=1.2):
    """
    使用幂次抛物线模拟高斯钟形曲线。
    天生保证 score=0 和 score=1 时奖励为 0。
    
    Args:
        score: float, [0, 1]
        sharpness: float, 控制曲线尖锐程度 (k值)。
                   k=1.0 : 圆顶 (最宽容)
                   k=5.0 : 标准高斯形态 (推荐)
                   k=10.0: 非常尖 (只奖励极其接近0.5的情况)
    """
    # 基础抛物线：4x(1-x)，在0.5处为1，在0和1处为0
    base = 4.0 * score * (1.0 - score)
    
    # 防止浮点误差导致负数（虽然理论上不会）
    base = max(0.0, base)
    
    # 通过指数控制胖瘦
    return base ** sharpness

    # --- 使用示例 ---
    # 推荐 sharpness = 5.0，效果非常接近 temperature=0.15 的高斯函数
    # reward = calculate_bell_reward(score, sharpness=5.0)

# 全局单例，避免重复创建
_similarity_filter = None

def get_similarity_filter():
    global _similarity_filter
    if _similarity_filter is None:
        _similarity_filter = MathSimilarityFilter()
    return _similarity_filter


def process_single_R_Zero_ref_q_round(idx, question, answer, response, reference_question):
    '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
    
    results = [str(extract_boxed_content(out.text)) for out in response.outputs]

    answer_counts = {}
    for res in list(results):
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

    if not answer_counts:
        majority_ans, max_count = '', 0
    else:
        majority_ans = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_ans]

    if 'none' == majority_ans.lower():
        score = 0.0
    else:
        score = max_count / len(results) if results else 0.0
    filter = get_similarity_filter()
    is_bad, message, filter_score = filter.is_bad_case(reference_question, question)
        #print(f'[process_single] Question {idx}: No valid labels found')
    
    #uncertainty_reward = 1 - 2 * abs(score - 0.5)
    uncertainty_reward = calculate_bell_reward(score)
    #reward = 0.0 if is_bad else uncertainty_reward
    if is_bad:
    # filter_score 越高，惩罚越重
        novelty_factor = max(0.0, 1.0 - filter_score)
        reward = uncertainty_reward * novelty_factor
    else:
        reward = uncertainty_reward
    #reward = uncertainty_reward if not is_bad else max(0.0, uncertainty_reward - filter_score)
    reward_info={
        'majority_accuracy': score,
        'all_labels': results,
        'answer_counts':answer_counts,
        'majority_accuracy': score,
        'majority_ans': majority_ans,
        'filter_info':{
            'is_bad': is_bad,
            'message': message,
            'filter_score': filter_score,
        },
        'reward':reward,
    }
    if random.randint(0, 64) == 0:
        print('='*100)
        print(f'{reward_info=}')
        #print('='*100)
        print('='*100)
    return {
        'idx':idx,
        'question': question,
        'answer': answer,
        'reward_info': reward_info,
        'reward':reward
    }

def process_single_R_Zero_ref_q(idx, question, answer, response, reference_question):
    '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
    
    results = [str(extract_boxed_content(out.text)) for out in response.outputs]

    answer_counts = {}
    for res in list(results):
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

    if not answer_counts:
        majority_ans, max_count = '', 0
    else:
        majority_ans = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_ans]

    if 'none' == majority_ans.lower():
        score = 0.0
    else:
        score = max_count / len(results) if results else 0.0
    filter = get_similarity_filter()
    is_bad, message, filter_score = filter.is_bad_case(reference_question, question)
        #print(f'[process_single] Question {idx}: No valid labels found')
    
    uncertainty_reward = 1 - 2 * abs(score - 0.5)
    #reward = 0.0 if is_bad else uncertainty_reward
    if is_bad:
    # filter_score 越高，惩罚越重
        novelty_factor = max(0.0, 1.0 - filter_score)
        reward = uncertainty_reward * novelty_factor
    else:
        reward = uncertainty_reward
    #reward = uncertainty_reward if not is_bad else max(0.0, uncertainty_reward - filter_score)
    reward_info={
        'majority_accuracy': score,
        'all_labels': results,
        'answer_counts':answer_counts,
        'majority_accuracy': score,
        'majority_ans': majority_ans,
        'filter_info':{
            'is_bad': is_bad,
            'message': message,
            'filter_score': filter_score,
        },
        'reward':reward,
    }
    if random.randint(0, 64) == 0:
        print('='*100)
        print(f'{reward_info=}')
        #print('='*100)
        print('='*100)
    return {
        'idx':idx,
        'question': question,
        'answer': answer,
        'reward_info': reward_info,
        'reward':reward
    }


def process_single_Rule(idx, question, answer, response):
    '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
    
    prompt=response.prompt
    assert question in prompt, f"question {question=} not in prompt {prompt=}"
    repetition_rates=[]
    repetion_penalty=[]
    responses_str=[]
    boxed_format=[]
    rsp_lengths=[]
    rewards = []
    answers_penalty=[]
    format_info=[]
    no_answer_penalties=[]
    for out in response.outputs:
        
        responses_str.append(out.text)
        format_result = format_check(out.text)
        rsp_length = len(out.token_ids)
        rep_penalty = 0.1 if format_result['repetition_rate'] >= 0.1 else 1.0
        boxed_penalty = 1.0 if format_result['boxed_count'] > 0 and format_result['boxed_count'] < 3 else 0.5
        answer_penalty = 0.5 if not format_result['final_answer'] else 1.0
        rsp_reward = max(0, rsp_length - 1024) / 1024
        #no_answer_penalty = format_result['no_answer_penalty']
        reward = rsp_reward * rep_penalty * boxed_penalty * answer_penalty 


        rsp_lengths.append(rsp_length)
        repetion_penalty.append(rep_penalty)
        boxed_format.append(boxed_penalty)
        answers_penalty.append(answer_penalty)
        format_info.append(format_result)
        rewards.append(reward)
        repetition_rates.append(format_result['repetition_rate'])
        #no_answer_penalties.append(no_answer_penalty)


    assert len(rsp_lengths) == len(repetition_rates) == len(repetion_penalty) == len(boxed_format) == len(answers_penalty) == len(rewards), "length mismatch"
    # if sum(no_answer_penalties) <= len(no_answer_penalties) // 2:
    #     reward = 0.0
    # else:
    reward = sum(rewards) / len(rewards)

    return {
        'idx':idx,
        'question': question,
        'answer': answer,
        'reward_info':{
            'response_str':responses_str,
            'repetition_rates':repetition_rates,
            'repetion_penalty':repetion_penalty,
            'boxed_format':boxed_format,
            'answers_penalty':answers_penalty,
            'format_info':format_info,
            'rsp_lengths':rsp_lengths,
            #'no_answer_penalties':no_answer_penalties,
            'rewards':rewards,
        },
        'reward':reward,
    }


def detect_real_vs_synthetic(qA, qB):
   """
   # LLM 判断正确，reward = max(0, 1 - conf)
   # LLM 判断错误，reward = max(0, min(1, conf))
   调用 LLM 判断 Question A/B 谁是真实，谁是合成
   :param qA: Question A 文本  虚假问题
   :param qB: Question B 文本  真实问题
   :param correct_real: 可选，实际真实题，用于计算 reward
   :return: dict {"real": "A/B", "synthetic": "A/B", "confidence": float, "reward": float or None}

   """
   reward = 0.0
   output_text = process_example(qA, qB)
   if output_text:
      result = extract_and_validate(output_text)
      realQ=result['real'].lower()
      syQ=result['synthetic'].lower()
      conf=float(result['confidence'])
      if 'a' in realQ and 'b' in syQ: # 判断错误
         reward=max(0, min(1, conf))
         #reward=max(0, 1-conf)
      elif 'b' in realQ and 'a' in syQ: 
         #reward=max(0, min(1, conf))
         reward=max(0, 1-conf)
   result["reward"] = reward
   return result


def format_check(response_str: str) -> Dict[str, Any]:
    """
    检查回答的格式，包括：
    1. 是否包含\boxed{}格式的答案
    2. 提取所有\boxed{}中的内容
    3. 计算重复率
    
    返回：
        Dict包含以下字段：
        - has_boxed: 是否包含\boxed
        - boxed_count: \boxed的数量
        - all_boxed_answers: 所有\boxed中的内容列表
        - final_answer: 最后一个\boxed的内容（通常是最终答案）
        - repetition_rate: 重复率
        - repetition_spans: 重复片段
    """
    
    # 提取所有\boxed内容
    all_boxed = extract_all_boxed_content(response_str)
    
    # 计算重复率
    repetition_rate, rep_spans = calc_repetition_rate(response_str)
    #boxed_cnt = len(all_boxed)
    # if (boxed_cnt > 0 and boxed_cnt < 3):
    #     format_score=1.0
    # if repetition_rate > 0.1:
    #     format_score=0.1

    # no_answer_penalty=1.0
    # for answer in all_boxed:
    #     if 'no answer' in answer.lower():
    #         no_answer_penalty=0.0
    #         break

    # 构建返回结果
    result = {
        "has_boxed": len(all_boxed) > 0,
        "boxed_count": len(all_boxed),
        "all_boxed_answers": all_boxed,
        "final_answer": all_boxed[-1] if all_boxed else None,
        "repetition_rate": repetition_rate,
        "repetition_spans": rep_spans,
        #"no_answer_penalty": no_answer_penalty,
    }
    
    return result


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def extract_boxed_content(text: str) -> str:
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



def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())


def _kmp_prefix_function_fast(s: str) -> np.ndarray:
    """KMP前缀函数 (NumPy实现，适合≤8192长度)"""
    arr = np.frombuffer(s.encode('utf-8', 'ignore'), dtype=np.uint8)
    n = len(arr)
    pi = np.zeros(n, dtype=np.int32)
    j = 0
    for i in range(1, n):
        while j > 0 and arr[i] != arr[j]:
            j = pi[j - 1]
        if arr[i] == arr[j]:
            j += 1
        pi[i] = j
    return pi


def _find_tandem_repeats_fast(text: str, min_repeat_len: int = 4) -> List[Tuple[int, int]]:
    """检测所有连续重复片段（基于KMP前缀函数）"""
    n = len(text)
    pi = _kmp_prefix_function_fast(text)
    spans = []
    for i in range(n):
        L = i + 1
        p = L - pi[i]
        if pi[i] > 0 and p >= min_repeat_len and L % p == 0:
            k = L // p
            if k >= 2:
                spans.append((L - p * k, L))
    # 合并相邻区间
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def calc_repetition_rate(text: str, min_repeat_len: int = 2) -> Tuple[float, List[Tuple[int, int, str]]]:
    """
    极速版：适用于文本长度 <= 8192。
    全局检测 + 合并重复片段 + 最长块占比。
    """
    text = _normalize_text(text)
    n = len(text)
    if n < 2 * min_repeat_len:
        return 0.0, []

    spans = _find_tandem_repeats_fast(text, min_repeat_len)
    if not spans:
        return 0.0, []

    # 合并并提取样本
    merged = []
    s, e = spans[0]
    for s2, e2 in spans[1:]:
        if s2 <= e:
            e = max(e, e2)
        else:
            merged.append((s, e))
            s, e = s2, e2
    merged.append((s, e))

    total_cov = sum(e - s for s, e in merged)
    max_block = max(e - s for s, e in merged)
    rate = 1 - (1 - total_cov / n) * (1 - max_block / n)

    samples = []
    for s, e in sorted(merged, key=lambda x: x[1] - x[0], reverse=True)[:5]:
        snippet = text[s:e]
        if len(snippet) > 200:
            snippet = snippet[:100] + " … " + snippet[-100:]
        samples.append((int(s), int(e), str(snippet)))

    return float(round(rate, 6)), samples
