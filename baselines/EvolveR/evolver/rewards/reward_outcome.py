from dataclasses import dataclass, field
import os
import random
import re
import requests
from typing import Any, Dict, List, Optional, Union

from verl.utils.reward_score.qa_em import em_check
from verl.utils.reward_score.qa_em_format import compute_score_em as compute_score_em_format
from verl.utils.reward_score.qa_f1 import f1_score_o2 as compute_f1_score
from verl.utils.reward_score.math import is_equiv, strip_string, last_boxed_only_string, remove_boxed

from evolver.rewards.config import *


def _use_math_outcome() -> bool:
    """Default True: math scoring for baselines/EvolveR. Set EVOLVER_QA_OUTCOME=1 for legacy QA path."""
    return os.environ.get("EVOLVER_QA_OUTCOME", "0") != "1"


def normalize_ground_truth(ground_truth: Union[str, List[str], Dict[str, Any]]) -> Dict[str, Any]:
    """SkillRL/DeepMath uses a bare string; original EvolveR uses {target: ...}."""
    if ground_truth is None:
        return {"target": ""}
    if isinstance(ground_truth, dict):
        if "target" in ground_truth:
            return ground_truth
        t = ground_truth.get("ground_truth", ground_truth.get("answer", ""))
        return {"target": t if isinstance(t, str) else str(t)}
    if isinstance(ground_truth, (list, tuple)) and len(ground_truth) > 0:
        return {"target": str(ground_truth[0])}
    return {"target": str(ground_truth)}


def extract_solution(solution_str: str) -> Optional[str]:
    """Prefer last <answer>...</answer>; if missing, try last \\boxed{...} (Hendrycks-style)."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if matches:
        return matches[-1].group(1).strip()
    box = last_boxed_only_string(solution_str)
    if box is not None:
        try:
            return remove_boxed(box).strip()
        except (AssertionError, Exception):
            return None
    return None


def compute_score_em(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """Legacy QA: EM in <answer> vs target."""
    g = normalize_ground_truth(ground_truth)
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {g['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str[:500]}...")

    if answer is None:
        return 0
    if em_check(answer, g["target"]):
        return score
    return format_score


def compute_score_math(solution_str: str, ground_truth) -> float:
    """Mathematical equivalence (strip + is_equiv) on answer span or full string for boxed."""
    g = normalize_ground_truth(ground_truth)
    target = g["target"]
    if not target and target != 0:
        return 0.0
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    try:
        s1, s2 = strip_string(str(answer)), strip_string(str(target))
        if s1 and s2 and is_equiv(s1, s2, verbose=False):
            return 1.0
    except Exception:
        pass
    if em_check(answer, target):
        return 1.0
    return 0.0


@dataclass
class OutcomeRewardOutput:
    reward: float
    metrics: dict = field(default_factory=dict)


def dv_reward_fn(queries: List[str], api_url: str = diversity_api_url, do_print=False) -> float:
    payload = {"queries": queries}
    try:
        output = requests.post(api_url, json=payload).json()
        if do_print:
            print(output)
        reward = output["overall_independence_score"]
        return reward
    except Exception as e:
        print(f"[WARNING] Independence score error! {str(e)}")
        return 0.0


def f1_reward_fn(
    solution_str: str, ground_truth, api_url: str = f1_api_url, do_print=False
) -> float:
    payload = {
        "generated_text": solution_str,
        "reference_points": ground_truth,
        "threshold": 0.75,
    }
    try:
        output = requests.post(api_url, json=payload).json()
        if do_print:
            print(output)
        reward = output["f1"]
        return reward
    except Exception as e:
        print(f"[WARNING] F1 score error! {str(e)}")
        return 0.0


def outcome_reward_fn(
    queries: List[str],
    solution_str: str,
    ground_truth: Union[str, List[str], Dict],
    question_type: str = "closed",
    config: Optional[Dict] = None,
) -> OutcomeRewardOutput:
    cfg = config or {}
    f1_api_url = cfg.get("f1_api_url")
    diversity_api_url = cfg.get("diversity_api_url")
    weights = cfg.get("weights", outcome_weights) if config else outcome_weights
    use_math = cfg.get("use_math", _use_math_outcome())
    g = normalize_ground_truth(ground_truth)

    metrics: Dict[str, Any] = {}

    answer = extract_solution(solution_str=solution_str)
    if answer is not None and question_type == "closed" and not use_math:
        try:
            f1, precision, recall = compute_f1_score(answer, g["target"])
            metrics["f1_score"] = f1
            metrics["precision"] = precision
            metrics["recall"] = recall
        except Exception:
            pass

    if question_type == "closed":
        if use_math:
            em_score = compute_score_math(solution_str, g)
            metrics["em_score"] = em_score
            metrics["em_score_format"] = 0.0
            total_reward = em_score
        else:
            em_score_format = compute_score_em_format(solution_str, g)
            em_score = compute_score_em(solution_str, g)
            metrics["em_score"] = em_score
            metrics["em_score_format"] = em_score_format
            total_reward = em_score
    else:
        if f1_api_url and weights.get("f1", 0) > 0:
            f1_score = f1_reward_fn(solution_str, ground_truth, f1_api_url)
            metrics["f1_score"] = f1_score
        else:
            f1_score = 0.0

        if diversity_api_url and weights.get("diversity", 0) > 0:
            diversity_score = dv_reward_fn(queries, diversity_api_url)
            metrics["diversity_score"] = diversity_score
        else:
            diversity_score = 0.0

        total_reward = (
            weights.get("f1", 0) * f1_score + weights.get("diversity", 0) * diversity_score
        )

    return OutcomeRewardOutput(reward=total_reward, metrics=metrics)
