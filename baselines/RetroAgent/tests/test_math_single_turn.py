from __future__ import annotations

from pathlib import Path
import sys


BASELINE_DIR = Path(__file__).resolve().parents[1]
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from agent_system.environments.env_package.math_single_turn.envs import MathSingleTurnEnvBatch


def _task(question: str, gt: str):
    return {
        "question": question,
        "ground_truth": gt,
        "data_source": "DeepMath-103K",
        "topic": "algebra",
        "index": 0,
    }


def test_math_env_correct_answer():
    env = MathSingleTurnEnvBatch()
    env.reset([_task("Compute 1+1", "2")])
    _, rewards, dones, infos = env.step(["<think>easy</think><answer>2</answer>"])
    assert rewards == [1.0]
    assert dones == [True]
    assert infos[0]["won"] is True
    assert infos[0]["format_ok"] is True


def test_math_env_equivalent_answer():
    env = MathSingleTurnEnvBatch()
    env.reset([_task("Compute 1/2 as decimal or fraction", "1/2")])
    _, rewards, _, infos = env.step(["<answer>0.5</answer>"])
    assert rewards == [1.0]
    assert infos[0]["pred_answer"] == "0.5"


def test_math_env_missing_answer_tag():
    env = MathSingleTurnEnvBatch()
    env.reset([_task("Compute 1+1", "2")])
    _, rewards, _, infos = env.step(["2"])
    assert rewards == [0.0]
    assert infos[0]["won"] is False
    assert infos[0]["format_ok"] is False

