from __future__ import annotations

from pathlib import Path
import sys

from omegaconf import OmegaConf


BASELINE_DIR = Path(__file__).resolve().parents[1]
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from agent_system.environments.env_manager import MathEnvironmentManager, math_projection


class FakeReflectionMemory:
    def __init__(self, *args, **kwargs):
        self.add_calls = []
        self.update_calls = []
        self.retrieve_calls = []

    def retrieve(self, current_task_description, top_k, filter_type):
        self.retrieve_calls.append((current_task_description, top_k, filter_type))
        return [{"text": "Past lesson", "type": "success"}]

    def update_utility(self, task_description, reflection_text, score):
        self.update_calls.append((task_description, reflection_text, score))

    def add(self, **kwargs):
        self.add_calls.append(kwargs)


class FakeEnv:
    def reset(self, kwargs=None):
        obs = [row["question"] for row in kwargs]
        infos = []
        for row in kwargs:
            infos.append(
                {
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "data_source": row["data_source"],
                    "topic": row["topic"],
                    "index": row["index"],
                    "task_key": f"[{row['topic']}] {row['question']}",
                    "won": False,
                    "task_score": 0.0,
                    "format_ok": False,
                    "pred_answer": "",
                }
            )
        return obs, infos


def _config(group_n: int = 2):
    return OmegaConf.create(
        {
            "env": {
                "rollout": {"n": group_n},
                "history_length": 1,
                "reflection_memory": {
                    "top_k": 1,
                    "alpha": 0.6,
                    "beta": 0.05,
                    "temperature": 0.1,
                    "ucb_scale": 1.0,
                    "enable_memory": True,
                    "retrieve_mode": "both",
                    "memory_start_cutoff": 0.0,
                    "group_outperformance": False,
                    "full_group_memory": False,
                    "group_relative_intrinsic_rewards": False,
                    "potential_based_on_binary_success": False,
                    "model_name": "fake",
                },
            }
        }
    )


def test_reflect_updates_memory_utility(monkeypatch):
    import agent_system.environments.env_manager as env_manager_module

    monkeypatch.setattr(env_manager_module, "ReflectionMemory", FakeReflectionMemory)
    manager = MathEnvironmentManager(FakeEnv(), math_projection, _config(group_n=2), "ucb")

    manager.reset(
        [
            {"question": "q1", "ground_truth": "1", "data_source": "DeepMath-103K", "topic": "alg", "index": 0},
            {"question": "q1", "ground_truth": "1", "data_source": "DeepMath-103K", "topic": "alg", "index": 1},
        ]
    )
    infos = [
        {"won": False, "data_source": "DeepMath-103K"},
        {"won": True, "data_source": "DeepMath-103K"},
    ]
    manager.reflect(infos)

    assert manager.reflection_memory.update_calls
    task_description, reflection_text, score = manager.reflection_memory.update_calls[0]
    assert task_description == "[alg] q1"
    assert reflection_text == "Past lesson"
    assert score == 1.0


def test_step_reflect_saves_lessons(monkeypatch):
    import agent_system.environments.env_manager as env_manager_module

    monkeypatch.setattr(env_manager_module, "ReflectionMemory", FakeReflectionMemory)
    manager = MathEnvironmentManager(FakeEnv(), math_projection, _config(group_n=1), "ucb")

    manager.reset(
        [
            {"question": "q2", "ground_truth": "2", "data_source": "DeepMath-103K", "topic": "alg", "index": 0},
        ]
    )
    manager.last_trajectories = ["[Observation 1: 'q2', Action 1: '<answer>2</answer>']"]
    infos = [{"won": True, "task_score": 1.0, "data_source": "DeepMath-103K"}]
    reflection = """
    {
      "subtasks": [
        {"name": "understand_problem", "description": "read it", "status": "completed"},
        {"name": "choose_strategy", "description": "picked arithmetic", "status": "completed"},
        {"name": "derive_solution", "description": "computed", "status": "completed"},
        {"name": "check_constraints", "description": "checked", "status": "completed"},
        {"name": "finalize_answer", "description": "answered", "status": "completed"}
      ],
      "task_success": true,
      "action_lesson": "Answer directly after solving.",
      "reasoning_lesson": "Verify the arithmetic once."
    }
    """
    _, reflect_rewards, intrinsic_rewards, _, _, current_scores = manager.step_reflect([reflection], infos)

    assert reflect_rewards.tolist() == [10.0]
    assert intrinsic_rewards.tolist() == [1.0]
    assert current_scores.tolist() == [1.0]
    assert manager.reflection_memory.add_calls
