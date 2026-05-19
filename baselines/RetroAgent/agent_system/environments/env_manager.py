from __future__ import annotations

import copy
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.env_package.math_single_turn import build_math_single_turn_envs
from agent_system.environments.math_utils import (
    build_lesson_text,
    compute_subtask_potential,
    normalize_reflection_payload,
    parse_reflection_json,
)
from agent_system.environments.prompts import MATH_REFLECT_TEMPLATE, MATH_TEMPLATE, MATH_TEMPLATE_NO_HIS
from agent_system.memory import ReflectionMemory, SimpleMemory


def math_projection(text_actions: List[str]) -> Tuple[List[str], List[bool]]:
    actions = [str(action or "") for action in text_actions]
    valids = [bool(action.strip()) for action in actions]
    return actions, valids


def _normalize_kwargs_list(kwargs: Any) -> List[Dict[str, Any]]:
    if kwargs is None:
        return []
    if isinstance(kwargs, np.ndarray):
        kwargs = kwargs.tolist()
    if isinstance(kwargs, list):
        return [dict(item) for item in kwargs if isinstance(item, dict)]
    if isinstance(kwargs, dict):
        if "question" in kwargs:
            return [dict(kwargs)]
        keys = [k for k in kwargs.keys() if k != "is_train"]
        if not keys:
            return []
        probe = kwargs[keys[0]]
        if isinstance(probe, np.ndarray):
            probe = probe.tolist()
        if isinstance(probe, list):
            rows: List[Dict[str, Any]] = []
            for idx in range(len(probe)):
                row = {}
                for key in keys:
                    value = kwargs[key]
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    row[key] = value[idx]
                if "is_train" in kwargs:
                    row["is_train"] = kwargs["is_train"]
                rows.append(row)
            return rows
        return [dict(kwargs)]
    raise TypeError(f"Unsupported kwargs type: {type(kwargs)!r}")


class MathEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, retrieve_type: str):
        self.memory = SimpleMemory()
        self.group_n = config.env.rollout.n

        mem_config = config.env.get("reflection_memory", {})
        filepath = mem_config.get("filepath", "math_reflections.json")
        alpha = mem_config.get("alpha", 0.7)
        beta = mem_config.get("beta", 0.05)
        temp = mem_config.get("temperature", 0.5)
        ucb_scale = mem_config.get("ucb_scale", 1.0)
        top_k = mem_config.get("top_k", 1)
        model_name = mem_config.get("model_name", "all-MiniLM-L6-v2")

        self.top_k = top_k
        self.memory_start_cutoff = mem_config.get("memory_start_cutoff", 0.0)
        self.current_progress_ratio = 0.0
        self.retrieve_mode = mem_config.get("retrieve_mode", "both")
        self.enable_memory = mem_config.get("enable_memory", True)
        self.group_outperformance = mem_config.get("group_outperformance", False)
        self.full_group_memory = mem_config.get("full_group_memory", False)
        self.group_relative_intrinsic_rewards = mem_config.get("group_relative_intrinsic_rewards", False)
        self.potential_based_on_binary_success = mem_config.get("potential_based_on_binary_success", False)

        self.reflection_memory = ReflectionMemory(
            filepath=filepath,
            model_name=model_name,
            alpha=alpha,
            beta=beta,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale,
        )

        self.task_trajectory_history: Dict[str, Dict[str, List[str]]] = {}
        self.task_potential_history: Dict[str, float] = {}
        self.batch_previous_potentials: List[float] = []
        self.current_reflections: List[str] = []
        self.retrieved_raw_reflections: List[List[Dict[str, Any]]] = []
        self.current_retrieval_types: List[str] = []
        self.batch_retrieved_types: List[List[str]] = []
        self.last_trajectories: List[str] = []
        self.questions: List[str] = []
        self.task_keys: List[str] = []

        super().__init__(envs, projection_f, config)

    def update_training_progress(self, current_step: int, total_steps: int):
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps

    def reset(self, kwargs) -> Dict[str, Any]:
        rows = _normalize_kwargs_list(kwargs)
        is_eval = not rows[0].get("is_train", True) if rows else False
        obs, infos = self.envs.reset(kwargs=rows)

        self.questions = [info.get("question", "") for info in infos]
        self.task_keys = [info.get("task_key", q) for info, q in zip(infos, self.questions)]
        self.pre_text_obs = list(obs)
        self.memory.reset(batch_size=len(obs))

        self.batch_size = len(obs)
        if self.batch_size == 0:
            return {"text": [], "image": None, "anchor": []}, infos
        if self.batch_size % self.group_n != 0:
            raise ValueError("Batch size must be divisible by group size")

        self.current_reflections = []
        self.retrieved_raw_reflections = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = []
        self.batch_retrieved_types = []

        in_warmup_period = (not is_eval) and (self.current_progress_ratio < self.memory_start_cutoff)
        group_split_index = 0 if self.full_group_memory else self.group_n // 2

        for i, task_key in enumerate(self.task_keys):
            prev_potential = self.task_potential_history.get(task_key, 0.0)
            self.batch_previous_potentials.append(prev_potential)

            formatted_reflections = ""
            raw_list_of_dicts: List[Dict[str, Any]] = []
            current_types_list: List[str] = []
            retrieval_type_str = "control"
            should_retrieve = False

            if self.enable_memory:
                if in_warmup_period:
                    should_retrieve = False
                elif is_eval:
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    should_retrieve = position_in_group >= group_split_index
                    retrieval_type_str = "experiment" if should_retrieve else "control"

            if should_retrieve:
                k = self.top_k if is_eval else 1
                raw_list_of_dicts = self.reflection_memory.retrieve(
                    current_task_description=task_key,
                    top_k=k,
                    filter_type=self.retrieve_mode,
                )
                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        text = item.get("text", "")
                        r_type = item.get("type", "unknown")
                        current_types_list.append(r_type)
                        formatted_lines.append(text)
                    formatted_reflections = "Past reflections on similar tasks:\n" + "\n".join(formatted_lines)
                    formatted_reflections += "\nUse them only if they still fit the current problem."

            self.current_reflections.append(formatted_reflections)
            self.retrieved_raw_reflections.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            infos[i]["reflection_types"] = current_types_list
            infos[i]["retrieval_group"] = retrieval_type_str

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": list(obs),
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store(
            {
                "text_obs": self.pre_text_obs,
                "action": actions,
                "reward": rewards,
                "dones": dones,
                "won": [info.get("won", False) for info in infos],
            }
        )
        self.pre_text_obs = list(next_obs)

        next_observations = {
            "text": self.build_text_obs(next_obs, init=False),
            "image": None,
            "anchor": list(next_obs),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        postprocess_text_obs: List[str] = []
        memory_contexts = None
        if not init and self.config.env.history_length > 0:
            memory_contexts, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
            )

        for i, question in enumerate(text_obs):
            reflections = self.current_reflections[i] if i < len(self.current_reflections) else ""
            if init or self.config.env.history_length <= 0 or memory_contexts is None:
                obs = MATH_TEMPLATE_NO_HIS.format(reflections=reflections, question=question)
            else:
                obs = MATH_TEMPLATE.format(
                    reflections=reflections,
                    question=question,
                    step_count=len(self.memory[i]),
                    memory_context=memory_contexts[i],
                )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def build_reflect_text_obs(self, infos: List[Dict[str, Any]]) -> List[str]:
        postprocess_text_obs: List[str] = []
        memory_contexts, _ = self.memory.fetch(15, obs_key="text_obs", action_key="action")

        for i, info in enumerate(infos):
            task_key = self.task_keys[i]
            question = self.questions[i]
            if task_key not in self.task_trajectory_history:
                self.task_trajectory_history[task_key] = {"successful": [], "failed": []}

            if info.get("won", False):
                self.task_trajectory_history[task_key]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task_key]["failed"].append(memory_contexts[i])

        self.last_trajectories = memory_contexts

        for i, info in enumerate(infos):
            task_key = self.task_keys[i]
            is_won = bool(info.get("won", False))
            if is_won:
                ref_hist = self.task_trajectory_history[task_key]["failed"]
                if ref_hist:
                    reference = "Reference Failed Trajectory (for comparison):\n" + ref_hist[-1]
                else:
                    reference = "No failed attempts available for comparison."
            else:
                ref_hist = self.task_trajectory_history[task_key]["successful"]
                if ref_hist:
                    reference = "Reference Successful Trajectory (for comparison):\n" + ref_hist[-1]
                else:
                    reference = "No successful attempts available for reference."

            obs = MATH_REFLECT_TEMPLATE.format(
                question=self.questions[i],
                reference_trajectory=reference,
                current_trajectory=memory_contexts[i],
            )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def reflect(self, infos: List[Dict[str, Any]]):
        reflect_obs_text = self.build_reflect_text_obs(infos)
        observations = {"text": reflect_obs_text, "image": None, "anchor": reflect_obs_text}

        for info in infos:
            info["is_action_valid"] = to_numpy(True)

        batch_size = len(self.task_keys)
        if batch_size % self.group_n != 0:
            raise ValueError("Batch size must be divisible by group size")

        num_groups = batch_size // self.group_n
        group_split_index = 0 if self.full_group_memory else self.group_n // 2

        for group_idx in range(num_groups):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index

            control_wins = sum(bool(infos[i].get("won", False)) for i in range(start_idx, min(mid_idx, end_idx)))
            experiment_wins = sum(bool(infos[i].get("won", False)) for i in range(mid_idx, end_idx))
            group_outperformed = experiment_wins > control_wins

            for i in range(mid_idx, end_idx):
                task_key = self.task_keys[i]
                is_success = bool(infos[i].get("won", False))
                utility_score = 1.0 if is_success else 0.0
                if self.group_outperformance and not (is_success and group_outperformed):
                    utility_score = 0.0

                for item in self.retrieved_raw_reflections[i]:
                    reflection_text = item.get("text", "")
                    if reflection_text:
                        self.reflection_memory.update_utility(
                            task_description=task_key,
                            reflection_text=reflection_text,
                            score=utility_score,
                        )

        return observations, infos

    def step_reflect(self, text_actions: List[str], infos: List[Dict[str, Any]]):
        reflect_rewards: List[float] = []
        current_scores = np.zeros(self.batch_size, dtype=np.float32)
        raw_improvements = np.zeros(self.batch_size, dtype=np.float32)
        is_won_array = np.zeros(self.batch_size, dtype=bool)

        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        for i, reflection_text in enumerate(text_actions):
            task_key = self.task_keys[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""
            prev_phi = self.batch_previous_potentials[i]
            actual_success = bool(infos[i].get("won", False))
            is_won_array[i] = actual_success
            current_phi = 0.0

            try:
                payload = normalize_reflection_payload(parse_reflection_json(reflection_text))
                subtask_phi = compute_subtask_potential(payload)
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 1.0 if actual_success else subtask_phi

                predicted_success = bool(payload.get("task_success", False))
                reflect_rewards.append(10.0 if predicted_success == actual_success else 0.0)

                if predicted_success == actual_success:
                    lesson_text = build_lesson_text(payload)
                    if lesson_text:
                        self.reflection_memory.add(
                            task_description=task_key,
                            reflection_text=lesson_text,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio,
                        )
            except Exception:
                reflect_rewards.append(0.0)
                current_phi = 1.0 if (self.potential_based_on_binary_success and actual_success) else 0.0

            current_scores[i] = current_phi
            raw_improvements[i] = max(0.0, current_phi - prev_phi)

        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size, dtype=np.float32)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            task_key = self.task_keys[start_idx]
            group_improvements = raw_improvements[start_idx:end_idx]
            group_scores = current_scores[start_idx:end_idx]

            if self.group_relative_intrinsic_rewards:
                group_mean_imp = float(np.mean(group_improvements)) if len(group_improvements) > 0 else 0.0
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements - group_mean_imp
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            current_group_mean_score = float(np.mean(group_scores)) if len(group_scores) > 0 else 0.0
            old_baseline = self.task_potential_history.get(task_key, 0.0)
            if current_group_mean_score > old_baseline:
                self.task_potential_history[task_key] = current_group_mean_score

        infos = [info.copy() for info in infos]
        for info in infos:
            info["is_action_valid"] = to_numpy(True)

        return (
            None,
            to_numpy(reflect_rewards),
            to_numpy(final_intrinsic_rewards),
            None,
            copy.deepcopy(infos),
            to_numpy(current_scores),
        )

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        reflect_rewards = kwargs.get("reflect_rewards", None)

        batch_size = len(total_batch_list)
        success = defaultdict(list)

        for bs in range(batch_size):
            r_reward = None
            if reflect_rewards is not None:
                try:
                    r_reward = reflect_rewards[bs]
                except IndexError:
                    r_reward = 0.0
            self._process_batch(bs, total_batch_list, total_infos, success, reflect_reward=r_reward)

        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success, reflect_reward=None):
        if reflect_reward is not None:
            val = float(reflect_reward.item()) if hasattr(reflect_reward, "item") else float(reflect_reward)
            success["reflect_success_rate"].append(val)
        else:
            success["reflect_success_rate"].append(0.0)

        found_active_step = False
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if not batch_item["active_masks"]:
                continue
            info = total_infos[batch_idx][i]
            won_value = float(info.get("won", 0.0))
            score_value = float(info.get("task_score", 0.0))
            success["success_rate"].append(won_value)
            success["task_score"].append(score_value)
            data_source = info.get("data_source", "math")
            success[f"{data_source}_success_rate"].append(won_value)
            found_active_step = True
            break

        if not found_active_step:
            success["success_rate"].append(0.0)
            success["task_score"].append(0.0)


def make_envs(config):
    from omegaconf import OmegaConf

    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")

    env_name = str(config.env.env_name)
    if env_name.lower() != "mathsingleturn":
        raise ValueError(f"Unsupported environment for baselines/RetroAgent: {env_name}")

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    _ = resources_per_worker  # reserved for future env backends

    projection_f = partial(math_projection)
    envs = build_math_single_turn_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        env_config=config.env,
    )
    val_envs = build_math_single_turn_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        is_train=False,
        env_config=config.env,
    )
    train_manager = MathEnvironmentManager(envs, projection_f, config, config.env.train_retrieve_type)
    val_manager = MathEnvironmentManager(val_envs, projection_f, config, config.env.eval_retrieve_type)
    return train_manager, val_manager
