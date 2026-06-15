"""DeepMath environment adapter for the original SkillOpt training loop."""
from __future__ import annotations

import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.deepmath.dataloader import DeepMathDataLoader
from skillopt.envs.deepmath.rollout import run_batch
from skillopt.gradient.reflect import run_minibatch_reflect


class DeepMathAdapter(EnvAdapter):
    """Adapter that makes SkillOpt run as a true DeepMath skill optimizer."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "8:1:1",
        split_seed: int = 42,
        split_output_dir: str = "",
        max_completion_tokens: int = 4096,
        exec_timeout: int = 600,
        task_timeout: int = 900,
        workers: int = 32,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
    ) -> None:
        self.max_completion_tokens = int(max_completion_tokens)
        self.exec_timeout = int(exec_timeout)
        self.task_timeout = int(task_timeout)
        self.workers = int(workers)
        self.analyst_workers = int(analyst_workers)
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.dataloader = DeepMathDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def build_reference_text(self, item: dict) -> str:
        gt = str(item.get("ground_truth") or "").strip()
        return f"## Ground Truth Answer\n{gt}" if gt else ""

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        return run_batch(
            items=list(env_manager),
            out_root=out_dir,
            skill_content=skill_content,
            workers=self.workers,
            exec_timeout=self.exec_timeout,
            max_completion_tokens=self.max_completion_tokens,
            task_timeout=self.task_timeout,
        )

    def reflect(self, results: list[dict], skill_content: str, out_dir: str, **kwargs) -> list[dict | None]:
        return run_minibatch_reflect(
            results=results,
            skill_content=skill_content,
            prediction_dir=kwargs.get("prediction_dir", os.path.join(out_dir, "predictions")),
            patches_dir=kwargs.get("patches_dir", os.path.join(out_dir, "patches")),
            workers=self.analyst_workers,
            failure_only=self.failure_only,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            random_seed=kwargs.get("random_seed"),
            error_system=self.get_error_minibatch_prompt(),
            success_system=self.get_success_minibatch_prompt(),
            step_buffer_context=kwargs.get("step_buffer_context", ""),
            meta_skill_context=kwargs.get("meta_skill_context", ""),
            update_mode=getattr(self, "_cfg", {}).get("skill_update_mode", "patch"),
        )

    def get_task_types(self) -> list[str]:
        return self.dataloader.get_task_types()
