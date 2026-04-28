"""
PPO training entry for math + experience (uses baselines/EvolveR/config/ppo_trainer_math.yaml).
Run:  export PYTHONPATH=/path/to/Skill_Evo/baselines/EvolveR:$PYTHONPATH
      python run_ppo_math.py
Vendored verl is unchanged; evolver is this tree.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import ray
import verl.trainer.main_ppo as verl_main_ppo

_CFG_DIR = Path(__file__).resolve().parent / "config"


@hydra.main(
    config_path=str(_CFG_DIR),
    config_name="ppo_trainer_math",
    version_base=None,
)
def main(config):
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"},
            }
        )
    ray.get(verl_main_ppo.main_task.remote(config))


if __name__ == "__main__":
    main()
