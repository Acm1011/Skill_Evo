from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict

_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_RETRO = _ROOT / "RetroAgent" / "rl_trained_self_reflection"
if str(_UPSTREAM_RETRO) not in sys.path:
    sys.path.append(str(_UPSTREAM_RETRO))

import verl.trainer.main_ppo as retro_main_ppo

_CFG_DIR = Path(__file__).resolve().parent / "config"


@hydra.main(
    config_path=str(_CFG_DIR),
    config_name="ppo_trainer_math",
    version_base=None,
)
def main(config):
    # The RetroAgent baseline config was copied from an older verl variant.
    # Normalize the small set of schema differences that newer Skill_Evo/verl
    # now expects so the baseline remains portable across machines.
    with open_dict(config):
        if "ray_kwargs" not in config:
            config.ray_kwargs = OmegaConf.create({})
        if "ray_init" in config and "ray_init" not in config.ray_kwargs:
            config.ray_kwargs.ray_init = OmegaConf.create(
                OmegaConf.to_container(config.ray_init, resolve=False)
            )
        if "timeline_json_file" not in config.ray_kwargs:
            config.ray_kwargs.timeline_json_file = None

        if "global_profiler" not in config:
            config.global_profiler = OmegaConf.create(
                {
                    "tool": None,
                    "steps": None,
                    "global_tool_config": {
                        "nsys": {
                            "controller_nsight_options": {},
                        }
                    },
                }
            )

        if "enable_resource_pool" not in config.reward_model:
            config.reward_model.enable_resource_pool = False
        if "reward_kwargs" not in config.reward_model:
            config.reward_model.reward_kwargs = {}
        if "nnodes" not in config.reward_model:
            config.reward_model.nnodes = 0
        if "n_gpus_per_node" not in config.reward_model:
            config.reward_model.n_gpus_per_node = 0

        if "sampler" not in config.data:
            config.data.sampler = None
    retro_main_ppo.run_ppo(config)


if __name__ == "__main__":
    main()
