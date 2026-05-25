from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict

_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_RETRO = _ROOT / "RetroAgent" / "rl_trained_self_reflection"
_VERL_GENERATED_CFG = _ROOT / "verl" / "trainer" / "config" / "_generated_ppo_trainer.yaml"
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
    # Start from the fully expanded current verl config, then overlay the
    # baseline config so we inherit new required schema while preserving the
    # RetroAgent-specific settings and CLI overrides.
    config = OmegaConf.merge(OmegaConf.load(_VERL_GENERATED_CFG), config)

    # Normalize remaining schema differences across old/new verl layouts.
    with open_dict(config):
        if "ray_kwargs" not in config:
            config.ray_kwargs = OmegaConf.create({})
        if "ray_init" in config and "ray_init" not in config.ray_kwargs:
            config.ray_kwargs.ray_init = OmegaConf.create(
                OmegaConf.to_container(config.ray_init, resolve=False)
            )
        if "timeline_json_file" not in config.ray_kwargs:
            config.ray_kwargs.timeline_json_file = None
    retro_main_ppo.run_ppo(config)


if __name__ == "__main__":
    main()
