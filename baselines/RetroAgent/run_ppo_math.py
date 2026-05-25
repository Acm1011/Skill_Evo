from __future__ import annotations

import sys
from pathlib import Path

import hydra

_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_RETRO = _ROOT / "RetroAgent" / "rl_trained_self_reflection"
if str(_UPSTREAM_RETRO) in sys.path:
    sys.path.remove(str(_UPSTREAM_RETRO))
sys.path.insert(0, str(_UPSTREAM_RETRO))

import verl.trainer.main_ppo as retro_main_ppo

_CFG_DIR = Path(__file__).resolve().parent / "config"


@hydra.main(
    config_path=str(_CFG_DIR),
    config_name="ppo_trainer_math",
    version_base=None,
)
def main(config):
    retro_main_ppo.run_ppo(config)


if __name__ == "__main__":
    main()
