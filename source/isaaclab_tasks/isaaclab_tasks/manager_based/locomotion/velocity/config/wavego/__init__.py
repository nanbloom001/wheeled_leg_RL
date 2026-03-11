# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-WAVEGO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.WavegoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:WavegoFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-WAVEGO-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.WavegoFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:WavegoFlatPPORunnerCfg",
    },
)
