# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .rough_env_cfg import QweDogRoughEnvCfg, QweDogRoughEnvCfg_PLAY
from .flat_env_cfg import QweDogFlatEnvCfg, QweDogFlatEnvCfg_PLAY

##
# Register Gym environments.
##

# Rough Terrain
gym.register(
    id="Isaac-Velocity-Rough-QweDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QweDogRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg", 
        "skrl_cfg_entry_point": f"{__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-QweDog-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QweDogRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

# Flat Terrain
gym.register(
    id="Isaac-Velocity-Flat-QweDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QweDogFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-QweDog-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QweDogFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
