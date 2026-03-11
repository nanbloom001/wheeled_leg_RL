# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WavegoFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 50
    experiment_name = "wavego_flat"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2, # 续训微调, 降低探索噪声
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        learning_rate=1e-3,
        num_learning_epochs=5,
        num_mini_batches=1, # 极速模式：多卡同步效率最高
        gamma=0.99,
        lam=0.95,
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
