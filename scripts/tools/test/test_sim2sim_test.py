import numpy as np

from scripts.sim2sim_test import (
    NUM_OBS,
    ObsNormalizer,
    apply_obs_normalization,
    build_policy_observation,
    joint_pos_rel,
    joint_vel_rel,
    projected_gravity,
)


def test_projected_gravity_identity():
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    out = projected_gravity(quat_wxyz, gravity)
    assert np.allclose(out, gravity, atol=1e-9)


def test_projected_gravity_quaternion_rotation():
    theta = np.pi / 2.0
    quat_wxyz = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0], dtype=np.float64)
    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    out = projected_gravity(quat_wxyz, gravity)
    expected = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    assert np.allclose(out, expected, atol=1e-8)


def test_joint_rel_terms():
    q = np.array([1.0, 2.0, 3.0])
    q0 = np.array([0.5, 2.5, 2.0])
    dq = np.array([0.2, -0.4, 0.8])
    dq0 = np.array([0.1, -0.2, 0.3])

    assert np.allclose(joint_pos_rel(q, q0), np.array([0.5, -0.5, 1.0]))
    assert np.allclose(joint_vel_rel(dq, dq0), np.array([0.1, -0.2, 0.5]))


def test_policy_observation_dim_and_order():
    root_lin_vel_b = np.array([1.0, 2.0, 3.0])
    root_ang_vel_b = np.array([4.0, 5.0, 6.0])
    root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    gravity = np.array([0.0, 0.0, -1.0])
    command = np.array([7.0, 8.0, 9.0])
    joint_pos = np.arange(12, dtype=np.float64)
    default_joint_pos = np.ones(12, dtype=np.float64)
    joint_vel = np.arange(12, dtype=np.float64) + 100.0
    default_joint_vel = np.zeros(12, dtype=np.float64)
    previous_action = np.arange(12, dtype=np.float64) + 200.0

    obs = build_policy_observation(
        root_lin_vel_b=root_lin_vel_b,
        root_ang_vel_b=root_ang_vel_b,
        root_link_quat_wxyz=root_quat_wxyz,
        gravity_vec_w=gravity,
        command=command,
        joint_pos=joint_pos,
        default_joint_pos=default_joint_pos,
        joint_vel=joint_vel,
        default_joint_vel=default_joint_vel,
        previous_action=previous_action,
    )

    assert obs.shape == (NUM_OBS,)
    assert np.allclose(obs[0:3], root_lin_vel_b)
    assert np.allclose(obs[3:6], root_ang_vel_b)
    assert np.allclose(obs[6:9], gravity)
    assert np.allclose(obs[9:12], command)
    assert np.allclose(obs[12:24], joint_pos - default_joint_pos)
    assert np.allclose(obs[24:36], joint_vel - default_joint_vel)
    assert np.allclose(obs[36:48], previous_action)


def test_normalization_and_clipping():
    obs = np.zeros(NUM_OBS, dtype=np.float64)
    mean = np.zeros(NUM_OBS, dtype=np.float64)
    std = np.ones(NUM_OBS, dtype=np.float64)
    std[0] = 0.01

    obs[0] = 100.0
    normalizer = ObsNormalizer(mean=mean, std=std, eps=1e-8, clip=5.0)
    obs_norm = apply_obs_normalization(obs, normalizer)

    assert obs_norm.shape == (NUM_OBS,)
    assert obs_norm[0] == 5.0
    assert np.allclose(obs_norm[1:], 0.0)