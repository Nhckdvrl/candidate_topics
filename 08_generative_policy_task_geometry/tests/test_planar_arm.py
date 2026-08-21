import numpy as np

from src.planar_arm import PlanarArm, generate_episode, generate_dataset


def test_jacobian_matches_finite_difference():
    arm = PlanarArm()
    q = np.array([0.2, -0.5, 0.7, -0.3])
    j = arm.jacobian(q)
    eps = 1e-6
    num = np.zeros_like(j)
    for k in range(arm.n_joints):
        dq = np.zeros(arm.n_joints)
        dq[k] = eps
        num[:, k] = (arm.fk(q + dq) - arm.fk(q - dq)) / (2 * eps)
    np.testing.assert_allclose(j, num, atol=1e-6, rtol=1e-5)


def test_expert_reaches_with_null_motion():
    arm = PlanarArm()
    rng = np.random.default_rng(3)
    wins = 0
    for _ in range(30):
        ep = generate_episode(arm, rng, null_gain=1.0)
        wins += int(ep["success"])
    assert wins >= 27


def test_dataset_shapes_and_nontrivial_actions():
    data, summary = generate_dataset(10, 2, horizon=8, seed=4, null_gain=1.0)
    assert data["obs"].shape[1] == 6
    assert data["action"].shape[1:] == (8, 4)
    assert summary["success_rate"] > 0.8
    assert np.std(data["action"]) > 0.05
