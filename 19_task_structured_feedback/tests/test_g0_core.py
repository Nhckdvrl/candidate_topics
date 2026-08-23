import numpy as np

from g0_core import (
    accommodation,
    construct_perturbations,
    finite_geometry_gate,
    response_metrics,
    verdict,
)


def test_constructed_null_is_full_pose_null_and_equal_norm():
    rng = np.random.default_rng(7)
    j6 = rng.normal(size=(6, 7))
    pair = construct_perturbations(j6[:3], j6[3:], epsilon=0.08)
    assert np.isclose(np.linalg.norm(pair.task), 0.08)
    assert np.isclose(np.linalg.norm(pair.null), 0.08)
    assert np.linalg.norm(j6 @ pair.null) < 1e-10
    assert np.linalg.norm(j6[:3] @ pair.task) > 1e-3


def test_absolute_target_metric_has_correct_semantics():
    q_target = np.zeros(7)
    d = np.array([0.08, 0, 0, 0, 0, 0, 0.0])
    assert accommodation(q_target, q_target, d) == 0.0
    m = response_metrics(q_target, q_target, q_target + d, d, d)
    assert np.isclose(m.correction_task, 1.0)
    assert np.isclose(m.correction_null, 0.0)
    assert np.isclose(m.delta_correction, 1.0)


def test_identical_policy_response_is_not_positive_by_construction():
    base = np.arange(7, dtype=float)
    dt = np.ones(7) * 0.01
    dn = np.array([1, -1, 1, -1, 1, -1, 1], dtype=float)
    dn = dn / np.linalg.norm(dn) * np.linalg.norm(dt)
    m = response_metrics(base, base, base, dt, dn)
    assert np.isclose(m.delta_correction, 0.0)


def test_finite_geometry_gate():
    ok, diag = finite_geometry_gate(0.012, 0.0008, np.deg2rad(0.3))
    assert ok
    assert diag["translation_ratio"] > 5


def test_verdicts():
    assert verdict(0.30, 0.08, 0.50) == "PROCEED_TASK_STRUCTURED_FEEDBACK"
    assert verdict(0.02, -0.03, 0.09) == "KILL_NO_MEANINGFUL_TASK_SELECTIVITY"
    assert verdict(0.15, -0.02, 0.31) == "INCONCLUSIVE_DO_NOT_TUNE"
