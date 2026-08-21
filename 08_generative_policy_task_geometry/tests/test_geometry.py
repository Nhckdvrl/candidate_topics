import numpy as np

from src.geometry import (
    task_null_projectors,
    decompose_variance,
    fiper_calibration_ranges,
    fiper_ace,
)


def test_projectors_are_orthogonal_and_complete():
    j = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    pt, pn, rt, rn = task_null_projectors(j)
    assert rt == 2 and rn == 2
    np.testing.assert_allclose(pt @ pt, pt, atol=1e-10)
    np.testing.assert_allclose(pn @ pn, pn, atol=1e-10)
    np.testing.assert_allclose(pt @ pn, 0.0, atol=1e-10)
    np.testing.assert_allclose(pt + pn, np.eye(4), atol=1e-10)


def test_variance_decomposition_recovers_task_vs_null():
    rng = np.random.default_rng(0)
    j = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    x_task = np.zeros((10000, 4))
    x_task[:, :2] = rng.normal(scale=2.0, size=(10000, 2))
    vt = decompose_variance(x_task, j)
    assert vt.task_per_dim > 3.5
    assert vt.null_per_dim < 1e-8

    x_null = np.zeros((10000, 4))
    x_null[:, 2:] = rng.normal(scale=2.0, size=(10000, 2))
    vn = decompose_variance(x_null, j)
    assert vn.null_per_dim > 3.5
    assert vn.task_per_dim < 1e-8


def test_dimension_normalization_prevents_null_dim_artifact():
    rng = np.random.default_rng(1)
    j = np.array([[1.0, 0.0, 0.0, 0.0]])
    x = rng.normal(size=(30000, 4))
    v = decompose_variance(x, j)
    assert 2.6 < v.null_total / v.task_total < 3.4
    assert 0.9 < v.null_per_dim / v.task_per_dim < 1.1


def test_fiper_ace_low_for_single_cell_high_for_spread():
    cal = np.zeros((100, 4, 2), dtype=float)
    cal[..., 0] = np.linspace(-1, 1, 100)[:, None]
    cal[..., 1] = np.linspace(-1, 1, 100)[:, None]
    ranges = fiper_calibration_ranges(cal)
    same = np.zeros((64, 4, 2))
    spread = np.zeros((64, 4, 2))
    vals = np.linspace(-0.9, 0.9, 64)
    spread[:, :, 0] = vals[:, None]
    spread[:, :, 1] = vals[::-1, None]
    assert fiper_ace(same, ranges, alpha=0.1) == 0.0
    assert fiper_ace(spread, ranges, alpha=0.1) > 5.0
