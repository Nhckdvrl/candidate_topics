import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from metrics import (
    double_center,
    kl_proxy_bits_per_byte,
    linear_cka,
    lower_clip_per_checkpoint,
    mean_cosine_drift,
    pair_outlier_mask,
    pooled_standardized_drift,
)


def test_double_center_zero_means():
    x = np.array([[1.0, 2.0, 7.0], [2.0, 4.0, 8.0], [0.0, 1.0, 3.0]])
    q = double_center(x)
    assert np.allclose(q.mean(axis=0), 0.0)
    assert np.allclose(q.mean(axis=1), 0.0)


def test_kl_proxy_zero_for_identical():
    q = np.array([1.0, 2.0, 3.0])
    assert kl_proxy_bits_per_byte(q, q, 100.0) == 0.0


def test_lower_clip_only_changes_lower_tail():
    x = np.array([[-100.0, -4.0, -3.0, -2.0, -1.0]])
    clipped = lower_clip_per_checkpoint(x, 0.2)
    assert clipped[0, 0] > x[0, 0]
    assert np.allclose(clipped[0, 1:], x[0, 1:])


def test_pair_outlier_mask_removes_extreme_change():
    ll = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.1, 0.2, 50.0],
        ]
    )
    keep, scores = pair_outlier_mask(ll, [(0, 1)], trim_fraction=0.2)
    assert keep.sum() == 4
    assert not keep[4]
    assert scores[4] == scores.max()


def test_linear_cka_identical_and_rotation_invariant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    assert linear_cka(x, x) > 0.999999
    assert linear_cka(x, x @ q) > 0.999999


def test_matched_drifts_detect_change():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 16))
    y = x.copy()
    assert mean_cosine_drift(x, y) < 1e-12
    assert pooled_standardized_drift(x, y) < 1e-12
    y[:, 0] += 2.0
    assert mean_cosine_drift(x, y) > 0.0
    assert pooled_standardized_drift(x, y) > 0.0
