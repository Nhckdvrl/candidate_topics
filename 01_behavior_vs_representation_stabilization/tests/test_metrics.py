import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from metrics import double_center, kl_proxy_bits_per_byte, linear_cka


def test_double_center_zero_means():
    x = np.array([[1., 2., 7.], [2., 4., 8.], [0., 1., 3.]])
    q = double_center(x)
    assert np.allclose(q.mean(axis=0), 0.0)
    assert np.allclose(q.mean(axis=1), 0.0)


def test_kl_proxy_zero_for_identical():
    q = np.array([1., 2., 3.])
    assert kl_proxy_bits_per_byte(q, q, 100.0) == 0.0


def test_linear_cka_identical_and_orthogonal_transform():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    assert linear_cka(x, x) > 0.999999
    # Linear CKA is invariant to orthogonal rotations.
    assert linear_cka(x, x @ q) > 0.999999
