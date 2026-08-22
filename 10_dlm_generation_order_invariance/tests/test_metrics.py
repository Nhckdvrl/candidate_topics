import math

from metrics import kendall_tau_b, spearman


def test_kendall_identity_and_reverse():
    x = [1, 2, 3, 4]
    assert math.isclose(kendall_tau_b(x, x), 1.0)
    assert math.isclose(kendall_tau_b(x, list(reversed(x))), -1.0)


def test_kendall_ties():
    x = [1, 1, 2, 3]
    y = [1, 1, 2, 3]
    assert math.isclose(kendall_tau_b(x, y), 1.0)


def test_spearman_monotonic():
    assert math.isclose(spearman([1, 2, 3], [10, 20, 30]), 1.0)
