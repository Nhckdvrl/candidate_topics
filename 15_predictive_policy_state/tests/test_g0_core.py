import torch

from g0_core import (
    future_change_target,
    linear_ridge_probe,
    paired_bootstrap_mean_ci,
    relative_change,
)


def test_future_change_target_shape_and_values():
    x = torch.zeros(2, 3, 4, 2, 2)
    x[:, :, 1] = 1.0
    x[:, :, 2] = 2.0
    x[:, :, 3] = 3.0
    y = future_change_target(x)
    assert y.shape == (2, 3 * 3 * 2 * 2)
    assert torch.allclose(y.mean(dim=1), torch.tensor([2.0, 2.0]))


def test_ridge_probe_recovers_linear_signal():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(120, 12, generator=g)
    w = torch.randn(12, 20, generator=g)
    y = x @ w + 0.01 * torch.randn(120, 20, generator=g)
    result = linear_ridge_probe(x[:90], y[:90], x[90:], y[90:], ridge=1e-2)
    assert result.r2 > 0.95
    assert result.mse < result.baseline_mse


def test_ridge_probe_does_not_create_signal_from_noise():
    g = torch.Generator().manual_seed(11)
    x = torch.randn(160, 24, generator=g)
    y = torch.randn(160, 32, generator=g)
    result = linear_ridge_probe(x[:120], y[:120], x[120:], y[120:], ridge=1e-2)
    assert result.r2 < 0.25


def test_bootstrap_and_relative_change():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mean, lo, hi = paired_bootstrap_mean_ci(x, seed=3, n_boot=500)
    assert lo <= mean <= hi
    assert abs(mean - 2.5) < 1e-9
    assert abs(relative_change(1.1, 1.0) - 0.1) < 1e-9
