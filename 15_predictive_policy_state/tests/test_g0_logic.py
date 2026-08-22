from types import SimpleNamespace

import torch

from g0_lightwam import _analyse


def test_promising_native_mediator_verdict_on_clean_synthetic_signal():
    g = torch.Generator().manual_seed(5)
    n_train, n_test, d, p = 90, 30, 10, 18
    xtr = torch.randn(n_train, d, generator=g)
    xte = torch.randn(n_test, d, generator=g)
    w = torch.randn(d, p, generator=g)
    ytr = xtr @ w + 0.01 * torch.randn(n_train, p, generator=g)
    yte = xte @ w + 0.01 * torch.randn(n_test, p, generator=g)

    train = {
        "feature_adapted": xtr,
        "feature_backbone": torch.randn(n_train, d, generator=g),
        "target_future": ytr,
        "loss_adapted": torch.full((n_train,), 1.0),
        "loss_backbone": torch.full((n_train,), 1.2),
        "action_shift": torch.full((n_train,), 0.2),
    }
    test = {
        "feature_adapted": xte,
        "feature_backbone": torch.randn(n_test, d, generator=g),
        "target_future": yte,
        "loss_adapted": torch.full((n_test,), 1.0),
        "loss_backbone": torch.full((n_test,), 1.2),
        "action_shift": torch.full((n_test,), 0.2),
    }
    args = SimpleNamespace(
        probe_ridge=1e-2,
        target_chunk_size=32,
        seed=1,
        bootstrap=300,
        min_relative_effect=0.05,
    )
    _, verdict = _analyse(train, test, args)
    assert verdict == "PROMISING_NATIVE_MEDIATOR"
