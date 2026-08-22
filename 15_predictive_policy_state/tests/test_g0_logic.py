from types import SimpleNamespace

import torch

from g0_lightwam import _analyse


def _args():
    return SimpleNamespace(
        probe_ridge=1e-2,
        target_chunk_size=32,
        seed=1,
        bootstrap=300,
        min_relative_effect=0.05,
    )


def test_proceed_verdict_requires_future_gain_and_action_effect():
    g = torch.Generator().manual_seed(5)
    n_train, n_test, d, p = 90, 30, 10, 18
    xtr = torch.randn(n_train, d, generator=g)
    xte = torch.randn(n_test, d, generator=g)
    w = torch.randn(d, p, generator=g)
    ytr = xtr @ w + 0.01 * torch.randn(n_train, p, generator=g)
    yte = xte @ w + 0.01 * torch.randn(n_test, p, generator=g)

    train = {
        "feature_normal": xtr,
        "feature_bypass": torch.randn(n_train, d, generator=g),
        "target_future": ytr,
        "loss_normal": torch.full((n_train,), 1.0),
        "loss_bypass": torch.full((n_train,), 1.2),
        "action_shift": torch.full((n_train,), 0.2),
    }
    test = {
        "feature_normal": xte,
        "feature_bypass": torch.randn(n_test, d, generator=g),
        "target_future": yte,
        "loss_normal": torch.full((n_test,), 1.0),
        "loss_bypass": torch.full((n_test,), 1.2),
        "action_shift": torch.full((n_test,), 0.2),
    }
    _, verdict = _analyse(train, test, _args())
    assert verdict == "PROCEED_TO_MATCHED_TRAINING"


def test_action_effect_without_future_gain_is_not_called_predictive_mediation():
    g = torch.Generator().manual_seed(9)
    n_train, n_test, d, p = 100, 40, 12, 20
    normal_train = torch.randn(n_train, d, generator=g)
    normal_test = torch.randn(n_test, d, generator=g)
    bypass_train = normal_train.clone()
    bypass_test = normal_test.clone()
    y_train = torch.randn(n_train, p, generator=g)
    y_test = torch.randn(n_test, p, generator=g)

    train = {
        "feature_normal": normal_train,
        "feature_bypass": bypass_train,
        "target_future": y_train,
        "loss_normal": torch.ones(n_train),
        "loss_bypass": torch.full((n_train,), 1.3),
        "action_shift": torch.full((n_train,), 0.3),
    }
    test = {
        "feature_normal": normal_test,
        "feature_bypass": bypass_test,
        "target_future": y_test,
        "loss_normal": torch.ones(n_test),
        "loss_bypass": torch.full((n_test,), 1.3),
        "action_shift": torch.full((n_test,), 0.3),
    }
    _, verdict = _analyse(train, test, _args())
    assert verdict == "ADAPTER_ACTION_EFFECT_WITHOUT_FUTURE_GAIN"
