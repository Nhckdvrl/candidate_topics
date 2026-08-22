"""Analysis helpers must behave correctly in both the KILL and the SURVIVE regime."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pusht.analysis import (
    binned_spread,
    matched_pairs,
    rollout_bootstrap,
    spearman,
    within_group_spearman,
)
from src.pusht.geometry_e2 import local_sensitivity


def _frame(n_rollouts=10, n_per=12, decoupled=False, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_rollouts):
        for p in range(n_per):
            ace = rng.uniform(0, 5)
            # KILL regime: outcome is a monotone function of ace.
            # SURVIVE regime: an unobserved factor moves the outcome at fixed ace.
            hidden = rng.uniform(0.2, 5.0) if decoupled else 1.0
            rows.append({"rollout": r, "probe": p, "ace": ace, "out": ace * hidden + 1e-3})
    return pd.DataFrame(rows)


def test_kill_regime_is_detected_as_near_perfect_correlation():
    df = _frame(decoupled=False)
    assert spearman(df.ace, df.out) > 0.99
    bs = binned_spread(df, "ace", "out", n_bins=5)
    assert bs.p90_over_p10.median() < 1.6


def test_survive_regime_leaves_spread_at_matched_score():
    df = _frame(decoupled=True, seed=1)
    bs = binned_spread(df, "ace", "out", n_bins=5)
    assert bs.p90_over_p10.median() > 3.0


def test_matched_pairs_respects_tolerance_and_uses_each_state_once():
    df = _frame(decoupled=True, seed=2)
    mp = matched_pairs(df, "ace", "out", tol_z=0.05)
    assert len(mp) > 0
    assert mp.dz.max() <= 0.05 + 1e-12
    assert len(set(mp.idx_hi)) == len(mp)
    assert len(set(mp.idx_lo)) == len(mp)
    assert set(mp.idx_hi).isdisjoint(set(mp.idx_lo))


def test_matched_pairs_never_pairs_within_one_rollout():
    df = _frame(decoupled=True, seed=3)
    mp = matched_pairs(df, "ace", "out", tol_z=0.2, require_different_rollout=True)
    roll = df["rollout"].to_numpy()
    assert all(roll[i] != roll[j] for i, j in zip(mp.idx_hi, mp.idx_lo))


def test_within_group_spearman_ignores_between_rollout_structure():
    # ace and out agree only across rollouts; inside a rollout they are unrelated.
    rng = np.random.default_rng(4)
    rows = []
    for r in range(8):
        offset = 10.0 * r
        for p in range(15):
            rows.append({"rollout": r, "probe": p, "ace": offset + rng.normal(), "out": offset + rng.normal()})
    df = pd.DataFrame(rows)
    assert spearman(df.ace, df.out) > 0.9
    assert abs(within_group_spearman(df, "ace", "out")) < 0.5


def test_rollout_bootstrap_resamples_rollouts_not_states():
    df = _frame(decoupled=True, seed=5)
    res = rollout_bootstrap(df, lambda d: spearman(d.ace, d.out), n=100, seed=0)
    assert res["n_ok"] > 50
    assert res["ci95"][0] <= res["point"] <= res["ci95"][1]


def test_local_sensitivity_recovers_a_known_linear_map():
    rng = np.random.default_rng(6)
    b, h, k = 128, 8, 8
    chunks = rng.normal(scale=5.0, size=(b, h, 2))
    w_true = np.zeros((h * 2, k * 2))
    w_true[0, 0] = 2.0  # only the first action dimension moves the block
    kps = (chunks.reshape(b, -1) @ w_true).reshape(b, k, 2)

    out = local_sensitivity(chunks, kps)
    assert out["r2_cv"] > 0.99
    assert out["sensitivity_rank95"] == 1
    # nearly all action variance lies in goal-equivalent directions (15 of 16 dims)
    assert out["task_fraction"] < 0.15
    assert out["predicted_outcome_dispersion_px"] > 0


def test_local_sensitivity_flags_an_untrustworthy_linearisation():
    rng = np.random.default_rng(7)
    b = 128
    chunks = rng.normal(size=(b, 8, 2))
    kps = rng.normal(size=(b, 8, 2))  # outcome unrelated to the action
    out = local_sensitivity(chunks, kps)
    assert out["r2_cv"] < 0.2
