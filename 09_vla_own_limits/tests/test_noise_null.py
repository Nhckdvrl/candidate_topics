"""Tests for the sampling-noise null control and the G1 power control."""
import numpy as np
import pandas as pd
import pytest

from src.noise_null import checkpoint_success_summary, permutation_noise_null
from src.relative_probe import absolute_success_metrics


def _panel(rate_a, rate_b, *, n_trials=8, seed=0):
    """Build a behavior panel from per-state true success probabilities."""
    rng = np.random.default_rng(seed)
    rows = []
    for init_idx, (pa, pb) in enumerate(zip(rate_a, rate_b, strict=True)):
        for cp, p in [("A", pa), ("B", pb)]:
            draws = rng.random(n_trials) < p
            for j in range(n_trials):
                rows.append({
                    "suite": "libero_10",
                    "task_id": 0,
                    "init_idx": init_idx,
                    "env_seed": 7,
                    "sim_state_hash": f"h{init_idx}",
                    "checkpoint": cp,
                    "policy_seed": 110000 + j,
                    "success": int(draws[j]),
                })
    return pd.DataFrame(rows)


def test_pure_sampling_noise_does_not_beat_its_own_null():
    """Two checkpoints with identical competence everywhere must not look like crossover."""
    p = np.full(150, 0.5)
    df = _panel(p, p, seed=3)
    r = permutation_noise_null(df, "A", "B", n_permutations=400, seed=0)
    # Some robust "wins" appear -- that is exactly the hazard this control exists for.
    assert r.observed_robust_disagree > 0
    # But they are indistinguishable from the relabeling null.
    assert r.p_bidirectional > 0.05
    assert r.p_robust_disagree > 0.05


def test_real_bidirectional_competence_beats_the_null():
    """Genuine opposite competence at many states must be far outside the null."""
    n = 150
    rate_a = np.full(n, 0.5)
    rate_b = np.full(n, 0.5)
    rate_a[:40], rate_b[:40] = 0.95, 0.05   # A truly wins
    rate_b[40:80], rate_a[40:80] = 0.95, 0.05  # B truly wins
    df = _panel(rate_a, rate_b, seed=4)
    r = permutation_noise_null(df, "A", "B", n_permutations=400, seed=0)
    assert r.observed_bidirectional >= 30
    assert r.p_bidirectional < 0.05
    assert r.p_robust_disagree < 0.05
    assert r.observed_bidirectional > r.null_bidirectional_p95


def test_null_holds_per_state_difficulty_exactly_fixed():
    """Relabeling must never change a state's pooled success count."""
    rng = np.random.default_rng(7)
    df = _panel(rng.random(20), rng.random(20), seed=5)
    pooled_before = df.groupby("init_idx").success.sum().to_dict()
    permutation_noise_null(df, "A", "B", n_permutations=50, seed=0)
    pooled_after = df.groupby("init_idx").success.sum().to_dict()
    assert pooled_before == pooled_after


def test_global_quality_gap_alone_gives_no_bidirectional_support():
    """A checkpoint that is simply better everywhere is not a crossover."""
    df = _panel(np.full(150, 0.9), np.full(150, 0.1), seed=6)
    r = permutation_noise_null(df, "A", "B", n_permutations=400, seed=0)
    assert r.observed_bidirectional == 0


def test_checkpoint_success_summary_reports_sane_rates():
    df = _panel(np.full(30, 1.0), np.full(30, 0.0), seed=1)
    s = {d["checkpoint"]: d for d in checkpoint_success_summary(df)}
    assert s["A"]["overall_success_rate"] == 1.0
    assert s["B"]["overall_success_rate"] == 0.0
    assert s["A"]["n_rollouts"] == 240


def test_absolute_control_detects_a_readout_with_no_success_signal():
    rng = np.random.default_rng(0)
    n = 100
    cp = np.asarray(["A"] * n + ["B"] * n)
    target = rng.random(2 * n)
    m = absolute_success_metrics(np.arange(2 * n).astype(str), cp, target, rng.normal(size=2 * n))
    assert abs(m["mean_within_checkpoint_spearman"]) < 0.15


def test_absolute_control_detects_a_readout_that_tracks_success():
    rng = np.random.default_rng(0)
    n = 100
    cp = np.asarray(["A"] * n + ["B"] * n)
    target = rng.random(2 * n)
    scores = target + 0.05 * rng.normal(size=2 * n)
    m = absolute_success_metrics(np.arange(2 * n).astype(str), cp, target, scores)
    assert m["mean_within_checkpoint_spearman"] > 0.9
