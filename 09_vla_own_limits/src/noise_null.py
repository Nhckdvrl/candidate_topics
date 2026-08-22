"""Sampling-noise null control for the G0 crossover claim.

pi0.5 is a stochastic flow policy, so `p_hat = successes/8` is a noisy estimate. Two
checkpoints with *identical* competence at a state can still show `|p_hat_A - p_hat_B| >=
0.5` purely because different Gaussian action-noise samples were drawn. With eight
rollouts that happens for about 3.8% of states at p=0.5, so a 150-state discovery panel
would be expected to produce roughly six spurious "A-wins" and six spurious "B-wins"
with no competence difference anywhere.

The frozen `min(n_a_wins, n_b_wins) >= 15` rule is therefore necessary but not
self-evidently sufficient. This module supplies the missing evidence: an exact
within-state relabeling test that asks how much bidirectional crossover the observed
rollout data would produce if checkpoint identity carried no information at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .panel import aggregate_success, validate_panel


@dataclass(frozen=True)
class NullResult:
    n_permutations: int
    observed_bidirectional: int
    observed_robust_disagree: int
    null_bidirectional_mean: float
    null_bidirectional_p95: float
    null_robust_disagree_mean: float
    null_robust_disagree_p95: float
    p_bidirectional: float
    p_robust_disagree: float

    def to_dict(self) -> dict:
        return {
            "n_permutations": self.n_permutations,
            "observed_bidirectional": self.observed_bidirectional,
            "observed_robust_disagree": self.observed_robust_disagree,
            "null_bidirectional_mean": self.null_bidirectional_mean,
            "null_bidirectional_p95": self.null_bidirectional_p95,
            "null_robust_disagree_mean": self.null_robust_disagree_mean,
            "null_robust_disagree_p95": self.null_robust_disagree_p95,
            "p_bidirectional": self.p_bidirectional,
            "p_robust_disagree": self.p_robust_disagree,
        }


def _state_outcome_matrix(df: pd.DataFrame, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    """Return [n_states, n_trials] success matrices for the two checkpoints."""
    d = df[df.checkpoint.isin([str(a), str(b)])]
    rows_a, rows_b = [], []
    for _sid, g in d.groupby("state_id", sort=True):
        ga = g[g.checkpoint == str(a)].sort_values("policy_seed")
        gb = g[g.checkpoint == str(b)].sort_values("policy_seed")
        if len(ga) == 0 or len(ga) != len(gb):
            continue
        rows_a.append(ga.success.to_numpy(dtype=np.int8))
        rows_b.append(gb.success.to_numpy(dtype=np.int8))
    if not rows_a:
        raise ValueError("no complete states for the requested pair")
    return np.stack(rows_a), np.stack(rows_b)


def permutation_noise_null(
    df: pd.DataFrame,
    a: str,
    b: str,
    *,
    min_trials: int = 8,
    rate_gap: float = 0.50,
    n_permutations: int = 2000,
    seed: int = 0,
) -> NullResult:
    """Exact within-state relabeling null for robust crossover counts.

    For every physical state the `2*n_trials` observed rollout outcomes are pooled and
    randomly re-split into two groups of `n_trials`. This holds each state's true
    difficulty *exactly* fixed (the pooled success count never changes) and destroys only
    the association between outcome and checkpoint identity. Whatever crossover survives
    is by construction sampling noise.

    Note this null is symmetric in A/B, whereas a genuine global quality gap would push
    wins toward one checkpoint and *lower* `min(n_a_wins, n_b_wins)`. The bidirectional
    statistic is therefore compared conservatively, and the direction-free
    `n_robust_disagree` count is reported alongside it as the cleaner noise diagnostic.
    """
    d = validate_panel(df, min_trials=min_trials)
    sa, sb = _state_outcome_matrix(d, a, b)
    n_states, n_trials = sa.shape
    if sb.shape != sa.shape:
        raise ValueError("checkpoint outcome matrices are not aligned")

    pooled = np.concatenate([sa, sb], axis=1)  # [n_states, 2*n_trials]
    obs_delta = sa.mean(axis=1) - sb.mean(axis=1)
    obs_a = int((obs_delta >= rate_gap).sum())
    obs_b = int((obs_delta <= -rate_gap).sum())
    obs_bi = min(obs_a, obs_b)
    obs_rd = obs_a + obs_b

    rng = np.random.default_rng(seed)
    null_bi = np.empty(n_permutations, dtype=np.int32)
    null_rd = np.empty(n_permutations, dtype=np.int32)
    for k in range(int(n_permutations)):
        # independent random re-split per state
        order = np.argsort(rng.random(pooled.shape), axis=1)
        shuffled = np.take_along_axis(pooled, order, axis=1)
        delta = shuffled[:, :n_trials].mean(axis=1) - shuffled[:, n_trials:].mean(axis=1)
        na = int((delta >= rate_gap).sum())
        nb = int((delta <= -rate_gap).sum())
        null_bi[k] = min(na, nb)
        null_rd[k] = na + nb

    return NullResult(
        n_permutations=int(n_permutations),
        observed_bidirectional=obs_bi,
        observed_robust_disagree=obs_rd,
        null_bidirectional_mean=float(null_bi.mean()),
        null_bidirectional_p95=float(np.quantile(null_bi, 0.95)),
        null_robust_disagree_mean=float(null_rd.mean()),
        null_robust_disagree_p95=float(np.quantile(null_rd, 0.95)),
        # +1 smoothing: a permutation p-value can never legitimately be exactly zero.
        p_bidirectional=float((1 + (null_bi >= obs_bi).sum()) / (1 + n_permutations)),
        p_robust_disagree=float((1 + (null_rd >= obs_rd).sum()) / (1 + n_permutations)),
    )


def checkpoint_success_summary(df: pd.DataFrame, *, min_trials: int = 8) -> list[dict]:
    """Overall per-checkpoint LIBERO success, for comparison against published numbers."""
    agg = aggregate_success(df, min_trials=min_trials)
    out = []
    for cp, g in agg.groupby("checkpoint", sort=True):
        out.append({
            "checkpoint": str(cp),
            "n_states": int(len(g)),
            "n_rollouts": int(g.n_trials.sum()),
            "overall_success_rate": float(g.n_success.sum() / g.n_trials.sum()),
            "mean_state_success_rate": float(g.success_rate.mean()),
        })
    return out
