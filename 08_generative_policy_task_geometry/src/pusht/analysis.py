"""Analysis for the PushT existence test.

The question this file answers is deliberately blunt:

    Given the scalar action-diversity score of a state, how well do we already know that
    state's true task-outcome dispersion?

If the answer is "essentially perfectly", the topic is dead and no amount of geometry
helps. The functions below are written so that the kill case is as easy to see as the
survive case.

Dependence structure: probe states come from closed-loop rollouts and are strongly
correlated within a rollout. Every uncertainty estimate here resamples **rollouts**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def within_group_spearman(df: pd.DataFrame, xcol: str, ycol: str, group: str = "rollout") -> float:
    """Average Spearman computed inside each rollout, weighted by rollout size.

    Guards against a pooled correlation that is really a between-rollout effect.
    """
    rs, ws = [], []
    for _, g in df.groupby(group):
        if len(g) < 5:
            continue
        r = spearman(g[xcol], g[ycol])
        if np.isfinite(r):
            rs.append(r)
            ws.append(len(g))
    if not rs:
        return float("nan")
    return float(np.average(rs, weights=ws))


def matched_pairs(
    df: pd.DataFrame,
    score_col: str,
    outcome_col: str,
    tol_z: float = 0.10,
    require_different_rollout: bool = True,
) -> pd.DataFrame:
    """Greedy 1:1 matching of states with nearly identical scalar diversity score.

    Pairs are formed between the top and bottom quartile of `outcome_col` so that the
    reported ratio is the *achievable* spread at matched entropy, then the analysis
    reports how tight the entropy match actually was. Each state is used at most once.

    `require_different_rollout` avoids pairing two probes from the same rollout, which
    would otherwise let a single lucky episode manufacture many "independent" pairs.
    """
    s = df[score_col].to_numpy(float)
    o = df[outcome_col].to_numpy(float)
    z = (s - s.mean()) / max(s.std(ddof=1), 1e-12)
    roll = df["rollout"].to_numpy()

    hi_idx = np.where(o >= np.quantile(o, 0.75))[0]
    lo_idx = np.where(o <= np.quantile(o, 0.25))[0]

    cands = []
    for i in hi_idx:
        for j in lo_idx:
            if require_different_rollout and roll[i] == roll[j]:
                continue
            dz = abs(z[i] - z[j])
            if dz <= tol_z:
                cands.append((dz, i, j))
    cands.sort(key=lambda t: t[0])

    used, rows = set(), []
    for dz, i, j in cands:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        rows.append(
            {
                "dz": dz,
                "idx_hi": i,
                "idx_lo": j,
                "score_hi": s[i],
                "score_lo": s[j],
                "outcome_hi": o[i],
                "outcome_lo": o[j],
                "ratio": o[i] / max(o[j], 1e-9),
                "diff": o[i] - o[j],
            }
        )
    return pd.DataFrame(rows)


def binned_spread(df: pd.DataFrame, score_col: str, outcome_col: str, n_bins: int = 8) -> pd.DataFrame:
    """Within narrow quantile bins of the scalar score, how wide is the outcome spread?

    This is the assumption-light version of the matched-pair test: no pairing, no
    threshold, just "hold the scalar roughly fixed and look".
    """
    d = df[[score_col, outcome_col]].dropna().copy()
    d["bin"] = pd.qcut(d[score_col], n_bins, labels=False, duplicates="drop")
    out = []
    for b, g in d.groupby("bin"):
        if len(g) < 10:
            continue
        o = g[outcome_col].to_numpy(float)
        p10, p50, p90 = np.percentile(o, [10, 50, 90])
        out.append(
            {
                "bin": int(b),
                "n": len(g),
                "score_lo": float(g[score_col].min()),
                "score_hi": float(g[score_col].max()),
                "outcome_p10": float(p10),
                "outcome_p50": float(p50),
                "outcome_p90": float(p90),
                "p90_over_p10": float(p90 / max(p10, 1e-9)),
            }
        )
    return pd.DataFrame(out)


def rollout_bootstrap(df: pd.DataFrame, stat_fn, n: int = 2000, seed: int = 0) -> dict:
    """Bootstrap by resampling whole rollouts (states within a rollout are dependent)."""
    rng = np.random.default_rng(seed)
    rollouts = df["rollout"].unique()
    vals = []
    for _ in range(n):
        pick = rng.choice(rollouts, size=len(rollouts), replace=True)
        boot = pd.concat([df[df["rollout"] == r] for r in pick], ignore_index=True)
        try:
            v = stat_fn(boot)
        except Exception:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"point": float("nan"), "ci95": [None, None], "n_ok": 0}
    return {
        "point": float(stat_fn(df)),
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "n_ok": len(vals),
    }
