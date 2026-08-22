"""Behavior-panel validation and robust same-state crossover statistics.

Topic 09 uses a stochastic generative policy. A single success/failure rollout is not a
property of (policy, state); it is one Monte-Carlo draw. This module therefore treats a
physical state as the unit of analysis and aggregates a *common* set of policy RNG seeds
for every checkpoint before deciding which checkpoint wins the state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

STATE_COLS = ["suite", "task_id", "init_idx", "env_seed"]
REQUIRED = set(STATE_COLS + ["sim_state_hash", "checkpoint", "policy_seed", "success"])


def make_state_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["suite"].astype(str)
        + "|t=" + df["task_id"].astype(int).astype(str)
        + "|i=" + df["init_idx"].astype(int).astype(str)
        + "|e=" + df["env_seed"].astype(int).astype(str)
    )


@dataclass(frozen=True)
class PairStats:
    checkpoint_a: str
    checkpoint_b: str
    n_states: int
    n_a_wins: int
    n_b_wins: int
    n_ambiguous: int
    min_trials_observed: int
    rate_gap: float

    @property
    def n_robust_disagree(self) -> int:
        return self.n_a_wins + self.n_b_wins

    @property
    def bidirectional_support(self) -> int:
        return min(self.n_a_wins, self.n_b_wins)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["n_robust_disagree"] = self.n_robust_disagree
        out["bidirectional_support"] = self.bidirectional_support
        return out


def validate_panel(df: pd.DataFrame, *, min_trials: int = 1) -> pd.DataFrame:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    d = df.copy()
    d["suite"] = d["suite"].astype(str)
    for c in ["task_id", "init_idx", "env_seed", "policy_seed"]:
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)
    d["checkpoint"] = d["checkpoint"].astype(str)
    d["sim_state_hash"] = d["sim_state_hash"].astype(str)
    if "status" in d.columns and not (d["status"].astype(str) == "ok").all():
        n_bad = int((d["status"].astype(str) != "ok").sum())
        raise ValueError(f"panel contains {n_bad} technical-error rows")
    d["success"] = pd.to_numeric(d["success"], errors="raise").astype(int)
    if not set(d.success.unique()) <= {0, 1}:
        raise ValueError("success must be binary 0/1")

    d["state_id"] = make_state_id(d)
    key = ["state_id", "checkpoint", "policy_seed"]
    if d.duplicated(key).any():
        raise ValueError("duplicate state/checkpoint/policy_seed rows")

    # Same task/init index is only scientifically the same state if the settled MuJoCo
    # dynamics state is identical across every checkpoint process and repeat.
    hashes_per_state = d.groupby("state_id")["sim_state_hash"].nunique()
    bad_hash = hashes_per_state[hashes_per_state != 1]
    if len(bad_hash):
        raise ValueError(f"sim_state_hash mismatch for {len(bad_hash)} physical states")

    checkpoints = tuple(sorted(d.checkpoint.unique()))
    if len(checkpoints) < 2:
        raise ValueError("need at least two checkpoints")

    # Common random numbers: every checkpoint must see exactly the same set of stochastic
    # policy draws for a physical state. Missing/mismatched draws are a technical error.
    for sid, g in d.groupby("state_id", sort=False):
        cps = tuple(sorted(g.checkpoint.unique()))
        if cps != checkpoints:
            raise ValueError(f"incomplete checkpoint panel for state {sid}")
        seed_sets = {
            cp: tuple(sorted(g.loc[g.checkpoint == cp, "policy_seed"].tolist()))
            for cp in checkpoints
        }
        reference = seed_sets[checkpoints[0]]
        if len(reference) < min_trials:
            raise ValueError(f"state {sid} has only {len(reference)} trials; need {min_trials}")
        if any(seed_sets[cp] != reference for cp in checkpoints[1:]):
            raise ValueError(f"policy_seed sets differ across checkpoints for state {sid}")

    return d


def aggregate_success(df: pd.DataFrame, *, min_trials: int = 1) -> pd.DataFrame:
    d = validate_panel(df, min_trials=min_trials)
    agg = (
        d.groupby(STATE_COLS + ["state_id", "sim_state_hash", "checkpoint"], as_index=False)
        .agg(n_trials=("success", "size"), n_success=("success", "sum"))
    )
    agg["success_rate"] = agg.n_success / agg.n_trials
    return agg


def pair_state_table(
    df: pd.DataFrame,
    a: str,
    b: str,
    *,
    min_trials: int = 8,
    rate_gap: float = 0.50,
) -> pd.DataFrame:
    if not 0 < rate_gap <= 1:
        raise ValueError("rate_gap must be in (0,1]")
    agg = aggregate_success(df, min_trials=min_trials)
    sub = agg[agg.checkpoint.isin([str(a), str(b)])]
    p = sub.pivot(index="state_id", columns="checkpoint", values="success_rate")
    n = sub.pivot(index="state_id", columns="checkpoint", values="n_trials")
    if str(a) not in p.columns or str(b) not in p.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=p.index)
    out["p_a"] = p[str(a)]
    out["p_b"] = p[str(b)]
    out["n_a"] = n[str(a)]
    out["n_b"] = n[str(b)]
    out = out.dropna()
    out["delta_p"] = out.p_a - out.p_b
    out["winner"] = np.where(
        out.delta_p >= rate_gap,
        "A",
        np.where(out.delta_p <= -rate_gap, "B", "ambiguous"),
    )
    meta = agg.drop_duplicates("state_id").set_index("state_id")[STATE_COLS + ["sim_state_hash"]]
    out = out.join(meta, how="left").reset_index()
    return out


def pair_stats(
    df: pd.DataFrame,
    a: str,
    b: str,
    *,
    min_trials: int = 8,
    rate_gap: float = 0.50,
) -> PairStats:
    t = pair_state_table(df, a, b, min_trials=min_trials, rate_gap=rate_gap)
    if len(t) == 0:
        return PairStats(str(a), str(b), 0, 0, 0, 0, 0, rate_gap)
    return PairStats(
        checkpoint_a=str(a),
        checkpoint_b=str(b),
        n_states=int(len(t)),
        n_a_wins=int((t.winner == "A").sum()),
        n_b_wins=int((t.winner == "B").sum()),
        n_ambiguous=int((t.winner == "ambiguous").sum()),
        min_trials_observed=int(min(t.n_a.min(), t.n_b.min())),
        rate_gap=float(rate_gap),
    )


def all_pair_stats(df: pd.DataFrame, *, min_trials: int = 8, rate_gap: float = 0.50) -> list[PairStats]:
    d = validate_panel(df, min_trials=min_trials)
    cps = sorted(d.checkpoint.unique())
    return [pair_stats(d, a, b, min_trials=min_trials, rate_gap=rate_gap) for a, b in combinations(cps, 2)]


def choose_identifiable_pair(
    df: pd.DataFrame, *, min_trials: int = 8, rate_gap: float = 0.50
) -> PairStats | None:
    stats = all_pair_stats(df, min_trials=min_trials, rate_gap=rate_gap)
    if not stats:
        return None
    return max(stats, key=lambda x: (x.bidirectional_support, x.n_robust_disagree, x.n_states))


def task_pair_table(
    df: pd.DataFrame,
    a: str,
    b: str,
    *,
    min_trials: int = 8,
    rate_gap: float = 0.50,
) -> list[dict]:
    d = validate_panel(df, min_trials=min_trials)
    rows = []
    for task_id, g in d.groupby("task_id", sort=True):
        s = pair_stats(g, a, b, min_trials=min_trials, rate_gap=rate_gap)
        r = s.to_dict()
        r["task_id"] = int(task_id)
        rows.append(r)
    return rows
