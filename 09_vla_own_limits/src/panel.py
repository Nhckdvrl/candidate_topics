"""Utilities for the same-state / multi-checkpoint behavioral panel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import pandas as pd

REQUIRED = {"task", "seed", "checkpoint", "success"}


@dataclass(frozen=True)
class PairStats:
    checkpoint_a: str
    checkpoint_b: str
    n_states: int
    n_both_success: int
    n_both_fail: int
    n_a_wins: int
    n_b_wins: int

    @property
    def n_disagree(self) -> int:
        return self.n_a_wins + self.n_b_wins

    @property
    def bidirectional_support(self) -> int:
        return min(self.n_a_wins, self.n_b_wins)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_disagree"] = self.n_disagree
        d["bidirectional_support"] = self.bidirectional_support
        return d


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    d = df[list(REQUIRED)].copy()
    if d.duplicated(["task", "seed", "checkpoint"]).any():
        raise ValueError("duplicate task/seed/checkpoint rows")
    vals = set(pd.to_numeric(d["success"], errors="raise").astype(int).unique())
    if not vals <= {0, 1}:
        raise ValueError("success must be binary 0/1")
    d["success"] = d["success"].astype(int)
    d["checkpoint"] = d["checkpoint"].astype(str)
    return d


def _paired_success(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    sub = df[df.checkpoint.isin([a, b])]
    wide = sub.pivot(index=["task", "seed"], columns="checkpoint", values="success")
    if a not in wide.columns or b not in wide.columns:
        return pd.DataFrame(columns=[a, b])
    return wide[[a, b]].dropna().astype(int)


def pair_stats(df: pd.DataFrame, a: str, b: str) -> PairStats:
    d = validate_panel(df)
    w = _paired_success(d, str(a), str(b))
    return PairStats(
        checkpoint_a=str(a),
        checkpoint_b=str(b),
        n_states=int(len(w)),
        n_both_success=int(((w[a] == 1) & (w[b] == 1)).sum()),
        n_both_fail=int(((w[a] == 0) & (w[b] == 0)).sum()),
        n_a_wins=int(((w[a] == 1) & (w[b] == 0)).sum()),
        n_b_wins=int(((w[a] == 0) & (w[b] == 1)).sum()),
    )


def all_pair_stats(df: pd.DataFrame) -> list[PairStats]:
    d = validate_panel(df)
    checkpoints = sorted(d.checkpoint.unique())
    return [pair_stats(d, a, b) for a, b in combinations(checkpoints, 2)]


def choose_identifiable_pair(df: pd.DataFrame) -> PairStats | None:
    """Choose by two-way support first, never by one-way dominance."""
    stats = all_pair_stats(df)
    if not stats:
        return None
    return max(stats, key=lambda x: (x.bidirectional_support, x.n_disagree, x.n_states))


def task_pair_table(df: pd.DataFrame, a: str, b: str) -> list[dict]:
    d = validate_panel(df)
    rows = []
    for task, g in d.groupby("task", sort=True):
        s = pair_stats(g, a, b)
        row = s.to_dict()
        row["task"] = task
        rows.append(row)
    return rows
