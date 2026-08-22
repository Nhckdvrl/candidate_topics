"""Progress diagnostic for a partially collected behavior panel.

This is **not** the G0 gate. `analyze_disagreement.py` is, and it runs once on the complete
panel. This script exists so a run that is clearly heading for
`STOP_NO_NATURAL_CROSSOVER` can be recognised before it burns another eight GPU-hours.

It can only ever motivate stopping early, never a pass, so it cannot turn into a
data-dependent route to a positive result. It reports on the subset of physical states that
are already complete -- every checkpoint present with the full seed set -- because a partial
state has a biased success rate: slow rollouts are exactly the failing ones, which run to
the step limit instead of terminating early.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from .panel import make_state_id, pair_stats


def complete_states(df: pd.DataFrame, *, n_seeds: int) -> pd.DataFrame:
    d = df.copy()
    d["state_id"] = make_state_id(d)
    checkpoints = sorted(d.checkpoint.astype(str).unique())
    counts = (
        d.groupby(["state_id", "checkpoint"], as_index=False)
        .policy_seed.nunique()
        .rename(columns={"policy_seed": "n_seeds"})
    )
    wide = counts.pivot(index="state_id", columns="checkpoint", values="n_seeds").fillna(0)
    missing = [c for c in checkpoints if c not in wide.columns]
    for c in missing:
        wide[c] = 0
    done = wide.index[(wide[checkpoints] >= n_seeds).all(axis=1)]
    return d[d.state_id.isin(done)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--rate-gap", type=float, default=0.50)
    args = p.parse_args()

    frames = []
    for x in args.csv:
        try:
            frames.append(pd.read_csv(x))
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
    if not frames:
        print("no readable panel files yet")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df[df.get("status", "ok").astype(str) == "ok"]

    print(f"rollout rows: {len(df)}")
    agg = df.groupby("checkpoint").success.agg(["sum", "size"])
    for cp, r in agg.iterrows():
        print(f"  {cp}: {int(r['sum'])}/{int(r['size'])} = {r['sum'] / r['size']:.3f}")

    full = complete_states(df, n_seeds=args.n_seeds)
    n_full = full.state_id.nunique() if len(full) else 0
    print(f"\nfully complete physical states: {n_full} / 150")
    if n_full < 2:
        print("not enough complete states to look at crossover yet")
        return

    print(f"\ncrossover on complete states only (gate needs >=15 each way on all 150):")
    for a, b in itertools.combinations(sorted(full.checkpoint.astype(str).unique()), 2):
        try:
            s = pair_stats(full, a, b, min_trials=args.n_seeds, rate_gap=args.rate_gap)
        except ValueError as e:
            print(f"  {a} vs {b}: {e}")
            continue
        print(
            f"  {a} vs {b}: {a}-wins={s.n_a_wins:3d}  {b}-wins={s.n_b_wins:3d}  "
            f"ambiguous={s.n_ambiguous:3d}  bidirectional={s.bidirectional_support:3d}"
        )


if __name__ == "__main__":
    main()
