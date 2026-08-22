"""G0: robust natural crossover using repeated common-noise rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .panel import all_pair_stats, choose_identifiable_pair, task_pair_table, validate_panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-trials", type=int, default=8)
    p.add_argument("--rate-gap", type=float, default=0.50)
    p.add_argument("--min-bidirectional", type=int, default=15)
    args = p.parse_args()

    df = pd.concat([pd.read_csv(x) for x in args.csv], ignore_index=True)
    df = validate_panel(df, min_trials=args.min_trials)
    stats = all_pair_stats(df, min_trials=args.min_trials, rate_gap=args.rate_gap)
    chosen = choose_identifiable_pair(df, min_trials=args.min_trials, rate_gap=args.rate_gap)

    report = {
        "inputs": [str(x) for x in args.csv],
        "n_rollout_rows": int(len(df)),
        "n_physical_states": int(df.state_id.nunique()),
        "checkpoints": sorted(df.checkpoint.unique().tolist()),
        "min_trials_per_checkpoint_state": int(args.min_trials),
        "robust_winner_rate_gap": float(args.rate_gap),
        "min_bidirectional_required": int(args.min_bidirectional),
        "pairs": [s.to_dict() for s in stats],
    }
    if chosen is None:
        report["selected_pair"] = None
        report["verdict"] = "STOP_NO_NATURAL_CROSSOVER"
    else:
        report["selected_pair"] = chosen.to_dict()
        report["selected_pair_by_task"] = task_pair_table(
            df,
            chosen.checkpoint_a,
            chosen.checkpoint_b,
            min_trials=args.min_trials,
            rate_gap=args.rate_gap,
        )
        report["verdict"] = (
            "G0_PASS_FREEZE_PAIR"
            if chosen.bidirectional_support >= args.min_bidirectional
            else "STOP_NO_NATURAL_CROSSOVER"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
