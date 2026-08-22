"""G0: test whether same-family checkpoints have enough natural two-way crossover."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .panel import all_pair_stats, choose_identifiable_pair, task_pair_table, validate_panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-bidirectional", type=int, default=15)
    args = p.parse_args()

    df = validate_panel(pd.read_csv(args.csv))
    stats = all_pair_stats(df)
    chosen = choose_identifiable_pair(df)

    report = {
        "n_rows": int(len(df)),
        "n_states": int(df[["task", "seed"]].drop_duplicates().shape[0]),
        "checkpoints": sorted(df.checkpoint.unique().tolist()),
        "pairs": [s.to_dict() for s in stats],
        "min_bidirectional_required": int(args.min_bidirectional),
    }

    if chosen is None:
        report["selected_pair"] = None
        report["verdict"] = "STOP_NO_NATURAL_CROSSOVER"
    else:
        report["selected_pair"] = chosen.to_dict()
        report["selected_pair_by_task"] = task_pair_table(
            df, chosen.checkpoint_a, chosen.checkpoint_b
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
