#!/usr/bin/env python3
"""Optional learning-curve plot; skipped cleanly when matplotlib is unavailable."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

CONDITIONS = ["uniform", "static", "balanced_slow", "balanced_fast"]


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping optional plot")
        return

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("outputs/pilot"))
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    series: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in CONDITIONS}
    for seed_dir in sorted(args.root.glob("seed*")):
        for cond in CONDITIONS:
            path = seed_dir / cond / "metrics.csv"
            if not path.exists():
                continue
            with path.open() as f:
                rows = list(csv.DictReader(f))
            x = np.asarray([float(r["step"]) for r in rows])
            y = np.asarray([float(r["token_accuracy"]) for r in rows])
            series[cond].append((x, y))

    plt.figure(figsize=(8, 5))
    for cond in CONDITIONS:
        runs = series[cond]
        if not runs:
            continue
        common_x = runs[0][0]
        if not all(np.array_equal(x, common_x) for x, _ in runs):
            raise SystemExit(f"step grids differ across seeds for {cond}")
        ys = np.stack([y for _, y in runs])
        mean = ys.mean(axis=0)
        plt.plot(common_x, mean, label=cond)
        if len(runs) > 1:
            plt.fill_between(common_x, ys.min(axis=0), ys.max(axis=0), alpha=0.12)
    plt.axhline(0.2, linestyle="--", linewidth=1, label="chance token accuracy")
    plt.xlabel("optimizer step")
    plt.ylabel("uniform-test token accuracy")
    plt.title("Topic 14: power-law head persistence")
    plt.legend()
    plt.tight_layout()
    out = args.output or (args.root / "learning_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
