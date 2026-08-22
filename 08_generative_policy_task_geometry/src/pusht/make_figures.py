"""Figures for E1. One scatter is the whole argument, so it should be readable."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import binned_spread


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--score", default="ace")
    p.add_argument("--outcome", default="outcome_kp_dispersion_px")
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    contact = df.frac_samples_with_contact > 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.scatter(df[args.score][~contact], df[args.outcome][~contact], s=14, alpha=0.55,
               label="pusher mostly not in contact", color="#6699cc")
    ax.scatter(df[args.score][contact], df[args.outcome][contact], s=14, alpha=0.55,
               label="pusher in contact", color="#cc5544")
    ax.set_xlabel("FIPER ACE (bits)")
    ax.set_ylabel("measured outcome dispersion (px)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("scalar action entropy vs true task-outcome dispersion")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    bs = binned_spread(df, args.score, args.outcome, n_bins=8)
    if len(bs):
        x = np.arange(len(bs))
        ax.fill_between(x, bs.outcome_p10, bs.outcome_p90, alpha=0.3, color="#6699cc",
                        label="p10-p90 of outcome")
        ax.plot(x, bs.outcome_p50, "o-", color="#33475b", label="median outcome")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(bs.score_lo, bs.score_hi)],
                           rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("ACE bin (bits)")
    ax.set_ylabel("outcome dispersion (px)")
    ax.set_title("outcome spread remaining inside narrow ACE bins")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    ax.scatter(df.act_rms_dispersion, df[args.outcome], s=14, alpha=0.55, color="#7a9a6b")
    ax.set_xlabel("action-space RMS dispersion (px)")
    ax.set_ylabel("outcome dispersion (px)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("estimator-free action dispersion vs outcome")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
