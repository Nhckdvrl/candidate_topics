"""The single figure that summarises why Topic 08 was archived."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

NEAR_PX = 20.0
OUTCOME = "branch_final_kp_dispersion_px"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True)
    p.add_argument("--null-glob", default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(args.glob))], ignore_index=True)
    near = df.agent_block_gap_px < NEAR_PX

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    ax.scatter(df.ace[~near], df[OUTCOME][~near], s=12, alpha=0.45, color="#6699cc",
               label=f"far from block (n={(~near).sum()})")
    ax.scatter(df.ace[near], df[OUTCOME][near], s=12, alpha=0.5, color="#cc5544",
               label=f"near block (n={near.sum()})")
    ax.set_xlabel("FIPER ACE (bits)")
    ax.set_ylabel("episode-level branch outcome dispersion (px)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("action entropy vs true functional uncertainty")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for mask, name, col in ((slice(None), "all states", "#33475b"),
                            (near, "near block", "#cc5544"),
                            (~near, "far from block", "#6699cc")):
        sub = df[mask] if not isinstance(mask, slice) else df
        o = sub[OUTCOME].to_numpy(float)
        y = o >= np.quantile(o, 0.75)
        if y.min() == y.max():
            continue
        fpr, tpr, _ = roc_curve(y, sub.ace.to_numpy(float))
        auc = roc_auc_score(y, sub.ace.to_numpy(float))
        ax.plot(fpr, tpr, color=col, label=f"{name}: AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ranking states by true functional uncertainty")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[2]
    if args.null_glob and sorted(glob.glob(args.null_glob)):
        nulldf = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(args.null_glob))],
                           ignore_index=True)
        nnear = nulldf.agent_block_gap_px < NEAR_PX
        data = [df[OUTCOME][near], nulldf[OUTCOME][nnear],
                df[OUTCOME][~near], nulldf[OUTCOME][~nnear]]
        labels = ["near\n(different\nchunks)", "near\n(SAME\nchunk)",
                  "far\n(different\nchunks)", "far\n(SAME\nchunk)"]
        ax.boxplot([d.dropna() for d in data], tick_labels=labels, showfliers=False)
        ax.set_ylabel("branch outcome dispersion (px)")
        ax.set_title("how much dispersion is the action choice?\n(null control: all branches share one chunk)")
    else:
        ax.text(0.5, 0.5, "null control pending", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
