from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def binary_entropy_from_margin(margin):
    p = np.clip(sigmoid(margin), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def summarize(path: Path):
    d = np.load(path, allow_pickle=False)
    true_margin = d["output_true_viable_margin"].astype(np.float64)
    p_true = sigmoid(true_margin)
    wrong = true_margin < 0
    strong_wrong = true_margin < -2.0
    return {
        "tag": str(d["tag"].item()),
        "condition": str(d["condition"].item()),
        "n": len(true_margin),
        "output_choice_acc": float((true_margin > 0).mean()),
        "mean_p_true_viable_pair": float(p_true.mean()),
        "mean_pair_entropy": float(binary_entropy_from_margin(true_margin).mean()),
        "mean_abs_margin": float(np.abs(true_margin).mean()),
        "wrong_commit_rate": float(wrong.mean()),
        "strong_wrong_commit_rate": float(strong_wrong.mean()),
        "mean_wrong_abs_margin": float((-true_margin[wrong]).mean()) if wrong.any() else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="Cheap teacher-forced first-fork debugging preflight.")
    p.add_argument("--input-dir", default="artifacts/preflight_states")
    p.add_argument("--tags", default="e01,e02,e04,e16")
    p.add_argument("--output", default="artifacts/state_preflight.csv")
    args = p.parse_args()

    rows = []
    for tag in [x.strip() for x in args.tags.split(",") if x.strip()]:
        path = Path(args.input_dir) / f"{tag}_original.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(summarize(path))
    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    meta = {
        "status": "sampling_required",
        "reason": "teacher-forced margins cannot establish pass@k coverage shrinkage",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
