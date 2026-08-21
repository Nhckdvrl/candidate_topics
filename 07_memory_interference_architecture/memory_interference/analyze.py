from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import bootstrap_model_gap, summarize


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_rows(run_dir / "results.jsonl")
    summary = summarize(rows)

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "num_updates", "ri_accuracy", "pi_accuracy", "I", "n_ri", "n_pi"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary)

    intrusion = defaultdict(list)
    for r in rows:
        if r.get("skipped", False) or r.get("correct", False):
            continue
        intrusion[(r["model"], r["condition"])].append(float(r["predicted_position"]))
    intrusion_summary = [
        {
            "model": m,
            "condition": c,
            "n_errors": len(vals),
            "mean_predicted_position": float(np.mean(vals)) if vals else float("nan"),
            "median_predicted_position": float(np.median(vals)) if vals else float("nan"),
        }
        for (m, c), vals in sorted(intrusion.items())
    ]
    with (run_dir / "intrusions.json").open("w", encoding="utf-8") as f:
        json.dump(intrusion_summary, f, indent=2)

    token_audit = []
    for model in sorted({r["model"] for r in rows}):
        for condition in ("RI", "PI"):
            subset = [
                r
                for r in rows
                if r["model"] == model
                and r["condition"] == condition
                and not r.get("skipped", False)
            ]
            target_lens = []
            boundary_shifts = []
            prompt_lens = []
            for r in subset:
                prompt_lens.append(int(r["prompt_tokens"]))
                for score in r.get("scores", []):
                    boundary_shifts.append(int(score.get("boundary_shift", 0)))
                    if score["candidate"] == r["target"]:
                        target_lens.append(int(score["token_count"]))
                        break
            token_audit.append(
                {
                    "model": model,
                    "condition": condition,
                    "n": len(subset),
                    "mean_target_token_count": float(np.mean(target_lens)) if target_lens else float("nan"),
                    "max_boundary_shift": max(boundary_shifts) if boundary_shifts else None,
                    "max_prompt_tokens": max(prompt_lens) if prompt_lens else None,
                }
            )
    with (run_dir / "token_audit.json").open("w", encoding="utf-8") as f:
        json.dump(token_audit, f, indent=2)

    models = sorted({r["model"] for r in rows})
    pairwise = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pairwise.append(
                {"model_a": a, "model_b": b, **bootstrap_model_gap(rows, a, b, n_boot=args.bootstrap)}
            )
    with (run_dir / "pairwise_bootstrap.json").open("w", encoding="utf-8") as f:
        json.dump(pairwise, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {run_dir/'summary.csv'} and diagnostics")


if __name__ == "__main__":
    main()
