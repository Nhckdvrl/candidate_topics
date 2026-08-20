from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from fate_labels import fate_from_correctness
from io_utils import load_shards


TASKS = (
    "recover_any",
    "overwrite_any",
    "transient_recovery",
    "transient_overwrite",
    "finish_correct_from_wrong",
    "finish_wrong_from_correct",
)
NOVEL_TASKS = {"transient_recovery", "transient_overwrite"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize surface-only trajectory class support before hidden extraction."
    )
    p.add_argument("--input-dir", default="artifacts/preflight/raw")
    p.add_argument("--output-dir", default="artifacts/preflight")
    p.add_argument("--label-mode", choices=["strict", "fallback"], default="strict")
    p.add_argument(
        "--min-novel-class",
        type=int,
        default=10,
        help="Preflight gate on a small run; 10/200 roughly targets ~50/1000.",
    )
    return p.parse_args()


def _task_target(labels: dict[str, np.ndarray], task: str, si: int):
    mapping = {
        "recover_any": "recoverable",
        "overwrite_any": "will_overwrite",
        "transient_recovery": "transient_recovery",
        "transient_overwrite": "transient_overwrite",
        "finish_correct_from_wrong": "finish_correct_from_wrong",
        "finish_wrong_from_correct": "finish_wrong_from_correct",
    }
    y = labels[mapping[task]][:, si]
    return y, y >= 0


def main() -> None:
    args = parse_args()
    data = load_shards(Path(args.input_dir), require_hidden=False)

    if args.label_mode == "strict":
        correct = np.asarray(data["correct_strict"])
        observed = np.asarray(data["observed_strict"])
    else:
        correct = np.asarray(data["correct_fallback"])
        observed = np.asarray(data["observed_fallback"])

    labels = fate_from_correctness(
        correct,
        np.asarray(data["capture_steps"]),
        observed,
    )

    rows = []
    for si, step in enumerate(np.asarray(data["capture_steps"]).tolist()):
        for task in TASKS:
            target, valid = _task_target(labels, task, si)
            y = target[valid].astype(int)
            counts = np.bincount(y, minlength=2) if len(y) else np.zeros(2, dtype=int)
            rows.append(
                {
                    "task": task,
                    "step": int(step),
                    "n": int(len(y)),
                    "positive": int(counts[1]),
                    "negative": int(counts[0]),
                    "min_class": int(counts.min()) if len(y) else 0,
                }
            )

    df = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "surface_class_counts.csv", index=False)

    final_observed = observed[:, -1]
    final_correct = correct[:, -1]
    answer_observed_rate = float(final_observed.mean())
    final_accuracy_all = float((final_observed & final_correct).mean())
    final_accuracy_observed = (
        float(final_correct[final_observed].mean())
        if final_observed.any()
        else float("nan")
    )

    novel = df[df["task"].isin(sorted(NOVEL_TASKS))]
    max_novel_support = int(novel["min_class"].max()) if len(novel) else 0
    ready = max_novel_support >= args.min_novel_class

    summary = {
        "status": "GO_HIDDEN_G0" if ready else "STOP_OR_CHANGE_GEOMETRY",
        "label_mode": args.label_mode,
        "final_answer_observed_rate": answer_observed_rate,
        "final_accuracy_all": final_accuracy_all,
        "final_accuracy_among_observed": final_accuracy_observed,
        "max_novel_min_class_support": max_novel_support,
        "min_novel_class_gate": int(args.min_novel_class),
        "metadata": data["metadata"],
    }
    (out_dir / "surface_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
