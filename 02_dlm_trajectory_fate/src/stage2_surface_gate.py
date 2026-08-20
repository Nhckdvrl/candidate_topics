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

LOCKED_SPECS = {
    "transient_recovery": {"step": 16, "lead": 4},
    "transient_overwrite": {"step": 4, "lead": 16},
}


def locked_support(labels: dict[str, np.ndarray], capture_steps: np.ndarray) -> list[dict]:
    rows = []
    steps = capture_steps.tolist()
    for task, spec in LOCKED_SPECS.items():
        if spec["step"] not in steps:
            raise ValueError(f"capture steps missing locked step {spec['step']}")
        si = steps.index(spec["step"])
        y = labels[task][:, si]
        lead_key = "recovery_lead" if task == "transient_recovery" else "overwrite_lead"
        lead = labels[lead_key][:, si]
        valid = y >= 0
        keep = valid & ((y == 0) | ((y == 1) & (lead >= spec["lead"])))
        yy = y[keep].astype(int)
        counts = np.bincount(yy, minlength=2) if len(yy) else np.zeros(2, dtype=int)
        rows.append(
            {
                "task": task,
                "step": spec["step"],
                "min_lead": spec["lead"],
                "n": int(len(yy)),
                "positive": int(counts[1]),
                "negative": int(counts[0]),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-positive", type=int, default=6)
    p.add_argument("--min-negative", type=int, default=20)
    args = p.parse_args()

    data = load_shards(Path(args.input_dir), require_hidden=False)
    labels = fate_from_correctness(
        np.asarray(data["correct_strict"]),
        np.asarray(data["capture_steps"]),
        np.asarray(data["observed_strict"]),
    )
    rows = locked_support(labels, np.asarray(data["capture_steps"]))
    for r in rows:
        r["support_ok"] = r["positive"] >= args.min_positive and r["negative"] >= args.min_negative

    passed = sum(bool(r["support_ok"]) for r in rows)
    status = "GO_BOTH" if passed == 2 else "GO_ONE" if passed == 1 else "STOP_LOW_LOCKED_SUPPORT"
    result = {
        "status": status,
        "min_positive": args.min_positive,
        "min_negative": args.min_negative,
        "metadata": data["metadata"],
        "tasks": rows,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "locked_surface_support.csv", index=False)
    (out / "locked_surface_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
