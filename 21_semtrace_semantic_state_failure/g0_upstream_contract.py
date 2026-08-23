#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Official fsyn_output_prediction summary.json")
    ap.add_argument("--out", default="artifacts/g0_upstream_contract.json")
    args = ap.parse_args()

    data = json.loads(Path(args.summary).read_text())
    points = sorted((int(k), float(v)) for k, v in data.items())
    if len(points) < 3:
        raise SystemExit("Need at least three target positions in the official summary")

    first_pos, first_acc = points[0]
    last_pos, last_acc = points[-1]
    midpoint = (first_pos + last_pos) / 2
    mid_pos, mid_acc = min(points, key=lambda x: abs(x[0] - midpoint))
    edge_mean = (first_acc + last_acc) / 2
    drop = edge_mean - mid_acc

    gate = {
        "at_least_3_positions": len(points) >= 3,
        "edge_mean_acc_ge_0.30": edge_mean >= 0.30,
        "edge_to_middle_drop_ge_0.20": drop >= 0.20,
    }
    result = {
        "summary_path": str(Path(args.summary).resolve()),
        "positions": points,
        "first": {"position": first_pos, "accuracy": first_acc},
        "middle": {"position": mid_pos, "accuracy": mid_acc},
        "last": {"position": last_pos, "accuracy": last_acc},
        "edge_mean_accuracy": edge_mean,
        "edge_to_middle_drop": drop,
        "gate": gate,
        "verdict": "UPSTREAM_SEED_REPRODUCED" if all(gate.values()) else "UPSTREAM_SEED_NOT_REPRODUCED",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["verdict"] != "UPSTREAM_SEED_REPRODUCED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
