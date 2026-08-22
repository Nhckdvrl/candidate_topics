#!/usr/bin/env python3
"""Validity diagnostic for a NEGATIVE G0 part-A result.

If the fixed token-mean summary were blind to the WAM adapters, a null
future-predictability gain would be an artifact of the summary rather than a
statement about the adapters. This script measures how much the adapters move
the exact feature the probe consumes.

It adds no fitting, no search and no alternative pooling; it only reports the
magnitude of `feature_normal - feature_bypass`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tensors", type=Path, required=True, help="g0_tensors.pt from a --save-tensors run")
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    payload = torch.load(args.tensors, weights_only=False)
    report: dict[str, dict[str, float]] = {}
    for split in ("train", "test"):
        normal = payload[split]["feature_normal"].float()
        bypass = payload[split]["feature_bypass"].float()
        delta = normal - bypass
        width = normal.shape[1] // args.num_layers
        entry = {
            "num_samples": int(normal.shape[0]),
            "feature_dim": int(normal.shape[1]),
            "rms_normal": float(normal.pow(2).mean().sqrt()),
            "rms_delta": float(delta.pow(2).mean().sqrt()),
            "relative_delta_norm": float(delta.norm(dim=1).mean() / normal.norm(dim=1).mean()),
            "across_sample_var_normal": float(normal.var(dim=0).sum()),
            "across_sample_var_delta": float(delta.var(dim=0).sum()),
        }
        for i in range(args.num_layers):
            sl = slice(i * width, (i + 1) * width)
            entry[f"relative_delta_norm_layer{i}"] = float(
                delta[:, sl].norm(dim=1).mean() / normal[:, sl].norm(dim=1).mean()
            )
        report[split] = entry

    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
