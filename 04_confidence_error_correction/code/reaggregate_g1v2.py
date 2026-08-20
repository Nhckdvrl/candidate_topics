#!/usr/bin/env python3
"""Recompute G-1v2 semantic measurements from saved G-1v1 permutation_probs.

No model inference is required. This script is the first action after the v1
measurement failure: it asks whether the already-saved 10x10 mapped
distributions support a cleaner log-space semantic measurement before spending
any more GPU time.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mcq_utils import (
    arithmetic_mean_distribution,
    geometric_mean_distribution,
    permutation_susceptibility,
    semantic_metrics,
)


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows_out = []
    for row in read_jsonl(args.input):
        mapped = row.get("permutation_probs")
        if not mapped:
            raise ValueError(f"{row.get('id')}: missing permutation_probs")
        k = len(mapped[0])
        if len(mapped) != k:
            raise ValueError(
                f"{row.get('id')}: expected a complete KxK balanced set, got "
                f"{len(mapped)} permutations for K={k}"
            )
        debiased = geometric_mean_distribution(mapped)
        arithmetic = arithmetic_mean_distribution(mapped)
        a = int(row["answer"])
        metrics = semantic_metrics(debiased, a)
        old_metrics = semantic_metrics(arithmetic, a)
        top_wrong_by_perm = [
            max((j for j in range(k) if j != a), key=lambda j: p[j]) for p in mapped
        ]
        modal_wrong, count = Counter(top_wrong_by_perm).most_common(1)[0]

        out = {
            **row,
            "semantic_probs_v1_arithmetic": arithmetic,
            "v1_arithmetic_p_correct": old_metrics["p_correct"],
            "v1_arithmetic_wrong_concentration": old_metrics["wrong_concentration"],
            "v1_arithmetic_top_wrong": old_metrics["top_wrong"],
            "semantic_probs": debiased,
            "semantic_probs_debiased": debiased,
            **metrics,
            "position_susceptibility_js": permutation_susceptibility(mapped, debiased),
            "modal_top_wrong": int(modal_wrong),
            "top_wrong_stability": count / len(top_wrong_by_perm),
            "top_wrong_by_perm": top_wrong_by_perm,
            "measurement_version": "g1v2_logmean_reaggregated",
        }
        rows_out.append(out)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"reaggregated {len(rows_out)} items -> {args.output}")


if __name__ == "__main__":
    main()
