#!/usr/bin/env python3
"""Minimal G0 scorer for Topic 16: citation transmutation.

This script deliberately does *not* infer claim equivalence, evidence status, or
certainty with an LLM. Those are the scientific measurements and must be
validated separately. The script freezes the row schema and primary statistic.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source_claim: str
    citing_claim: str
    same_claim: bool
    new_primary_evidence: bool
    source_certainty: float
    citing_certainty: float

    @property
    def delta(self) -> float:
        return self.citing_certainty - self.source_certainty


def _as_bool(value, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a JSON Boolean, got {value!r}")


def load_edges(path: Path) -> list[Edge]:
    rows: list[Edge] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                edge = Edge(
                    edge_id=str(obj["edge_id"]),
                    source_claim=str(obj["source_claim"]),
                    citing_claim=str(obj["citing_claim"]),
                    same_claim=_as_bool(obj["same_claim"], "same_claim"),
                    new_primary_evidence=_as_bool(
                        obj["new_primary_evidence"], "new_primary_evidence"
                    ),
                    source_certainty=float(obj["source_certainty"]),
                    citing_certainty=float(obj["citing_certainty"]),
                )
            except KeyError as exc:
                raise ValueError(f"line {lineno}: missing field {exc}") from exc
            if not 0.0 <= edge.source_certainty <= 1.0:
                raise ValueError(f"line {lineno}: source_certainty outside [0,1]")
            if not 0.0 <= edge.citing_certainty <= 1.0:
                raise ValueError(f"line {lineno}: citing_certainty outside [0,1]")
            rows.append(edge)
    if not rows:
        raise ValueError("input contains no rows")
    return rows


def bootstrap_mean_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return lo, hi


def summarize_group(edges: Iterable[Edge], n_boot: int, seed: int) -> dict:
    edges = list(edges)
    deltas = [e.delta for e in edges]
    if not deltas:
        return {
            "n": 0,
            "mean_delta": None,
            "median_delta": None,
            "upward_fraction": None,
            "bootstrap_95ci": None,
        }
    lo, hi = bootstrap_mean_ci(deltas, n_boot=n_boot, seed=seed)
    return {
        "n": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "upward_fraction": sum(d > 0 for d in deltas) / len(deltas),
        "bootstrap_95ci": [lo, hi],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args()

    edges = load_edges(args.input)
    same_claim = [e for e in edges if e.same_claim]
    no_new = [e for e in same_claim if not e.new_primary_evidence]
    with_new = [e for e in same_claim if e.new_primary_evidence]

    result = {
        "all_rows": len(edges),
        "same_claim_rows": len(same_claim),
        "primary_no_new_evidence": summarize_group(
            no_new, n_boot=args.bootstrap, seed=args.seed
        ),
        "secondary_with_new_evidence": summarize_group(
            with_new, n_boot=args.bootstrap, seed=args.seed + 1
        ),
    }

    if no_new and with_new:
        result["difference_in_mean_delta_no_new_minus_with_new"] = (
            statistics.fmean(e.delta for e in no_new)
            - statistics.fmean(e.delta for e in with_new)
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
