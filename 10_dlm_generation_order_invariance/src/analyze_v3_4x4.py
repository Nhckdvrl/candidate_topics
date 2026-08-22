from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from schema import read_jsonl
from sudoku4 import Sudoku4Transform


def tau_b(a: dict[str, int], b: dict[str, int]) -> float:
    keys = [k for k in a if k in b]
    concordant = discordant = ties_a = ties_b = 0
    for i, ki in enumerate(keys):
        for kj in keys[i + 1 :]:
            da, db = a[ki] - a[kj], b[ki] - b[kj]
            if da == 0:
                ties_a += 1
            if db == 0:
                ties_b += 1
            elif da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    den_a = concordant + discordant + ties_a
    den_b = concordant + discordant + ties_b
    return (concordant - discordant) / ((den_a * den_b) ** 0.5) if den_a and den_b else float("nan")


def rank_order(indices: list[int], kind: str) -> dict[str, int]:
    if kind == "row-major":
        order = sorted(indices)
    elif kind == "boundary-first":
        order = sorted(indices, key=lambda i: (min(i, 15 - i), i))
    else:
        raise ValueError(kind)
    return {str(cell): rank for rank, cell in enumerate(order, start=1)}


def bootstrap_ci(values: list[float], seed: int = 20260822, n_boot: int = 5000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(mean(draw))
    samples.sort()
    return [samples[int(0.025 * n_boot)], samples[int(0.975 * n_boot)]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="LOCKED_CONFIG_V3.json")
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--manifest", default="data/manifest_v3_4x4.jsonl")
    ap.add_argument("--split", choices=["discovery", "confirmation"], default="discovery")
    ap.add_argument("--out", default="results/g0_v3_4x4_summary.json")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    traces = [r for path in args.traces for r in read_jsonl(path)]
    traces = [r for r in traces if r["split"] == args.split]
    if any(r.get("protocol_version") != cfg["protocol_version"] for r in traces):
        raise RuntimeError("trace protocol mismatch")
    keys = [(r["puzzle_id"], r["variant_id"]) for r in traces]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate trace keys")
    by = defaultdict(dict)
    for r in traces:
        by[r["puzzle_id"]][r["variant_id"]] = r

    identity = [v["identity"] for v in by.values() if "identity" in v]
    iso = [r for v in by.values() for k, r in v.items() if k.startswith("iso-")]
    identity_exact = sum(bool(r["exact_solution"]) for r in identity)
    iso_exact = sum(bool(r["exact_solution"]) for r in iso)
    pairs = []
    repeat_taus = []
    null_rows = []
    for variants in by.values():
        ident = variants.get("identity")
        if ident is None:
            continue
        repeat = variants.get("identity-repeat")
        if repeat is not None:
            repeat_taus.append(tau_b(ident["finalization_step"], repeat["finalization_step"]))
        source_blanks = [int(i) for i in ident["blank_indices"]]
        row_null = rank_order(source_blanks, "row-major")
        boundary_null = rank_order(source_blanks, "boundary-first")
        for variant_id, transformed in variants.items():
            if not variant_id.startswith("iso-"):
                continue
            transform = Sudoku4Transform.from_dict(transformed["transform"])
            mapped = {str(i): transformed["finalization_step"][str(transform.map_index(i))] for i in source_blanks}
            transformed_blanks = [transform.map_index(i) for i in source_blanks]
            row_null_transformed = rank_order(transformed_blanks, "row-major")
            boundary_null_transformed = rank_order(transformed_blanks, "boundary-first")
            row_null_mapped = {str(i): row_null_transformed[str(transform.map_index(i))] for i in source_blanks}
            boundary_null_mapped = {str(i): boundary_null_transformed[str(transform.map_index(i))] for i in source_blanks}
            pairs.append({
                "puzzle_id": transformed["puzzle_id"],
                "variant_id": variant_id,
                "identity_exact": bool(ident["exact_solution"]),
                "isomorph_exact": bool(transformed["exact_solution"]),
                "tau_iso": tau_b(ident["finalization_step"], mapped) if ident["exact_solution"] and transformed["exact_solution"] else None,
                "tau_row_major_null": tau_b(row_null, row_null_mapped),
                "tau_boundary_first_null": tau_b(boundary_null, boundary_null_mapped),
            })

    both = [p for p in pairs if p["tau_iso"] is not None]
    by_puzzle_pairs = defaultdict(list)
    for p in pairs:
        by_puzzle_pairs[p["puzzle_id"]].append(p)
    cluster_flip = [mean(float(p["identity_exact"] != p["isomorph_exact"]) for p in ps) for ps in by_puzzle_pairs.values()]
    cluster_both = [
        mean(p["tau_iso"] for p in ps if p["tau_iso"] is not None)
        for ps in by_puzzle_pairs.values()
        if any(p["tau_iso"] is not None for p in ps)
    ]
    cluster_row_null = [
        mean(p["tau_row_major_null"] for p in ps if p["tau_iso"] is not None)
        for ps in by_puzzle_pairs.values()
        if any(p["tau_iso"] is not None for p in ps)
    ]
    cluster_boundary_null = [
        mean(p["tau_boundary_first_null"] for p in ps if p["tau_iso"] is not None)
        for ps in by_puzzle_pairs.values()
        if any(p["tau_iso"] is not None for p in ps)
    ]
    cluster_excess_row = [a - b for a, b in zip(cluster_both, cluster_row_null)]
    cluster_excess_boundary = [a - b for a, b in zip(cluster_both, cluster_boundary_null)]
    summary = {
        "protocol_version": cfg["protocol_version"],
        "split": args.split,
        "n_puzzles": len(by),
        "n_identity_exact": identity_exact,
        "identity_exact_accuracy": identity_exact / len(identity) if identity else None,
        "n_isomorph_traces": len(iso),
        "n_isomorph_exact": iso_exact,
        "isomorph_exact_accuracy": iso_exact / len(iso) if iso else None,
        "solve_flip_count": sum(p["identity_exact"] != p["isomorph_exact"] for p in pairs),
        "solve_flip_rate": sum(p["identity_exact"] != p["isomorph_exact"] for p in pairs) / len(pairs) if pairs else None,
        "solve_flip_rate_puzzle_cluster_bootstrap_95ci": bootstrap_ci(cluster_flip),
        "n_both_exact_pairs": len(both),
        "n_both_exact_puzzles": len({p["puzzle_id"] for p in both}),
        "same_serialization_repeat_tau": {
            "n": len(repeat_taus), "mean": mean(repeat_taus) if repeat_taus else None,
            "min": min(repeat_taus) if repeat_taus else None,
        },
        "native_scheduler_pick_same_fraction": {
            "mean": mean(r["metadata"]["native_scheduler_pick_same_fraction"] for r in traces) if traces else None,
            "min": min(r["metadata"]["native_scheduler_pick_same_fraction"] for r in traces) if traces else None,
        },
        "tau_iso_mean_both_exact": mean(p["tau_iso"] for p in both) if both else None,
        "tau_iso_puzzle_cluster_mean": mean(cluster_both) if cluster_both else None,
        "tau_iso_puzzle_cluster_bootstrap_95ci": bootstrap_ci(cluster_both),
        "tau_row_major_null_puzzle_cluster_mean": mean(cluster_row_null) if cluster_row_null else None,
        "tau_boundary_first_null_puzzle_cluster_mean": mean(cluster_boundary_null) if cluster_boundary_null else None,
        "tau_excess_over_row_major_puzzle_cluster_mean": mean(cluster_excess_row) if cluster_excess_row else None,
        "tau_excess_over_row_major_puzzle_cluster_bootstrap_95ci": bootstrap_ci(cluster_excess_row),
        "tau_excess_over_boundary_first_puzzle_cluster_mean": mean(cluster_excess_boundary) if cluster_excess_boundary else None,
        "tau_excess_over_boundary_first_puzzle_cluster_bootstrap_95ci": bootstrap_ci(cluster_excess_boundary),
        "tau_iso_per_puzzle": pairs,
        "interpretation_note": "4x4 published-setting G0; solve flips are outcome non-equivariance, and tau is conditional on both exact.",
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
