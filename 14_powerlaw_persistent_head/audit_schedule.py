#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from core import (
    PROFILES,
    all_permutations,
    head_overlap,
    key_schedule,
    make_power_batch,
    map_orders,
    mapping_seed_for_seed,
    max_map_run,
    schedule_digests,
)


def batch_digest(x: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(x.tobytes())
    h.update(y.tobytes())
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    p.add_argument("--mapping-seed", type=int, default=1729)
    p.add_argument("--stream-seed", type=int, default=31415)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--seeds", default="0")
    args = p.parse_args()
    pr = PROFILES[args.profile]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    report = {
        "profile": args.profile,
        "core_steps": pr.core_steps,
        "phase_steps": pr.phase_steps,
        "lr_schedule": pr.lr_schedule,
        "seeds": {},
    }
    mapping_seeds = []

    if args.profile == "paper_anchor":
        report["note"] = "paper_anchor is uniform/static only; no slow-fast schedule"
    else:
        slow = key_schedule("slow", pr.core_steps, pr.phase_steps)
        fast = key_schedule("fast", pr.core_steps, pr.phase_steps)
        ds, df = schedule_digests(slow), schedule_digests(fast)
        report.update(
            {
                "slow_max_map_run": max_map_run(slow),
                "fast_max_map_run": max_map_run(fast),
                "slow": ds,
                "fast": df,
                "same_multiset": ds["multiset_digest"] == df["multiset_digest"],
                "different_order": ds["temporal_digest"] != df["temporal_digest"],
            }
        )
        if not report["same_multiset"] or not report["different_order"]:
            raise SystemExit("schedule multiset/order identity failed")
        if report["fast_max_map_run"] != 1 or report["slow_max_map_run"] != pr.phase_steps:
            raise SystemExit("persistence manipulation failed")

    perm = all_permutations()
    for seed in seeds:
        eff = mapping_seed_for_seed(args.mapping_seed, seed)
        mapping_seeds.append(eff)
        ma, mb = map_orders(eff)
        overlap = head_overlap(ma, mb)
        x1, y1 = make_power_batch(seed, "A", 17, 32, args.alpha, eff, args.stream_seed, perm)
        x2, y2 = make_power_batch(seed, "A", 17, 32, args.alpha, eff, args.stream_seed, perm)
        if not np.array_equal(x1, x2) or not np.array_equal(y1, y2):
            raise SystemExit(f"batch regeneration failed for seed {seed}")
        if overlap != 0:
            raise SystemExit(f"A/B top20% heads overlap for seed {seed}: {overlap}")
        report["seeds"][str(seed)] = {
            "effective_mapping_seed": eff,
            "head_overlap_top20pct": overlap,
            "representative_batch_digest": batch_digest(x1, y1),
        }

    if len(mapping_seeds) > 1 and len(set(mapping_seeds)) != len(mapping_seeds):
        raise SystemExit("mapping seeds are not unique across replication seeds")
    report["mapping_seeds_unique"] = len(mapping_seeds) == len(set(mapping_seeds))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
