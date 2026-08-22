#!/usr/bin/env python3
"""Audit the temporal intervention without training a model."""
from __future__ import annotations

import argparse
import json

from experiment import HOPS, N_SKILLS, block_stream_seed, largest_remainder_counts, make_base_order, make_shift_schedule, rank_to_skill_for_shift, schedule_audit, zipf_prob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--mapping-seed", type=int, default=1729)
    p.add_argument("--schedule-seed", type=int, default=2718)
    p.add_argument("--block-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    base = make_base_order(args.mapping_seed)
    reports = {}
    shifts_by_condition = {}
    for condition in ["static", "balanced_slow", "balanced_fast"]:
        shifts = make_shift_schedule(condition, args.cycles, args.schedule_seed)
        shifts_by_condition[condition] = shifts
        reports[condition] = schedule_audit(condition, shifts, base, args.alpha)

    # Verify exact realized counts using the same integer rank histogram used by training.
    rank_counts = largest_remainder_counts(zipf_prob(args.alpha), args.block_steps * args.batch_size * HOPS)
    realized = {}
    for condition in ["balanced_slow", "balanced_fast"]:
        counts = [0] * N_SKILLS
        for shift in shifts_by_condition[condition]:
            r2s = rank_to_skill_for_shift(base, int(shift))
            for rank, skill in enumerate(r2s):
                counts[int(skill)] += int(rank_counts[rank])
        realized[condition] = {
            "min": min(counts),
            "max": max(counts),
            "all_equal": min(counts) == max(counts),
            "total": sum(counts),
        }
    reports["realized_counts"] = realized

    same_block_multiset = True
    for cycle in range(args.cycles):
        lo, hi = cycle * N_SKILLS, (cycle + 1) * N_SKILLS
        slow_keys = sorted(
            (int(shift), block_stream_seed("balanced_slow", i, int(shift), 0, 31415))
            for i, shift in enumerate(shifts_by_condition["balanced_slow"][lo:hi], start=lo)
        )
        fast_keys = sorted(
            (int(shift), block_stream_seed("balanced_fast", i, int(shift), 0, 31415))
            for i, shift in enumerate(shifts_by_condition["balanced_fast"][lo:hi], start=lo)
        )
        same_block_multiset = same_block_multiset and (slow_keys == fast_keys)
    reports["same_balanced_block_multiset"] = same_block_multiset

    slow = reports["balanced_slow"]
    fast = reports["balanced_fast"]
    checks = {
        "slow_exact_rank_occupancy": slow["occupancy_is_exactly_balanced"],
        "fast_exact_rank_occupancy": fast["occupancy_is_exactly_balanced"],
        "slow_exact_realized_counts": realized["balanced_slow"]["all_equal"],
        "fast_exact_realized_counts": realized["balanced_fast"]["all_equal"],
        "slow_fast_same_total_positions": realized["balanced_slow"]["total"] == realized["balanced_fast"]["total"],
        "slow_fast_same_training_block_multiset": same_block_multiset,
        "slow_more_persistent_lag1": slow["lag1_log_weight_corr"] > fast["lag1_log_weight_corr"] + 0.25,
        "slow_longer_head_runs": slow["mean_max_head_run_blocks_per_skill"] > 2 * fast["mean_max_head_run_blocks_per_skill"],
    }
    reports["checks"] = checks
    print(json.dumps(reports, indent=2))
    if not all(bool(v) for v in checks.values()):
        raise SystemExit("schedule audit FAILED")


if __name__ == "__main__":
    main()
