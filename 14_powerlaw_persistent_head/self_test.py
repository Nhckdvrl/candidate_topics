#!/usr/bin/env python3
"""Cheap deterministic tests for the task algebra and causal schedule construction."""
from __future__ import annotations

import numpy as np

from experiment import (
    HOPS,
    N_SKILLS,
    all_permutations,
    block_stream_seed,
    compose_numpy,
    exact_rank_block,
    largest_remainder_counts,
    make_base_order,
    make_shift_schedule,
    rank_to_skill_for_shift,
    schedule_audit,
    zipf_prob,
)


def test_permutation_algebra() -> None:
    table = all_permutations()
    assert table.shape == (120, 5)
    assert len({tuple(x) for x in table.tolist()}) == 120
    assert np.all(np.sort(table, axis=1) == np.arange(5))

    ident = np.arange(5, dtype=np.int64)
    rng = np.random.default_rng(7)
    for _ in range(100):
        a, b, c = table[rng.integers(0, len(table), size=3)]
        # compose_numpy([a,b]) implements a∘b under the paper's sigma[pi] notation.
        ab = compose_numpy(np.asarray([[a, b]], dtype=np.int64))[0]
        lhs = compose_numpy(np.asarray([[ab, c]], dtype=np.int64))[0]
        bc = compose_numpy(np.asarray([[b, c]], dtype=np.int64))[0]
        rhs = compose_numpy(np.asarray([[a, bc]], dtype=np.int64))[0]
        assert np.array_equal(lhs, rhs)
        assert np.array_equal(compose_numpy(np.asarray([[ident, a]], dtype=np.int64))[0], a)
        assert np.array_equal(compose_numpy(np.asarray([[a, ident]], dtype=np.int64))[0], a)


def test_balanced_schedules() -> None:
    cycles = 2
    base = make_base_order(1729)
    slow_shifts = make_shift_schedule("balanced_slow", cycles, 2718)
    fast_shifts = make_shift_schedule("balanced_fast", cycles, 2718)

    for shifts in (slow_shifts, fast_shifts):
        assert len(shifts) == cycles * N_SKILLS
        for c in range(cycles):
            part = np.sort(shifts[c * N_SKILLS : (c + 1) * N_SKILLS])
            assert np.array_equal(part, np.arange(N_SKILLS))

    slow = schedule_audit("balanced_slow", slow_shifts, base, 1.5)
    fast = schedule_audit("balanced_fast", fast_shifts, base, 1.5)
    assert slow["occupancy_is_exactly_balanced"]
    assert fast["occupancy_is_exactly_balanced"]
    assert slow["lag1_log_weight_corr"] > fast["lag1_log_weight_corr"] + 0.25
    assert slow["mean_max_head_run_blocks_per_skill"] > 2 * fast["mean_max_head_run_blocks_per_skill"]

    # Exact realized count equality, not just equal expected probabilities.
    block_steps, batch_size = 3, 16
    total_positions = block_steps * batch_size * HOPS
    rank_counts = largest_remainder_counts(zipf_prob(1.5), total_positions)
    realized = {}
    for name, shifts in {"slow": slow_shifts, "fast": fast_shifts}.items():
        counts = np.zeros(N_SKILLS, dtype=np.int64)
        for shift in shifts:
            counts[rank_to_skill_for_shift(base, int(shift))] += rank_counts
        assert counts.min() == counts.max()
        realized[name] = counts
    assert np.array_equal(realized["slow"], realized["fast"])


def test_identical_balanced_block_multiset() -> None:
    cycles = 2
    slow = make_shift_schedule("balanced_slow", cycles, 2718)
    fast = make_shift_schedule("balanced_fast", cycles, 2718)
    for cycle in range(cycles):
        lo, hi = cycle * N_SKILLS, (cycle + 1) * N_SKILLS
        slow_keys = sorted(
            (int(shift), block_stream_seed("balanced_slow", i, int(shift), 3, 31415))
            for i, shift in enumerate(slow[lo:hi], start=lo)
        )
        fast_keys = sorted(
            (int(shift), block_stream_seed("balanced_fast", i, int(shift), 3, 31415))
            for i, shift in enumerate(fast[lo:hi], start=lo)
        )
        assert slow_keys == fast_keys


def test_common_rank_stream() -> None:
    p = zipf_prob(1.5)
    a = exact_rank_block(p, block_steps=2, batch_size=32, hops=HOPS, block_seed=123)
    b = exact_rank_block(p, block_steps=2, batch_size=32, hops=HOPS, block_seed=123)
    c = exact_rank_block(p, block_steps=2, batch_size=32, hops=HOPS, block_seed=124)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    target = largest_remainder_counts(p, a.size)
    actual = np.bincount(a.reshape(-1), minlength=N_SKILLS)
    assert np.array_equal(actual, target)


def main() -> None:
    test_permutation_algebra()
    test_balanced_schedules()
    test_identical_balanced_block_multiset()
    test_common_rank_stream()
    print("SELF_TEST_OK")


if __name__ == "__main__":
    main()
