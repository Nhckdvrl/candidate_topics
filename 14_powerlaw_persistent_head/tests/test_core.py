import numpy as np
import torch

from core import (
    LOCKED_FULL_SEEDS,
    N_SKILLS,
    StateTrackingTransformer,
    all_permutations,
    compose_numpy,
    head_overlap,
    key_schedule,
    make_power_batch,
    map_orders,
    mapping_seed_for_seed,
    max_map_run,
    paper_lr_at_step,
    schedule_digests,
)


def test_group_size_and_identity():
    p = all_permutations()
    assert p.shape == (120, 5)
    ids = np.tile(np.arange(5), (3, 4, 1))
    assert np.all(compose_numpy(ids) == np.arange(5))


def test_s5_is_closed_under_composition_exhaustively():
    p = all_permutations()
    universe = {tuple(x) for x in p.tolist()}
    for i in range(N_SKILLS):
        lhs = np.repeat(p[i][None, :], N_SKILLS, axis=0)
        pairs = np.stack([lhs, p], axis=1)
        out = compose_numpy(pairs)
        assert all(tuple(x) in universe for x in out.tolist())


def test_composition_orientation():
    # p=(01), q=(12). Our oracle implements p∘q: apply q to indices, then p.
    p = np.array([1, 0, 2, 3, 4])
    q = np.array([0, 2, 1, 3, 4])
    out = compose_numpy(np.array([[p, q]]))[0]
    assert np.array_equal(out, p[q])


def test_slow_fast_same_multiset_different_order():
    slow = key_schedule("slow", 200, 100)
    fast = key_schedule("fast", 200, 100)
    ds, df = schedule_digests(slow), schedule_digests(fast)
    assert ds["multiset_digest"] == df["multiset_digest"]
    assert ds["temporal_digest"] != df["temporal_digest"]
    assert max_map_run(slow) == 100
    assert max_map_run(fast) == 1


def test_batch_key_identity():
    p = all_permutations()
    x1, y1 = make_power_batch(3, "A", 17, 16, 1.5, 1729, 31415, p)
    x2, y2 = make_power_batch(3, "A", 17, 16, 1.5, 1729, 31415, p)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_mapping_pair_is_paired_but_randomized_across_replications():
    effective = [mapping_seed_for_seed(1729, s) for s in LOCKED_FULL_SEEDS]
    assert len(set(effective)) == len(effective)
    maps = []
    for ms in effective:
        a, b = map_orders(ms)
        assert head_overlap(a, b) == 0
        maps.append(tuple(a.tolist()))
    assert len(set(maps)) == len(maps)


def test_persistence_h_extremes():
    k1 = key_schedule("persistence", 20, 10, 1)
    k5 = key_schedule("persistence", 20, 10, 5)
    assert max_map_run(k1) == 1
    assert max_map_run(k5) == 5
    assert schedule_digests(k1)["multiset_digest"] == schedule_digests(k5)["multiset_digest"]


def test_paper_lr_schedule():
    peak = 2e-4
    assert np.isclose(paper_lr_at_step(0, 200_000, peak), peak / 1000)
    assert np.isclose(paper_lr_at_step(999, 200_000, peak), peak)
    assert np.isclose(paper_lr_at_step(199_999, 200_000, peak), 0.1 * peak)


def test_model_forward():
    m = StateTrackingTransformer(d_model=32, layers=1, heads=4, ff_mult=2)
    out = m(torch.randint(0, 5, (2, 20)))
    assert out.shape == (2, 5, 5)
