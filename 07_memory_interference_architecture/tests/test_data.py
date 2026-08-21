from memory_interference.data import build_episode, select_query_keys


def toy_pool(n=12):
    return {
        "bird": [f"bird{i}" for i in range(n)],
        "tool": [f"tool{i}" for i in range(n)],
        "fruit": [f"fruit{i}" for i in range(n)],
        "art": [f"art{i}" for i in range(n)],
    }


def test_episode_has_distinct_histories_and_expected_lengths():
    ep = build_episode(toy_pool(), episode_id=0, num_keys=3, num_updates=4, seed=7)
    assert len(ep.assignments) == 3 * 5
    assert all(len(v) == 5 for v in ep.histories.values())
    assert all(len(set(v)) == 5 for v in ep.histories.values())
    for key in ep.categories:
        assert ep.first(key) != ep.latest(key)


def test_episode_is_deterministic():
    a = build_episode(toy_pool(), episode_id=3, num_keys=3, num_updates=2, seed=11)
    b = build_episode(toy_pool(), episode_id=3, num_keys=3, num_updates=2, seed=11)
    assert a == b


def test_query_key_selection_is_deterministic_and_bounded():
    ep = build_episode(toy_pool(), episode_id=1, num_keys=4, num_updates=2, seed=5)
    q1 = select_query_keys(ep, 2, 99)
    q2 = select_query_keys(ep, 2, 99)
    assert q1 == q2
    assert len(set(q1)) == 2
    assert set(q1).issubset(ep.categories)


def test_updates_are_round_balanced_and_preserve_per_key_update_order():
    ep = build_episode(toy_pool(), episode_id=9, num_keys=3, num_updates=3, seed=17)
    updates = [a for a in ep.assignments if not a.is_initial]
    for start in range(0, len(updates), len(ep.categories)):
        block = updates[start : start + len(ep.categories)]
        assert {a.category for a in block} == set(ep.categories)
    assert all(a.category != b.category for a, b in zip(updates, updates[1:]))
    for key in ep.categories:
        presented = [a.value for a in ep.assignments if a.category == key]
        assert tuple(presented) == ep.histories[key]
