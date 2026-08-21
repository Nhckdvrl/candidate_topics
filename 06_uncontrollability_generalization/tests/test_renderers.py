from src.renderers import TEST_FAMILIES, TRAIN_FAMILIES, render_episode_intro, heldout_family_key, training_family_keys


def test_train_test_families_disjoint():
    train = {f.key for f in TRAIN_FAMILIES}
    test = {f.key for f in TEST_FAMILIES}
    assert train.isdisjoint(test)


def test_concentrated_repeats_one_family():
    keys = training_family_keys(7, "concentrated", 10)
    assert len(keys) == 10
    assert len(set(keys)) == 1


def test_distributed_uses_all_ten_families_for_ten_episodes():
    keys = training_family_keys(7, "distributed", 10)
    assert len(keys) == 10
    assert len(set(keys)) == 10


def test_intro_never_leaks_controllability_condition():
    text = render_episode_intro("greenhouse", 3, 10, 6).lower()
    assert "controllable" not in text
    assert "uncontrollable" not in text
    assert all(tok in text.upper() for tok in ["A", "B", "C", "WAIT"])


def test_heldout_family_is_deterministic():
    assert heldout_family_key(5) == heldout_family_key(5)
