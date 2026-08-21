from src.renderers import get_family, schedule_for
from src.runner import FORBIDDEN_CONSTRUCT_WORDS, SYSTEM_PROMPT


def test_concentrated_and_distributed_have_same_episode_count():
    c = schedule_for("concentrated", 10, pair_id=3)
    d = schedule_for("distributed", 10, pair_id=3)
    assert len(c) == len(d) == 10
    assert len(set(c)) == 1
    assert len(set(d)) == 10


def test_surface_actions_are_three_way_and_unique_within_family():
    for key in schedule_for("distributed", 10, pair_id=0) + ["orbital_station"]:
        actions = get_family(key).surface_actions
        assert len(actions) == 3
        assert len(set(actions)) == 3


def test_prompts_do_not_name_target_construct():
    texts = [SYSTEM_PROMPT]
    for key in schedule_for("distributed", 10, pair_id=0) + ["orbital_station"]:
        r = get_family(key)
        texts += [r.render_start(1, 1), r.render_trial(1)]
    joined = " ".join(texts).lower()
    for bad in FORBIDDEN_CONSTRUCT_WORDS:
        assert bad not in joined


def test_concentrated_family_rotates_across_pairs():
    seen = {schedule_for("concentrated", 10, pair_id=i)[0] for i in range(10)}
    assert len(seen) == 10
