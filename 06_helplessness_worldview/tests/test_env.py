from src.config import ExperimentConfig
from src.env import build_session_spec, controlled_feedback, validate_yoke, yoked_feedback


def test_session_spec_is_deterministic_and_balanced():
    cfg = ExperimentConfig(episodes=10, trials_per_episode=3)
    a = build_session_spec("distributed", 7, cfg, 123)
    b = build_session_spec("distributed", 7, cfg, 123)
    assert a == b
    assert len(a.episodes) == 10
    assert len({e.family for e in a.episodes}) == 10


def test_controllable_action_changes_success_probability():
    cfg = ExperimentConfig(trials_per_episode=1)
    wins_eff = 0
    wins_other = 0
    # Across deterministic seeds, effective action should win far more often.
    for pair_id in range(500):
        spec = build_session_spec("concentrated", pair_id, cfg, 99).episodes[0]
        eff = spec.effective_action
        other = "b" if eff == "a" else "a"
        wins_eff += controlled_feedback(eff, spec, 0, cfg).success
        wins_other += controlled_feedback(other, spec, 0, cfg).success
    assert wins_eff - wins_other > 250


def test_yoked_feedback_exactly_replays_master_outcomes():
    cfg = ExperimentConfig()
    master = [True, False, True, True, False]
    actions = ["a", "wait", "b", "wait", "a"]
    replay = [yoked_feedback(a, y, cfg).success for a, y in zip(actions, master)]
    validate_yoke(master, replay)
    assert replay == master


def test_novel_test_kernel_is_identical_across_diversity_for_same_pair():
    cfg = ExperimentConfig(episodes=10, trials_per_episode=2, test_trials=4)
    c = build_session_spec("concentrated", 3, cfg, 777)
    d = build_session_spec("distributed", 3, cfg, 777)
    assert c.test == d.test


def test_latent_training_randomness_is_matched_across_diversity():
    cfg = ExperimentConfig(episodes=10, trials_per_episode=4)
    c = build_session_spec("concentrated", 5, cfg, 888)
    d = build_session_spec("distributed", 5, cfg, 888)
    assert [e.effective_action for e in c.episodes] == [e.effective_action for e in d.episodes]
    assert [e.uniforms for e in c.episodes] == [e.uniforms for e in d.episodes]
