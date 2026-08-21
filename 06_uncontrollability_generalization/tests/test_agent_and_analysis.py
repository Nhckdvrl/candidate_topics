from src.agent import parse_action
from src.analyze import contrasts, subject_metrics


def test_parse_action_strict_and_recoverable():
    assert parse_action("A") == "A"
    assert parse_action("WAIT") == "WAIT"
    assert parse_action("I choose B.") == "B"
    assert parse_action("no valid choice") is None


def _subject(seed, diversity, hc, first_active, late_active):
    steps = []
    for epi in range(10):
        for st in range(2):
            active = bool(late_active) if epi >= 7 else True
            steps.append({
                "phase": "train", "episode_idx": epi, "step_idx": st, "format_valid": True,
                "active": active, "state_after": 1, "improved": active,
            })
    for st in range(3):
        active = bool(first_active) if st == 0 else True
        steps.append({
            "phase": "test", "episode_idx": 0, "step_idx": st, "format_valid": True,
            "active": active, "state_after": 1, "improved": active,
        })
    return {
        "base_seed": seed,
        "diversity": diversity,
        "history_controllability": hc,
        "test_family": "observatory",
        "steps": steps,
    }


def test_interaction_contrast_sign():
    subjects = [
        _subject(0, "concentrated", "controllable", 1, 1),
        _subject(0, "distributed", "controllable", 1, 1),
        _subject(0, "concentrated", "uncontrollable", 1, 0),
        _subject(0, "distributed", "uncontrollable", 0, 0),
    ]
    df = subject_metrics(subjects)
    c = contrasts(df, "test_step1_active")
    assert c["diversity_interaction"] == -1.0
    assert c["pooled_U_minus_C"] == -0.5
