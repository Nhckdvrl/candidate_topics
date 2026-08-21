from src.analyze import subject_metrics


def _subject(hist, div, active_train_first, active_test_first):
    steps = []
    for epi in range(3):
        for st in range(2):
            active = bool(active_train_first if st == 0 else False)
            steps.append({
                "phase": "train", "episode_idx": epi, "step_idx": st,
                "format_valid": True, "active": active,
                "state_after": 1, "improved": False,
            })
    for st in range(2):
        active = bool(active_test_first if st == 0 else False)
        steps.append({
            "phase": "test", "episode_idx": 0, "step_idx": st,
            "format_valid": True, "active": active,
            "state_after": 1, "improved": active,
        })
    return {
        "base_seed": 0, "diversity": div, "history_controllability": hist,
        "test_family": "observatory", "steps": steps,
    }


def test_late_episode_first_action_metric_is_separate_from_late_rate():
    df = subject_metrics([_subject("controllable", "concentrated", True, True)])
    row = df.iloc[0]
    assert row.train_late_episode_first_active == 1.0
    assert row.train_late_active == 0.5
    assert row.test_step1_active == 1.0
