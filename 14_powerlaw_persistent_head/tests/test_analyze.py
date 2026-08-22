import numpy as np

from analyze import decide_full, decide_paper_anchor


def test_persistent_head_pass():
    decision, _ = decide_full(
        np.array([0.05, 0.04, 0.06, 0.05, 0.02]),
        np.array([0.12, 0.15, 0.11, 0.10, 0.08]),
    )
    assert decision == "PASS_PERSISTENT_HEAD_HELPS"


def test_rapid_switching_pass():
    decision, _ = decide_full(
        np.array([0.05, 0.04, 0.06, 0.05, 0.02]),
        np.array([-0.12, -0.15, -0.11, -0.10, -0.08]),
    )
    assert decision == "PASS_RAPID_SWITCHING_HELPS"


def test_clean_null_requires_valid_anchor():
    decision, _ = decide_full(
        np.array([0.05, 0.04, 0.06, 0.05, 0.02]),
        np.array([0.01, -0.02, 0.00, 0.02, -0.01]),
    )
    assert decision == "KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT"


def test_weak_anchor_never_kills_persistence_question():
    decision, _ = decide_full(
        np.array([0.01, -0.01, 0.00, 0.01, 0.00]),
        np.array([0.20, 0.22, 0.18, 0.19, 0.21]),
    )
    assert decision == "CORE_ANCHOR_WEAK_NO_PERSISTENCE_CONCLUSION"


def test_middle_effect_is_inconclusive_not_forced_to_null():
    decision, _ = decide_full(
        np.array([0.05, 0.04, 0.06, 0.05, 0.02]),
        np.array([0.07, 0.08, 0.06, 0.07, 0.08]),
    )
    assert decision == "INCONCLUSIVE_FIXED_PROTOCOL_NO_TUNING"


def test_paper_anchor_diagnostic_pass_and_fail():
    decision, _ = decide_paper_anchor(
        np.array([0.12, 0.11, 0.09]), np.array([0.70, 0.65, 0.60])
    )
    assert decision == "PAPER_ANCHOR_REPRODUCED"
    decision, _ = decide_paper_anchor(
        np.array([0.02, -0.01, 0.01]), np.array([0.25, 0.30, 0.20])
    )
    assert decision == "TECHNICAL_SEED_REPRODUCTION_FAILED_DEBUG_BEFORE_SCIENCE"
