import numpy as np

from g0_core import (
    Condition,
    GateConfig,
    analyze_records,
    intervene_absolute_action,
    task_effect_success,
)


def _action_state():
    state = {
        "left_hand": np.array([0.1, 0.2]),
        "right_hand": np.array([0.3, 0.4]),
        "left_arm": np.arange(7.0),
        "right_arm": np.arange(7.0) + 10,
        "rpy": np.array([0.1, 0.2, 0.3]),
        "height": np.array([0.9]),
    }
    action = {
        "left_hand": np.array([1.0, 1.0]),
        "right_hand": np.array([1.0, 1.0]),
        "left_arm": np.ones(7),
        "right_arm": np.ones(7) * 2,
        "rpy": np.ones(3),
        "height": np.array([1.1]),
        "torso_vx": np.array([0.2]),
        "torso_vy": np.array([-0.1]),
        "torso_vyaw": np.array([0.4]),
        "target_yaw": np.array([1.5]),
    }
    return action, state


def test_right_disabled_holds_only_right_groups():
    action, state = _action_state()
    out = intervene_absolute_action(action, state, Condition.RIGHT_DISABLED)
    np.testing.assert_allclose(out["right_arm"], state["right_arm"])
    np.testing.assert_allclose(out["right_hand"], state["right_hand"])
    np.testing.assert_allclose(out["left_arm"], action["left_arm"])
    np.testing.assert_allclose(out["torso_vx"], action["torso_vx"])


def test_full_hold_removes_intentional_motion():
    action, state = _action_state()
    out = intervene_absolute_action(action, state, Condition.FULL_HOLD)
    for key in ("left_hand", "right_hand", "left_arm", "right_arm", "rpy", "height"):
        np.testing.assert_allclose(out[key], state[key])
    for key in ("torso_vx", "torso_vy", "torso_vyaw"):
        np.testing.assert_allclose(out[key], 0.0)
    np.testing.assert_allclose(out["target_yaw"], state["rpy"][-1])


def test_task_effect_success_is_outcome_space():
    assert task_effect_success("close_door", -0.2)
    assert not task_effect_success("close_door", -0.1)
    assert task_effect_success("open_faucet", 0.8)
    assert task_effect_success("open_faucet", -0.8)
    assert not task_effect_success("open_faucet", 0.2)


def _rows(n=24, right_success=8, full_success=0, oracle_success=24, canonical_success=24):
    rows = []
    for i in range(n):
        values = {
            "canonical": i < canonical_success,
            "oracle_right_disabled": i < oracle_success,
            "right_disabled": i < right_success,
            "full_hold": i < full_success,
        }
        for c, success in values.items():
            rows.append(
                {
                    "task": "close_door",
                    "config_id": str(i),
                    "condition": c,
                    "success": success,
                    "route_verified": True if c == "right_disabled" and success else None,
                }
            )
    return rows


def test_promising_gate():
    report = analyze_records(_rows())
    assert report["n_matched_configs"] == 24
    assert report["substitution_events"] == 8
    assert report["verdict"] == "PROMISING_MOTOR_SUBSTITUTION"


def test_oracle_failure_is_prerequisite_failure():
    report = analyze_records(_rows(oracle_success=10))
    assert report["verdict"] == "PREREQUISITE_FAIL_ALTERNATIVE_FEASIBILITY"


def test_no_evidence_gate():
    report = analyze_records(_rows(right_success=1))
    assert report["verdict"] == "NO_EVIDENCE_IN_PSI0_G0"


def test_missing_conditions_are_not_primary_units():
    rows = _rows()
    rows = [
        r for r in rows
        if not (r["config_id"] == "0" and r["condition"] == "full_hold")
    ]
    report = analyze_records(rows, GateConfig(min_matched_configs=20))
    assert report["n_matched_configs"] == 23
