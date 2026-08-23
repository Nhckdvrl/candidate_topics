import numpy as np
import pytest

from g0_core import (
    Condition,
    GateConfig,
    analyze_records,
    realized_motion_l2,
    task_effect_success,
)

ALL = [c.value for c in Condition]


def _rows(
    n=24,
    canonical=24,
    right_frozen=4,          # freezing the arm in place costs the policy a lot
    right_disabled=8,
    left_disabled=24,
    both_arms=0,
    full_hold=0,
    oracle=24,
    canonical_right_route=True,
    clamp_leak=0.02,
):
    per_cond = {
        "canonical": canonical,
        "right_frozen": right_frozen,
        "right_disabled": right_disabled,
        "left_disabled": left_disabled,
        "both_arms_disabled": both_arms,
        "full_hold": full_hold,
        "oracle_right_disabled": oracle,
    }
    rows = []
    for i in range(n):
        for c in ALL:
            success = i < per_cond[c]
            row = {
                "task": "close_door",
                "config_id": str(i),
                "condition": c,
                "success": success,
            }
            if c == "canonical":
                row["canonical_right_route"] = canonical_right_route
            if c == "right_disabled":
                row["route_verified"] = True if success else None
                row["right_arm_clamp_leak_rad"] = clamp_leak
            rows.append(row)
    return rows


def test_task_effect_success_is_outcome_space():
    assert task_effect_success("close_door", -0.2)
    assert not task_effect_success("close_door", -0.1)
    assert task_effect_success("open_faucet", 0.8)
    assert task_effect_success("open_faucet", -0.8)
    assert not task_effect_success("open_faucet", 0.2)


def test_realized_motion_l2_is_path_length():
    states = [{"a": np.zeros(2)}, {"a": np.array([3.0, 4.0])}, {"a": np.zeros(2)}]
    assert realized_motion_l2(states, "a") == pytest.approx(10.0)


def test_promising_gate():
    report = analyze_records(_rows())
    assert report["n_matched_configs"] == 24
    assert report["substitution_events"] == 8
    assert report["verdict"] == "PROMISING_MOTOR_SUBSTITUTION"


def test_canonical_competence_is_first_gate():
    assert analyze_records(_rows(canonical=10))["verdict"] == "PREREQUISITE_FAIL_CANONICAL"


def test_arm_with_no_motor_program_is_killed():
    """Freezing the right arm in place costs nothing -> nothing to substitute for."""
    report = analyze_records(_rows(right_frozen=23))
    assert report["arm_program_cost"] == pytest.approx(1 / 24)
    assert report["verdict"] == "PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM"


def test_canonical_route_must_run_through_the_right_side():
    report = analyze_records(_rows(canonical_right_route=False))
    assert report["canonical_right_route_rate"] == 0.0
    assert report["verdict"] == "PREREQUISITE_FAIL_ROUTE_NOT_RIGHT_SIDE"


def test_leaking_clamp_is_rejected():
    report = analyze_records(_rows(clamp_leak=0.9))
    assert report["verdict"] == "PREREQUISITE_FAIL_INTERVENTION_LEAK"


def test_body_only_route_is_killed():
    """If the task survives losing both arms, one arm cannot stand in for the other."""
    report = analyze_records(_rows(both_arms=12))
    assert report["verdict"] == "PREREQUISITE_FAIL_BODY_ONLY_ROUTE"


def test_oracle_failure_is_prerequisite_failure():
    report = analyze_records(_rows(oracle=10))
    assert report["verdict"] == "PREREQUISITE_FAIL_ALTERNATIVE_FEASIBILITY"


def test_accidental_success_is_prerequisite_failure():
    report = analyze_records(_rows(full_hold=12))
    assert report["verdict"] == "PREREQUISITE_FAIL_NEGATIVE_CONTROL"


def test_no_evidence_gate():
    assert analyze_records(_rows(right_disabled=1))["verdict"] == "NO_EVIDENCE_IN_PSI0_G0"


def test_unverified_route_does_not_claim_substitution():
    rows = _rows()
    for r in rows:
        if r["condition"] == "right_disabled" and r["success"]:
            r["route_verified"] = False
    assert analyze_records(rows)["verdict"] == "PROMISING_NEEDS_ROUTE_VERIFICATION"


def test_missing_conditions_are_not_primary_units():
    rows = [
        r for r in _rows()
        if not (r["config_id"] == "0" and r["condition"] == "full_hold")
    ]
    report = analyze_records(rows, GateConfig(min_matched_configs=20))
    assert report["n_matched_configs"] == 23


def test_missing_oracle_does_not_zero_the_matched_count():
    """The scripted oracle is run separately; its absence must not hide the gates."""
    rows = [r for r in _rows() if r["condition"] != "oracle_right_disabled"]
    report = analyze_records(rows)
    assert report["n_matched_configs"] == 24
    assert report["n_oracle_rows"] == 0
    assert report["verdict"] == "PREREQUISITE_PENDING_ALTERNATIVE_FEASIBILITY"


def test_missing_oracle_still_reports_earlier_gate_failures():
    rows = [
        r for r in _rows(right_frozen=23) if r["condition"] != "oracle_right_disabled"
    ]
    assert analyze_records(rows)["verdict"] == "PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM"
