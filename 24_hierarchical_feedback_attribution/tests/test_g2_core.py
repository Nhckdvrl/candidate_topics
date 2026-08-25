"""Pure-logic tests for the frozen Topic 24 G2 reversal predicate.

Pins down the correction identified in G1_RESULTS.md's post-result audit:
a reversal requires opposite-signed, independently-significant effects in
both directions, not merely "some navigation effect is significant."
"""
import pytest

from g2_core import FORCES_N, MIN_MATCHED_CONFIGS, evaluate, reversal_at_force, structural_violations


def row(cfg, force, direction, cond, success):
    # Matches the real pipeline: RR/LL are reused from G0, LR/RL are newly
    # collected here and carry overwrite counters as proof the hook fired.
    upper_replayed = cond in ("RR", "LR")
    nav_replayed = cond in ("RR", "RL")
    steps = 400
    source = "g2_new" if cond in ("LR", "RL") else "g0_reused"
    r = dict(
        config_id=cfg, force_n=force, direction=direction, condition=cond,
        success=bool(success), upper_replayed=upper_replayed, nav_replayed=nav_replayed,
        steps=steps, source=source,
    )
    if cond in ("LR", "RL"):
        r.update(server_queries=17, nav_overwrites=steps if nav_replayed else 0,
                  upper_overwrites=steps if upper_replayed else 0)
    elif cond == "RR":
        r["server_queries"] = 0
    return r


def panel_at(force, n=MIN_MATCHED_CONFIGS, *, RR_left, LR_left, RR_right, LR_right):
    rows = []
    for i in range(n):
        rows.append(row(f"c{i}", force, "left", "RR", i < round(RR_left * n)))
        rows.append(row(f"c{i}", force, "left", "LR", i < round(LR_left * n)))
        rows.append(row(f"c{i}", force, "right", "RR", i < round(RR_right * n)))
        rows.append(row(f"c{i}", force, "right", "LR", i < round(LR_right * n)))
    return rows


def full_grid(overrides=None):
    """All three forces with a real reversal at each, unless overridden."""
    overrides = overrides or {}
    default = dict(RR_left=0.1, LR_left=0.4, RR_right=0.6, LR_right=0.2)
    rows = []
    for f in FORCES_N:
        kw = dict(default)
        kw.update(overrides.get(f, {}))
        rows += panel_at(f, **kw)
    return rows


def test_N_is_LR_minus_RR():
    rows = panel_at(100.0, n=100, RR_left=0.1, LR_left=0.4, RR_right=0.6, LR_right=0.2)
    r = reversal_at_force(rows, 100.0)
    assert abs(r["N_left"][0] - 0.3) < 1e-9
    assert abs(r["N_right"][0] - (-0.4)) < 1e-9


def test_reversal_requires_opposite_sign_not_just_any_significant_effect():
    # left real and positive, right exactly flat -> NOT a reversal, unlike g1_core's `any()`.
    rows = panel_at(100.0, RR_left=0.1, LR_left=0.4, RR_right=0.5, LR_right=0.5)
    r = reversal_at_force(rows, 100.0)
    assert r["left_significant"] is True
    assert r["right_significant"] is False
    assert r["reversal_established"] is False


def test_reversal_requires_both_directions_significant_and_opposite():
    rows = panel_at(100.0, RR_left=0.1, LR_left=0.4, RR_right=0.6, LR_right=0.2)
    r = reversal_at_force(rows, 100.0)
    assert r["left_significant"] and r["right_significant"] and r["opposite_sign"]
    assert r["reversal_established"] is True


def test_same_signed_large_effects_are_not_a_reversal():
    # Both directions positive and large: real effects, same sign -> not a reversal.
    rows = panel_at(100.0, RR_left=0.1, LR_left=0.4, RR_right=0.1, LR_right=0.4)
    r = reversal_at_force(rows, 100.0)
    assert r["opposite_sign"] is False
    assert r["reversal_established"] is False


def test_structural_violation_wrong_force():
    rows = panel_at(100.0, RR_left=0.1, LR_left=0.4, RR_right=0.6, LR_right=0.2)
    rows[0]["force_n"] = 75.0
    assert any("force_n must be one of" in v for v in structural_violations(rows))


def test_rr_touching_server_is_violation():
    rows = panel_at(100.0, RR_left=0.1, LR_left=0.4, RR_right=0.6, LR_right=0.2)
    rr = next(r for r in rows if r["condition"] == "RR")
    rr["server_queries"] = 2
    assert any("RR contacted the policy server" in v for v in structural_violations(rows))


def test_partial_overwrite_is_violation():
    rows = panel_at(100.0, RR_left=0.1, LR_left=0.4, RR_right=0.6, LR_right=0.2)
    lr = next(r for r in rows if r["condition"] == "LR")
    lr["upper_overwrites"] = lr["steps"] - 1
    assert any("overwrote" in v for v in structural_violations(rows))


def test_missing_force_stops_before_reading_effects():
    rows = panel_at(100.0, RR_left=0.1, LR_left=0.4, RR_right=0.6, LR_right=0.2)
    res = evaluate(rows)
    assert res["verdict"] == "INSUFFICIENT_MATCHED_CONFIGS"
    assert set(res["missing_forces"]) == {50.0, 150.0}


def test_confirmed_across_grid_when_all_three_forces_reverse():
    rows = full_grid()
    res = evaluate(rows)
    assert res["verdict"] == "REVERSAL_CONFIRMED_ACROSS_FORCE_GRID"
    assert set(res["reversal_established_at"]) == set(FORCES_N)


def test_confirmed_at_some_forces_only():
    rows = full_grid({50.0: dict(RR_left=0.5, LR_left=0.5, RR_right=0.5, LR_right=0.5)})
    res = evaluate(rows)
    assert res["verdict"] == "REVERSAL_CONFIRMED_AT_SOME_FORCES_ONLY"
    assert 50.0 not in res["reversal_established_at"]
    assert 100.0 in res["reversal_established_at"]


def test_not_established_when_no_force_reverses():
    flat = dict(RR_left=0.5, LR_left=0.5, RR_right=0.5, LR_right=0.5)
    rows = full_grid({f: flat for f in FORCES_N})
    res = evaluate(rows)
    assert res["verdict"] == "REVERSAL_NOT_ESTABLISHED_OUTSIDE_100N"
    assert res["reversal_established_at"] == []
