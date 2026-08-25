"""Pure-logic tests for the frozen Topic 24 G1 channel-factorization decision.

No simulator, no policy. Pins down what each of the four hybrid conditions is
allowed to claim before any real number exists.
"""
import pytest

from g1_core import MIN_MATCHED_CONFIGS, direction_result, evaluate, structural_violations


def row(cfg, direction, cond, success):
    """A well-formed row. RR/LL mimic G0-reused rows; LR/RL mimic new hybrids."""
    upper_replayed = cond in ("RR", "LR")
    nav_replayed = cond in ("RR", "RL")
    steps = 400
    r = dict(
        config_id=cfg, direction=direction, condition=cond, force_n=100.0,
        success=bool(success),
        upper_replayed=upper_replayed, nav_replayed=nav_replayed,
        steps=steps,
    )
    if cond in ("LR", "RL"):
        r.update(
            source="g1_new", server_queries=17,
            nav_overwrites=steps if nav_replayed else 0,
            upper_overwrites=steps if upper_replayed else 0,
        )
    else:
        r.update(source="g0_reused")
        if cond == "RR":
            r["server_queries"] = 0
    return r


def panel(n=MIN_MATCHED_CONFIGS, *, direction="right", RR=0.0, LR=0.0, RL=0.0, LL=1.0):
    rows = []
    for i in range(n):
        rows.append(row(f"c{i}", direction, "RR", i < RR * n))
        rows.append(row(f"c{i}", direction, "LR", i < LR * n))
        rows.append(row(f"c{i}", direction, "RL", i < RL * n))
        rows.append(row(f"c{i}", direction, "LL", i < LL * n))
    return rows


def test_direction_result_decomposes_all_four_effects():
    rows = panel(10, RR=0.0, LR=0.8, RL=0.1, LL=1.0)
    r = direction_result(rows, "right")
    assert r.RR == 0.0 and r.LR == 0.8 and r.RL == 0.1 and r.LL == 1.0
    assert abs(r.nav_effect_upper_replayed - 0.8) < 1e-12   # LR - RR
    assert abs(r.nav_effect_upper_live - 0.9) < 1e-12       # LL - RL
    assert abs(r.upper_effect_nav_replayed - 0.1) < 1e-12   # RL - RR
    assert abs(r.upper_effect_nav_live - 0.2) < 1e-12       # LL - LR


def test_incomplete_quadruples_raise():
    rows = [row("a", "left", "RR", True), row("a", "left", "LR", True)]
    with pytest.raises(ValueError, match="no complete"):
        direction_result(rows, "left")


def test_structural_violation_wrong_channel_flags():
    rows = panel(4)
    rows[0]["upper_replayed"] = False  # this row is condition RR, must be True
    assert any("implies upper channel replayed" in v for v in structural_violations(rows))


def test_structural_violation_wrong_force():
    rows = panel(4)
    rows[0]["force_n"] = 50.0
    assert any("only runs force_n=100" in v for v in structural_violations(rows))


def test_rr_touching_server_is_a_violation():
    rows = panel(4)
    rr = next(r for r in rows if r["condition"] == "RR")
    rr["server_queries"] = 3
    assert any("RR contacted the policy server" in v for v in structural_violations(rows))


def test_hybrid_that_never_queried_the_live_vla_is_a_violation():
    rows = panel(4)
    lr = next(r for r in rows if r["condition"] == "LR")
    lr["server_queries"] = 0
    assert any("never queried the live VLA" in v for v in structural_violations(rows))


def test_partial_channel_overwrite_is_a_violation():
    # The hook must fire on every tick, not most of them.
    rows = panel(4)
    lr = next(r for r in rows if r["condition"] == "LR")
    lr["upper_overwrites"] = lr["steps"] - 3
    assert any("overwrote" in v and "of 400 ticks" in v for v in structural_violations(rows))


def test_channel_claimed_live_but_overwritten_is_a_violation():
    rows = panel(4)
    lr = next(r for r in rows if r["condition"] == "LR")
    lr["nav_overwrites"] = 5  # LR keeps navigation live
    assert any("claimed live but was overwritten" in v for v in structural_violations(rows))


def test_reused_g0_rows_are_exempt_from_overwrite_counters():
    # RR/LL come from G0 and carry no counters; that must not be flagged.
    rows = panel(4)
    assert structural_violations(rows) == []


def test_missing_direction_stops_before_reading_effects():
    rows = panel(MIN_MATCHED_CONFIGS, direction="right")
    res = evaluate(rows)
    assert res["verdict"] == "INSUFFICIENT_MATCHED_CONFIGS"
    assert "bootstrap" not in res


def test_too_few_configs_in_one_direction_stops():
    rows = panel(MIN_MATCHED_CONFIGS, direction="left") + panel(MIN_MATCHED_CONFIGS - 1, direction="right")
    res = evaluate(rows)
    assert res["verdict"] == "INSUFFICIENT_MATCHED_CONFIGS"


def test_navigation_channel_causes_reversal():
    # nav flips outcome regardless of upper; upper does nothing on its own.
    rows = (
        panel(MIN_MATCHED_CONFIGS, direction="left", RR=0.0, LR=0.9, RL=0.0, LL=0.9)
        + panel(MIN_MATCHED_CONFIGS, direction="right", RR=0.0, LR=0.9, RL=0.0, LL=0.9)
    )
    res = evaluate(rows)
    assert res["verdict"] == "NAVIGATION_CHANNEL_CAUSES_REVERSAL"


def test_upper_body_channel_causes_reversal():
    rows = (
        panel(MIN_MATCHED_CONFIGS, direction="left", RR=0.0, LR=0.0, RL=0.9, LL=0.9)
        + panel(MIN_MATCHED_CONFIGS, direction="right", RR=0.0, LR=0.0, RL=0.9, LL=0.9)
    )
    res = evaluate(rows)
    assert res["verdict"] == "UPPER_BODY_CHANNEL_CAUSES_REVERSAL"


def test_no_single_channel_effect_when_all_flat():
    rows = (
        panel(MIN_MATCHED_CONFIGS, direction="left", RR=0.5, LR=0.5, RL=0.5, LL=0.5)
        + panel(MIN_MATCHED_CONFIGS, direction="right", RR=0.5, LR=0.5, RL=0.5, LL=0.5)
    )
    res = evaluate(rows)
    assert res["verdict"] == "CROSS_CHANNEL_INTERACTION_OR_NO_SINGLE_CHANNEL_EFFECT"


def test_both_channels_contribute():
    rows = (
        panel(MIN_MATCHED_CONFIGS, direction="left", RR=0.0, LR=0.6, RL=0.6, LL=0.9)
        + panel(MIN_MATCHED_CONFIGS, direction="right", RR=0.0, LR=0.6, RL=0.6, LL=0.9)
    )
    res = evaluate(rows)
    assert res["verdict"] == "BOTH_CHANNELS_CONTRIBUTE"
