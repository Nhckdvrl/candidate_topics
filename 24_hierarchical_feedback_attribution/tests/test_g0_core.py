"""Pure-logic tests for the frozen Topic 24 G0 decision procedure.

No simulator, no policy. These pin the semantics so the frozen contract cannot
be quietly weakened once real numbers exist.
"""
import numpy as np
import pytest

from g0_core import (
    FORCES_N,
    MIN_MATCHED_CONFIGS,
    attribution,
    clustered_bootstrap,
    evaluate,
    fidelity_control,
    per_cell,
    push_effective,
    structural_violations,
)


def row(cfg, force, direction, cond, success, **kw):
    r = dict(
        config_id=cfg, force_n=float(force), direction=direction, condition=cond,
        success=bool(success), server_queries=(12 if cond == "fresh" else 0),
        tape_exhausted_early=False,
        push_applied=force > 0, push_tick=100,
        push_displacement_m=(0.08 if force > 0 else None),
    )
    r.update(kw)
    return r


def panel(n_cfg=MIN_MATCHED_CONFIGS, *, fresh=1.0, vla=1.0, act=1.0, zero_ok=True):
    """Build a full grid where each condition succeeds on a fixed fraction."""
    rows = []
    for i in range(n_cfg):
        rows.append(row(f"c{i}", 0, "none", "fresh", True))
        rows.append(row(f"c{i}", 0, "none", "vla_replay", zero_ok))
        rows.append(row(f"c{i}", 0, "none", "actuator_replay", zero_ok))
        for force in FORCES_N:
            for d in ("left", "right"):
                rows.append(row(f"c{i}", force, d, "fresh", i < fresh * n_cfg))
                rows.append(row(f"c{i}", force, d, "vla_replay", i < vla * n_cfg))
                rows.append(row(f"c{i}", force, d, "actuator_replay", i < act * n_cfg))
    return rows


def test_attribution_splits_the_two_seams():
    rows = panel(10, fresh=1.0, vla=0.8, act=0.3)
    a = attribution([r for r in rows if r["force_n"] > 0])
    assert a.fresh == 1.0 and a.vla_replay == 0.8 and a.actuator_replay == 0.3
    assert abs(a.delta_high - 0.2) < 1e-12
    assert abs(a.delta_low - 0.5) < 1e-12
    assert a.residual == 0.3


def test_incomplete_triples_are_dropped_not_imputed():
    rows = [row("a", 100, "left", "fresh", True), row("a", 100, "left", "vla_replay", True)]
    with pytest.raises(ValueError, match="no complete"):
        attribution(rows)


def test_every_grid_cell_is_reported():
    cells = per_cell(panel(4))
    assert len(cells) == 1 + len(FORCES_N) * 2
    assert [c.force_n for c in cells][0] == 0.0


def test_replay_touching_the_server_is_structural_failure():
    rows = panel(4)
    rows[1]["server_queries"] = 1
    assert any("policy server" in v for v in structural_violations(rows))


def test_nonzero_force_without_an_applied_push_is_structural_failure():
    rows = panel(4)
    target = next(r for r in rows if r["force_n"] > 0)
    target["push_applied"] = False
    assert any("no applied push" in v for v in structural_violations(rows))


def test_control_column_with_a_push_is_structural_failure():
    rows = panel(4)
    target = next(r for r in rows if r["force_n"] == 0)
    target["push_applied"] = True
    assert any("control cell" in v for v in structural_violations(rows))


def test_push_tick_must_match_across_conditions():
    rows = panel(4)
    target = next(r for r in rows if r["force_n"] > 0 and r["condition"] == "vla_replay")
    target["push_tick"] = 7
    assert any("push tick differs" in v for v in structural_violations(rows))


def test_fidelity_control_reads_only_the_zero_force_column():
    ok = fidelity_control(panel(10, fresh=0.0, vla=0.0, act=0.0, zero_ok=True))
    assert ok["pass"] and ok["n_configs"] == 10
    bad = fidelity_control(panel(10, zero_ok=False))
    assert not bad["pass"]


def test_ineffective_push_is_caught_at_the_largest_force():
    rows = panel(4)
    for r in rows:
        if r["force_n"] == max(FORCES_N):
            r["push_displacement_m"] = 0.001
    assert not push_effective(rows)["pass"]
    assert push_effective(panel(4))["pass"]


def test_fidelity_failure_blocks_every_delta():
    res = evaluate(panel(zero_ok=False), n_boot=200)
    assert res["verdict"] == "PREREQUISITE_FAIL_REPLAY_FIDELITY"
    assert "pooled_perturbed" not in res


def test_structural_failure_outranks_fidelity():
    rows = panel(zero_ok=False)
    rows[1]["server_queries"] = 4
    assert evaluate(rows, n_boot=200)["verdict"] == "PREREQUISITE_FAIL_STRUCTURAL"


def test_too_few_matched_configs_stops_before_reading_deltas():
    res = evaluate(panel(MIN_MATCHED_CONFIGS - 1), n_boot=200)
    assert res["verdict"] == "INSUFFICIENT_MATCHED_CONFIGS"
    assert "pooled_perturbed" not in res


def test_no_robustness_phenomenon_when_even_actuator_replay_survives():
    assert evaluate(panel(), n_boot=200)["verdict"] == "NO_ROBUSTNESS_PHENOMENON"


def test_fresh_collapse_is_reported_as_nothing_to_attribute():
    res = evaluate(panel(fresh=0.0, vla=0.0, act=0.0), n_boot=200)
    assert res["verdict"] == "FRESH_COLLAPSE_NOTHING_TO_ATTRIBUTE"


def test_wbc_level_dominates_when_only_the_lower_gap_is_real():
    res = evaluate(panel(fresh=1.0, vla=1.0, act=0.2), n_boot=200)
    assert res["verdict"] == "WBC_LEVEL_DOMINATES"
    assert res["pooled_perturbed"]["delta_high"] == 0.0


def test_vla_level_dominates_when_only_the_upper_gap_is_real():
    res = evaluate(panel(fresh=1.0, vla=0.2, act=0.2), n_boot=200)
    assert res["verdict"] == "VLA_LEVEL_DOMINATES"


def test_both_levels_contribute():
    res = evaluate(panel(fresh=1.0, vla=0.6, act=0.2), n_boot=200)
    assert res["verdict"] == "BOTH_LEVELS_CONTRIBUTE"


def test_small_gaps_are_not_promoted_to_contributions():
    # Both gaps point the right way but are below the pre-registered minimum
    # worthy effect, and actuator_replay is low enough that the no-phenomenon
    # rule does not fire first.
    n = MIN_MATCHED_CONFIGS
    res = evaluate(panel(n, fresh=21 / n, vla=20 / n, act=19 / n), n_boot=200)
    assert res["pooled_perturbed"]["delta_high"] < 0.10
    assert res["verdict"] == "NO_MEANINGFUL_LEARNED_FEEDBACK_CONTRIBUTION"


def test_bootstrap_clusters_configs_and_brackets_the_point_estimate():
    rows = [r for r in panel(12, fresh=1.0, vla=0.7, act=0.3) if r["force_n"] > 0]
    ci = clustered_bootstrap(rows, n_boot=200, seed=1)
    for k in ("delta_high", "delta_low"):
        pt, lo, hi = ci[k]
        assert lo <= pt <= hi
        assert np.isfinite(lo) and np.isfinite(hi)
