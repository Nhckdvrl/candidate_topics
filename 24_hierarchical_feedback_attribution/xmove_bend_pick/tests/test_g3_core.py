"""Pure-logic tests for the corrected Topic 24 G3 (VLA feedback) decision.

No simulator, no policy. Pins the four-way verdict and the structural gates
before any real rollout exists.
"""
import pytest

from g3_core import (
    FORCES_N, MIN_MATCHED_CONFIGS, N_CONFIGS,
    cell_result, evaluate, fidelity_control, per_cell,
    push_effective, structural_violations,
)


def row(cfg, force, direction, cond, success):
    return dict(
        config_id=cfg, force_n=float(force), direction=direction, condition=cond,
        success=bool(success), server_queries=(12 if cond == "fresh" else 0),
        tape_exhausted_early=False,
        push_applied=force > 0, push_tick=100,
        push_displacement_m=(0.08 if force > 0 else None),
    )


def panel(n=MIN_MATCHED_CONFIGS, *, fresh=1.0, vla=1.0, zero_ok=True):
    rows = []
    for i in range(n):
        rows.append(row(f"c{i}", 0, "none", "fresh", zero_ok))
        rows.append(row(f"c{i}", 0, "none", "vla_replay", zero_ok))
        for force in FORCES_N:
            for d in ("left", "right"):
                rows.append(row(f"c{i}", force, d, "fresh", i < fresh * n))
                rows.append(row(f"c{i}", force, d, "vla_replay", i < vla * n))
    return rows


def test_cell_result_is_fresh_minus_vla_replay():
    rows = panel(10, fresh=0.8, vla=0.3)
    c = cell_result(rows, 100.0, "left")
    assert abs(c.fresh - 0.8) < 1e-9
    assert abs(c.vla_replay - 0.3) < 1e-9
    assert abs(c.delta_VLA - 0.5) < 1e-9


def test_no_actuator_replay_in_conditions():
    rows = panel(4)
    assert all(r["condition"] in ("fresh", "vla_replay") for r in rows)


def test_every_grid_cell_reported():
    cells = per_cell(panel(4))
    assert len(cells) == 1 + len(FORCES_N) * 2
    assert cells[0].force_n == 0.0


def test_vla_replay_touching_server_is_structural_failure():
    rows = panel(4)
    rows[1]["server_queries"] = 3
    assert any("vla_replay contacted the policy server" in v for v in structural_violations(rows))


def test_fresh_may_query_server():
    rows = panel(4)
    assert structural_violations(rows) == []


def test_fidelity_control_reads_only_zero_force():
    ok = fidelity_control(panel(10, zero_ok=True))
    assert ok["pass"]
    bad = fidelity_control(panel(10, zero_ok=False))
    assert not bad["pass"]


def test_push_ineffective_caught_at_largest_force():
    rows = panel(4)
    for r in rows:
        if r["force_n"] == max(FORCES_N):
            r["push_displacement_m"] = 0.001
    assert not push_effective(rows)["pass"]


def test_structural_failure_blocks_grid():
    rows = panel(zero_ok=False)
    rows[1]["server_queries"] = 4
    res = evaluate(rows)
    assert res["verdict"] == "PREREQUISITE_FAIL_STRUCTURAL"
    assert "grid" not in res


def test_fidelity_failure_blocks_grid():
    res = evaluate(panel(zero_ok=False))
    assert res["verdict"] == "PREREQUISITE_FAIL_REPLAY_FIDELITY"
    assert "grid" not in res


def test_insufficient_configs_blocks_grid():
    res = evaluate(panel(MIN_MATCHED_CONFIGS - 1))
    assert res["verdict"] == "INSUFFICIENT_MATCHED_CONFIGS"
    assert "grid" not in res


def test_consistently_helpful():
    # fresh always wins: delta_VLA positive and large everywhere.
    res = evaluate(panel(fresh=1.0, vla=0.2))
    assert res["verdict"] == "CONSISTENTLY_HELPFUL"


def test_consistently_harmful():
    res = evaluate(panel(fresh=0.2, vla=1.0))
    assert res["verdict"] == "CONSISTENTLY_HARMFUL"


def test_no_established_vla_value_when_flat():
    res = evaluate(panel(fresh=0.5, vla=0.5))
    assert res["verdict"] == "NO_ESTABLISHED_VLA_VALUE"


def test_signed_heterogeneity_requires_hand_construction():
    # Build a panel where left cells favor fresh and right cells favor vla_replay.
    rows = []
    n = MIN_MATCHED_CONFIGS
    for i in range(n):
        rows.append(row(f"c{i}", 0, "none", "fresh", True))
        rows.append(row(f"c{i}", 0, "none", "vla_replay", True))
        for force in FORCES_N:
            rows.append(row(f"c{i}", force, "left", "fresh", i < 0.9 * n))
            rows.append(row(f"c{i}", force, "left", "vla_replay", i < 0.1 * n))
            rows.append(row(f"c{i}", force, "right", "fresh", i < 0.1 * n))
            rows.append(row(f"c{i}", force, "right", "vla_replay", i < 0.9 * n))
    res = evaluate(rows)
    assert res["verdict"] == "SIGNED_HETEROGENEITY"
    signs = {k: v["sign"] for k, v in res["grid"].items()}
    assert any(s == "positive" for s in signs.values())
    assert any(s == "negative" for s in signs.values())


def test_frozen_constants_unchanged():
    assert N_CONFIGS == 28
    assert MIN_MATCHED_CONFIGS == 22
