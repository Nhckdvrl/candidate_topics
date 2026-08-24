"""Pure-logic tests for the frozen P0 replay-fidelity gate.

No simulator, no policy. These pin the semantics the gate is supposed to have,
so a later change to `p0_analyze` cannot silently weaken it.
"""
from p0_analyze import structural_violations, trajectory_fidelity


def rec(cfg, cond, success=True, steps=100, tape_len=100, queries=0, **kw):
    r = dict(
        config_id=cfg, condition=cond, force_n=0.0, direction="none",
        success=success, steps=steps, tape_len=tape_len,
        tape_exhausted_early=False, server_queries=queries,
        effect_qpos=-0.17, terminal_base_xyz=[0.0, 0.0, 0.7],
    )
    r.update(kw)
    return r


def full_panel(n=10, **overrides):
    rows = []
    for i in range(n):
        for cond in ("fresh", "vla_replay", "actuator_replay"):
            rows.append(rec(f"c{i}", cond, **overrides.get(cond, {})))
    return rows


def test_clean_panel_has_no_structural_violation():
    assert structural_violations(full_panel()) == []


def test_replay_touching_the_policy_server_is_a_violation():
    rows = full_panel()
    rows[1]["server_queries"] = 3
    bad = structural_violations(rows)
    assert len(bad) == 1 and "policy server" in bad[0]


def test_fresh_may_query_the_server():
    rows = full_panel()
    for r in rows:
        if r["condition"] == "fresh":
            r["server_queries"] = 12
    assert structural_violations(rows) == []


def test_short_tape_consumption_is_a_violation():
    rows = full_panel()
    rows[1]["steps"] = 90
    bad = structural_violations(rows)
    assert len(bad) == 1 and "90 of 100" in bad[0]


def test_exhausted_tape_is_a_violation():
    rows = full_panel()
    rows[2]["tape_exhausted_early"] = True
    assert any("exhausted" in b for b in structural_violations(rows))


def test_p0_rows_must_be_unperturbed():
    rows = full_panel()
    rows[0]["force_n"] = 100.0
    assert any("unperturbed" in b for b in structural_violations(rows))


def test_trajectory_fidelity_reports_worst_config():
    rows = full_panel(n=3)
    for r in rows:
        if r["config_id"] == "c1" and r["condition"] == "actuator_replay":
            r["effect_qpos"] = 0.30
            r["terminal_base_xyz"] = [0.5, 0.0, 0.7]
    f = trajectory_fidelity(rows, None)
    assert abs(f["vla_replay"]["max_terminal_door_dev_rad"]) < 1e-12
    assert abs(f["actuator_replay"]["max_terminal_door_dev_rad"] - 0.47) < 1e-9
    assert abs(f["actuator_replay"]["max_terminal_base_dev_m"] - 0.5) < 1e-9
