from pathlib import Path

from g0_core import attribution, clustered_bootstrap, p0_fidelity
from p0_replay_contract import ActuatorReference, ReplayTape, TapeRow, VlaCommand


def row(cfg, force, direction, cond, success):
    return dict(config_id=cfg, force_n=force, direction=direction, condition=cond, success=success)


def test_attribution_decomposes_feedback_ladder():
    rows = []
    for cfg in range(10):
        for cond, succ in [("fresh", 1), ("vla_replay", int(cfg < 8)), ("actuator_replay", int(cfg < 3))]:
            rows.append(row(cfg, 100, "left", cond, succ))
    a = attribution(rows)
    assert a.fresh == 1.0
    assert a.vla_replay == 0.8
    assert a.actuator_replay == 0.3
    assert abs(a.high_level_gain - 0.2) < 1e-12
    assert abs(a.low_level_gain - 0.5) < 1e-12


def test_incomplete_cells_are_not_imputed():
    rows = [row("a", 100, "left", "fresh", 1), row("a", 100, "left", "vla_replay", 1)]
    try:
        attribution(rows)
    except ValueError as exc:
        assert "no complete" in str(exc)
    else:
        raise AssertionError("expected no-complete-cell error")


def test_p0_requires_both_replay_levels_to_match_live():
    rows = []
    for cfg in range(10):
        for cond in ("fresh", "vla_replay", "actuator_replay"):
            rows.append(row(cfg, 0, "none", cond, 1))
    assert p0_fidelity(rows)["pass"]
    rows[-1]["success"] = 0
    assert p0_fidelity(rows)["pass"]
    rows[-4]["success"] = 0
    assert not p0_fidelity(rows)["pass"]


def test_cluster_bootstrap_is_finite():
    rows = []
    for cfg in range(8):
        for force in (50, 100, 150):
            for direction in ("left", "right"):
                for cond, succ in [("fresh", 1), ("vla_replay", int(cfg < 6)), ("actuator_replay", int(cfg < 3))]:
                    rows.append(row(cfg, force, direction, cond, succ))
    ci = clustered_bootstrap(rows, n_boot=200, seed=1)
    assert ci["high_level_gain"][1] <= ci["high_level_gain"][0] <= ci["high_level_gain"][2]
    assert ci["low_level_gain"][1] <= ci["low_level_gain"][0] <= ci["low_level_gain"][2]


def test_tape_round_trip(tmp_path: Path):
    tape = ReplayTape([
        TapeRow(
            0,
            VlaCommand((1.0, 2.0), (0.1, 0.2, 0.3, 0.4), (0.74,)),
            ActuatorReference((1.0, 2.0, 3.0), (0.0,) * 7, (1.0,) * 7),
        )
    ])
    path = tmp_path / "tape.jsonl"
    tape.to_jsonl(path)
    assert tape.exact_equal(ReplayTape.from_jsonl(path))


def test_tape_rejects_noncontiguous_steps():
    try:
        ReplayTape([TapeRow(1, VlaCommand((), (), ()), ActuatorReference((), (), ()))])
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("expected contiguous-step error")
