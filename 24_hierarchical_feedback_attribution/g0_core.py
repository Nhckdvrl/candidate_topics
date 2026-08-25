"""Topic 24 G0 statistics: three-level feedback attribution on a matched panel.

Frozen before data collection. Two primary quantities, one per seam that exists
in the released source:

    delta_high = S_fresh      - S_vla_replay        VLA-level online feedback
    delta_low  = S_vla_replay - S_actuator_replay   WBC / reference-generation feedback
    residual   = S_actuator_replay                  servo + mechanics + task tolerance

`delta_low` is deliberately *not* called the low-level controller contribution.
`actuator_replay` is still a closed loop below the seam it cuts: joint servo/PD
feedback, actuator dynamics, passive mechanical stabilization and task tolerance
all survive it. P0b narrows the term further — below the VLA seam the arms and
hands are open-loop interpolation, so `delta_low` can only carry
locomotion/balance state feedback.

Every force cell of the grid is reported. No cell is selected after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

CONDITIONS = ("fresh", "vla_replay", "actuator_replay")

# --- frozen panel ----------------------------------------------------------
FORCES_N = (50.0, 100.0, 150.0)
DIRECTIONS = ("left", "right")
PUSH_DURATION_S = 0.2
N_CONFIGS = 30
# A cell needs this many complete config triples to be read at all.
MIN_MATCHED_CONFIGS = 24
# Replay fidelity is re-verified inside G0, on the force=0 control column, under
# the exact G0 code path including tape extension. Same numbers as the P0 gate.
FIDELITY_MIN_SUCCESS = 0.90
FIDELITY_MAX_DROP = 0.10
# Minimum contribution worth reporting as a contribution, pooled over the grid.
MIN_WORTHY_DELTA = 0.10
# A push that does not move the robot is not an intervention.
MIN_PUSH_DISPLACEMENT_M = 0.02
# Above this, `fresh` never needed feedback recovery anywhere on the grid.
NO_PHENOMENON_FLOOR = 0.90
# Below this, `fresh` has collapsed and there is nothing to attribute.
FRESH_COLLAPSE_CEILING = 0.10


@dataclass(frozen=True)
class CellResult:
    force_n: float
    direction: str
    n_configs: int
    fresh: float
    vla_replay: float
    actuator_replay: float
    delta_high: float
    delta_low: float
    residual: float


@dataclass(frozen=True)
class Attribution:
    fresh: float
    vla_replay: float
    actuator_replay: float
    delta_high: float
    delta_low: float
    residual: float
    n_cells: int


def _triples(rows: Iterable[dict]) -> list[dict[str, dict]]:
    """Group rows into complete (config, force, direction) condition triples.

    Incomplete groups are dropped, never imputed.
    """
    by_key: dict[tuple[str, float, str], dict[str, dict]] = {}
    for r in rows:
        cond = str(r["condition"])
        if cond not in CONDITIONS:
            continue
        key = (str(r["config_id"]), float(r["force_n"]), str(r["direction"]))
        slot = by_key.setdefault(key, {})
        if cond in slot:
            raise ValueError(f"duplicate row for {key} / {cond}")
        slot[cond] = r
    return [v for v in by_key.values() if set(v) == set(CONDITIONS)]


def attribution(rows: Iterable[dict]) -> Attribution:
    triples = _triples(rows)
    if not triples:
        raise ValueError("no complete fresh/vla_replay/actuator_replay triples")
    m = {
        c: float(np.mean([int(bool(t[c]["success"])) for t in triples]))
        for c in CONDITIONS
    }
    return Attribution(
        fresh=m["fresh"],
        vla_replay=m["vla_replay"],
        actuator_replay=m["actuator_replay"],
        delta_high=m["fresh"] - m["vla_replay"],
        delta_low=m["vla_replay"] - m["actuator_replay"],
        residual=m["actuator_replay"],
        n_cells=len(triples),
    )


def per_cell(rows: Iterable[dict]) -> list[CellResult]:
    """Report every force/direction cell of the grid, in a fixed order."""
    rows = list(rows)
    cells: list[CellResult] = []
    keys = sorted({(float(r["force_n"]), str(r["direction"])) for r in rows})
    for force, direction in keys:
        sub = [r for r in rows if float(r["force_n"]) == force and str(r["direction"]) == direction]
        try:
            a = attribution(sub)
        except ValueError:
            continue
        cells.append(CellResult(
            force_n=force, direction=direction, n_configs=a.n_cells,
            fresh=a.fresh, vla_replay=a.vla_replay, actuator_replay=a.actuator_replay,
            delta_high=a.delta_high, delta_low=a.delta_low, residual=a.residual,
        ))
    return cells


def clustered_bootstrap(
    rows: Iterable[dict], *, n_boot: int = 10000, seed: int = 20260824
) -> dict[str, tuple[float, float, float]]:
    """Resample physical configs, keeping each config's whole force panel intact.

    Configs are the unit of independence: the same scene draw contributes one row
    per force cell per condition, so resampling cells would understate the CI.
    """
    rows = list(rows)
    configs = sorted({str(r["config_id"]) for r in rows})
    if len(configs) < 2:
        raise ValueError("need >=2 config clusters")
    by_cfg = {c: [r for r in rows if str(r["config_id"]) == c] for c in configs}
    point = attribution(rows)
    rng = np.random.default_rng(seed)
    hi, lo = [], []
    for _ in range(n_boot):
        boot: list[dict] = []
        for j, c in enumerate(rng.choice(configs, size=len(configs), replace=True)):
            for r in by_cfg[str(c)]:
                x = dict(r)
                x["config_id"] = f"{c}__boot{j}"
                boot.append(x)
        a = attribution(boot)
        hi.append(a.delta_high)
        lo.append(a.delta_low)

    def pack(pt: float, xs: list[float]) -> tuple[float, float, float]:
        a, b = np.quantile(xs, [0.025, 0.975])
        return float(pt), float(a), float(b)

    return {
        "delta_high": pack(point.delta_high, hi),
        "delta_low": pack(point.delta_low, lo),
    }


def fidelity_control(rows: Iterable[dict]) -> dict:
    """Re-verify P0 replay fidelity on the force=0 column of this very panel."""
    zero = [r for r in rows if float(r["force_n"]) == 0.0]
    if not _triples(zero):
        return {"present": False, "pass": False}
    a = attribution(zero)
    ok = (
        min(a.fresh, a.vla_replay, a.actuator_replay) >= FIDELITY_MIN_SUCCESS
        and (a.fresh - a.vla_replay) <= FIDELITY_MAX_DROP
        and (a.fresh - a.actuator_replay) <= FIDELITY_MAX_DROP
    )
    return {
        "present": True, "n_configs": a.n_cells, "fresh": a.fresh,
        "vla_replay": a.vla_replay, "actuator_replay": a.actuator_replay,
        "pass": bool(ok),
    }


def structural_violations(rows: Iterable[dict]) -> list[str]:
    """Contracts, not thresholds. None of these may be relaxed after seeing data."""
    bad: list[str] = []
    by_key: dict[tuple[str, float, str], list[dict]] = {}
    for r in rows:
        key = f"{r['config_id']}/f{r['force_n']}{r['direction']}/{r['condition']}"
        if r["condition"] != "fresh" and r.get("server_queries", 0) != 0:
            bad.append(f"{key}: replay contacted the policy server")
        if r["condition"] != "fresh" and r.get("tape_exhausted_early"):
            bad.append(f"{key}: tape exhausted before the horizon")
        if float(r["force_n"]) > 0.0 and not r.get("push_applied"):
            bad.append(f"{key}: nonzero force cell recorded no applied push")
        if float(r["force_n"]) == 0.0 and r.get("push_applied"):
            bad.append(f"{key}: control cell recorded an applied push")
        by_key.setdefault(
            (str(r["config_id"]), float(r["force_n"]), str(r["direction"])), []
        ).append(r)
    # The same disturbance must be delivered identically in all three conditions.
    # Only meaningful when a push actually exists: on the force=0 control column,
    # `fresh` records push_tick=None (nothing to derive it from) while the two
    # replay conditions still carry the tape's recorded tick even though no push
    # is applied there, so the two are not comparable and not a contract at all.
    for (cfg, force, direction), group in by_key.items():
        if force <= 0.0:
            continue
        ticks = {r.get("push_tick") for r in group}
        if len(ticks) > 1:
            bad.append(f"{cfg}/f{force}{direction}: push tick differs across conditions {sorted(ticks)}")
    return bad


def push_effective(rows: Iterable[dict]) -> dict:
    """Did the largest force actually move the robot? Structural, not scientific."""
    top = max((float(r["force_n"]) for r in rows), default=0.0)
    sub = [
        r for r in rows
        if float(r["force_n"]) == top and r.get("push_displacement_m") is not None
    ]
    if top == 0.0 or not sub:
        return {"present": False, "pass": False, "force_n": top}
    med = float(np.median([float(r["push_displacement_m"]) for r in sub]))
    return {
        "present": True, "force_n": top, "median_displacement_m": med,
        "pass": bool(med >= MIN_PUSH_DISPLACEMENT_M),
    }


def evaluate(rows: Iterable[dict], *, n_boot: int = 10000) -> dict:
    """Frozen decision procedure. Prerequisites are checked before any delta is read."""
    rows = list(rows)
    cells = per_cell(rows)
    perturbed = [r for r in rows if float(r["force_n"]) > 0.0]
    violations = structural_violations(rows)
    fidelity = fidelity_control(rows)
    push = push_effective(rows)
    n_cfg_ok = all(c.n_configs >= MIN_MATCHED_CONFIGS for c in cells) if cells else False

    verdict: str | None = None
    if violations:
        verdict = "PREREQUISITE_FAIL_STRUCTURAL"
    elif not fidelity["pass"]:
        verdict = "PREREQUISITE_FAIL_REPLAY_FIDELITY"
    elif not push["pass"]:
        verdict = "PREREQUISITE_FAIL_PUSH_INEFFECTIVE"
    elif not cells or not n_cfg_ok:
        verdict = "INSUFFICIENT_MATCHED_CONFIGS"

    result: dict = {
        "verdict": verdict,
        "structural_violations": violations,
        "fidelity_control": fidelity,
        "push_effective": push,
        "per_cell": [asdict(c) for c in cells],
    }
    if verdict is not None:
        return result

    pooled = attribution(perturbed)
    result["pooled_perturbed"] = asdict(pooled)
    result["bootstrap"] = clustered_bootstrap(perturbed, n_boot=n_boot)
    pcells = [c for c in cells if c.force_n > 0.0]

    if all(c.fresh >= NO_PHENOMENON_FLOOR and c.actuator_replay >= NO_PHENOMENON_FLOOR
           for c in pcells):
        result["verdict"] = "NO_ROBUSTNESS_PHENOMENON"
        return result
    if all(c.fresh <= FRESH_COLLAPSE_CEILING for c in pcells):
        result["verdict"] = "FRESH_COLLAPSE_NOTHING_TO_ATTRIBUTE"
        return result

    hi_pt, hi_lo, _ = result["bootstrap"]["delta_high"]
    lo_pt, lo_lo, _ = result["bootstrap"]["delta_low"]
    high_real = hi_lo > 0.0 and hi_pt >= MIN_WORTHY_DELTA
    low_real = lo_lo > 0.0 and lo_pt >= MIN_WORTHY_DELTA
    result["verdict"] = {
        (True, True): "BOTH_LEVELS_CONTRIBUTE",
        (True, False): "VLA_LEVEL_DOMINATES",
        (False, True): "WBC_LEVEL_DOMINATES",
        (False, False): "NO_MEANINGFUL_LEARNED_FEEDBACK_CONTRIBUTION",
    }[(high_real, low_real)]
    return result


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("records", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    result = evaluate(load_jsonl(args.records))
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
