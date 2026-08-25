"""Topic 24 G3 (corrected): does live VLA feedback have reliably positive
causal value under disturbance, with a capable WBC left online in both arms?

    delta_VLA(f, d) = S_fresh(f, d) - S_vla_replay(f, d)

fresh:      live observation -> live VLA -> live WBC -> robot
vla_replay: recorded pre-disturbance VLA plan -> live WBC (live proprio) -> robot

Both conditions leave the WBC fully live; only whether the VLA gets to
re-observe and re-plan differs. P0'/P0b' already established (before this
panel ran) that vla_replay is lossless under no disturbance and that the WBC
seam is genuinely state-dependent on this task, so the comparison means
exactly what it claims to mean without re-deriving anything here.

No pooled pre-summary: the full force x direction grid is read first, per the
lesson G0's own pooled delta_high near-zero taught this project the hard way.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

CONDITIONS = ("fresh", "vla_replay")
FORCES_N = (50.0, 100.0, 150.0)
DIRECTIONS = ("left", "right")
N_CONFIGS = 28
MIN_MATCHED_CONFIGS = 22  # ~0.79 of 28, matches G0's 24/30 proportion
MIN_WORTHY_EFFECT = 0.10
MIN_PUSH_DISPLACEMENT_M = 0.02
FIDELITY_MIN_SUCCESS = 0.90
FIDELITY_MAX_DROP = 0.10


@dataclass(frozen=True)
class CellResult:
    force_n: float
    direction: str
    n_configs: int
    fresh: float
    vla_replay: float
    delta_VLA: float


def _pairs(rows: Iterable[dict], force_n: float, direction: str) -> list[dict[str, dict]]:
    by_cfg: dict[str, dict[str, dict]] = {}
    for r in rows:
        if float(r["force_n"]) != force_n or str(r["direction"]) != direction:
            continue
        cond = str(r["condition"])
        if cond not in CONDITIONS:
            continue
        slot = by_cfg.setdefault(str(r["config_id"]), {})
        if cond in slot:
            raise ValueError(f"duplicate row for {r['config_id']}/{force_n}/{direction}/{cond}")
        slot[cond] = r
    return [v for v in by_cfg.values() if set(v) == set(CONDITIONS)]


def cell_result(rows: Iterable[dict], force_n: float, direction: str) -> CellResult:
    pairs = _pairs(rows, force_n, direction)
    if not pairs:
        raise ValueError(f"no complete fresh/vla_replay pairs for {force_n}N/{direction}")
    fresh = float(np.mean([int(bool(p["fresh"]["success"])) for p in pairs]))
    vla = float(np.mean([int(bool(p["vla_replay"]["success"])) for p in pairs]))
    return CellResult(force_n, direction, len(pairs), fresh, vla, fresh - vla)


def per_cell(rows: Iterable[dict]) -> list[CellResult]:
    rows = list(rows)
    out = []
    for force_n in (0.0,) + FORCES_N:
        directions = ("none",) if force_n == 0.0 else DIRECTIONS
        for direction in directions:
            try:
                out.append(cell_result(rows, force_n, direction))
            except ValueError:
                continue
    return out


def clustered_bootstrap(
    rows: Iterable[dict], force_n: float, direction: str, *, n_boot: int = 10000, seed: int = 20260825
) -> tuple[float, float, float]:
    pairs = _pairs(rows, force_n, direction)
    point = cell_result(rows, force_n, direction).delta_VLA
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pairs), size=len(pairs))
        fresh = np.mean([int(bool(pairs[i]["fresh"]["success"])) for i in idx])
        vla = np.mean([int(bool(pairs[i]["vla_replay"]["success"])) for i in idx])
        boot.append(fresh - vla)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def structural_violations(rows: Iterable[dict]) -> list[str]:
    bad: list[str] = []
    by_key: dict[tuple[str, float, str], list[dict]] = {}
    for r in rows:
        key = f"{r['config_id']}/f{r['force_n']}{r['direction']}/{r['condition']}"
        if r["condition"] == "vla_replay" and r.get("server_queries", 0) != 0:
            bad.append(f"{key}: vla_replay contacted the policy server")
        if r["condition"] == "vla_replay" and r.get("tape_exhausted_early"):
            bad.append(f"{key}: tape exhausted before the horizon")
        if float(r["force_n"]) > 0.0 and not r.get("push_applied"):
            bad.append(f"{key}: nonzero force cell recorded no applied push")
        if float(r["force_n"]) == 0.0 and r.get("push_applied"):
            bad.append(f"{key}: control cell recorded an applied push")
        by_key.setdefault(
            (str(r["config_id"]), float(r["force_n"]), str(r["direction"])), []
        ).append(r)
    for (cfg, force, direction), group in by_key.items():
        if force <= 0.0:
            continue
        ticks = {r.get("push_tick") for r in group}
        if len(ticks) > 1:
            bad.append(f"{cfg}/f{force}{direction}: push tick differs between conditions {sorted(ticks)}")
    return bad


def fidelity_control(rows: Iterable[dict]) -> dict:
    zero = [r for r in rows if float(r["force_n"]) == 0.0]
    if not _pairs(zero, 0.0, "none"):
        return {"present": False, "pass": False}
    c = cell_result(zero, 0.0, "none")
    ok = (
        min(c.fresh, c.vla_replay) >= FIDELITY_MIN_SUCCESS
        and (c.fresh - c.vla_replay) <= FIDELITY_MAX_DROP
    )
    return {
        "present": True, "n_configs": c.n_configs,
        "fresh": c.fresh, "vla_replay": c.vla_replay, "pass": bool(ok),
    }


def push_effective(rows: Iterable[dict]) -> dict:
    top = max((float(r["force_n"]) for r in rows), default=0.0)
    sub = [r for r in rows if float(r["force_n"]) == top and r.get("push_displacement_m") is not None]
    if top == 0.0 or not sub:
        return {"present": False, "pass": False, "force_n": top}
    med = float(np.median([float(r["push_displacement_m"]) for r in sub]))
    return {"present": True, "force_n": top, "median_displacement_m": med,
            "pass": bool(med >= MIN_PUSH_DISPLACEMENT_M)}


def evaluate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    violations = structural_violations(rows)
    fidelity = fidelity_control(rows)
    push = push_effective(rows)
    cells = per_cell(rows)
    perturbed_cells = [c for c in cells if c.force_n > 0.0]
    n_cfg_ok = all(c.n_configs >= MIN_MATCHED_CONFIGS for c in cells) if cells else False

    result: dict = {
        "structural_violations": violations,
        "fidelity_control": fidelity,
        "push_effective": push,
        "per_cell": [asdict(c) for c in cells],
    }

    if violations:
        result["verdict"] = "PREREQUISITE_FAIL_STRUCTURAL"
        return result
    if not fidelity["pass"]:
        result["verdict"] = "PREREQUISITE_FAIL_REPLAY_FIDELITY"
        return result
    if not push["pass"]:
        result["verdict"] = "PREREQUISITE_FAIL_PUSH_INEFFECTIVE"
        return result
    if not cells or not n_cfg_ok:
        result["verdict"] = "INSUFFICIENT_MATCHED_CONFIGS"
        return result

    grid = {}
    for c in perturbed_cells:
        pt, lo, hi = clustered_bootstrap(rows, c.force_n, c.direction)
        real = (lo > 0 or hi < 0) and abs(pt) >= MIN_WORTHY_EFFECT
        grid[(c.force_n, c.direction)] = {
            "point": pt, "ci_lo": lo, "ci_hi": hi, "established": bool(real),
            "sign": ("positive" if pt > 0 else "negative" if pt < 0 else "zero"),
        }
    result["grid"] = {f"{f}N/{d}": v for (f, d), v in grid.items()}

    established = [v for v in grid.values() if v["established"]]
    pos = [v for v in established if v["point"] > 0]
    neg = [v for v in established if v["point"] < 0]

    if not established:
        verdict = "NO_ESTABLISHED_VLA_VALUE"
    elif pos and neg:
        verdict = "SIGNED_HETEROGENEITY"
    elif neg and not pos:
        verdict = "CONSISTENTLY_HARMFUL"
    else:
        verdict = "CONSISTENTLY_HELPFUL"
    result["verdict"] = verdict
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
