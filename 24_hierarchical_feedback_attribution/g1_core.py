"""Topic 24 G1: which VLA command channel causes the right-push reversal?

G0 found `delta_high` pooled to near zero, but the six per-cell values are not
individually small: all three `left`-push cells are positive, all three
`right`-push cells are negative. `vla_replay` (both channels replayed) beats
`fresh` (both channels live) specifically under a `right` push. G1 asks which
half of the VLA command carries that sign flip, by factoring the seam itself
rather than inspecting hidden state.

The VLA command has two independently addressable parts:

    navigate_cmd, base_height_command      (navigation/base channel)
    target_upper_body_pose                 (upper-body channel)

Four conditions, replaying each channel independently:

    RR   both channels replayed         == G0's `vla_replay`
    LR   navigation live, upper replayed
    RL   navigation replayed, upper live
    LL   both channels live             == G0's `fresh`

RR and LL are not re-collected: they are the existing G0 `vla_replay` and
`fresh` rows at force=100N, read from the frozen G0 records. Only LR and RL
are new data.

Frozen operating point: 100N, both directions, the same 30 configs as G0.
Chosen because it is the diagnostic point in G0 with the largest, and
oppositely signed, delta_high across directions (+0.233 left, -0.200 right) —
not because it looks best. Not tuned after seeing G1 data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

CONDITIONS = ("RR", "LR", "RL", "LL")
FORCE_N = 100.0
MIN_MATCHED_CONFIGS = 24
MIN_WORTHY_EFFECT = 0.10


@dataclass(frozen=True)
class DirectionResult:
    direction: str
    n_configs: int
    RR: float
    LR: float
    RL: float
    LL: float
    nav_effect_upper_replayed: float   # LR - RR
    nav_effect_upper_live: float       # LL - RL
    upper_effect_nav_replayed: float   # RL - RR
    upper_effect_nav_live: float       # LL - LR


def _triples(rows: Iterable[dict], direction: str) -> list[dict[str, dict]]:
    by_cfg: dict[str, dict[str, dict]] = {}
    for r in rows:
        if str(r["direction"]) != direction or float(r["force_n"]) != FORCE_N:
            continue
        cond = str(r["condition"])
        if cond not in CONDITIONS:
            continue
        slot = by_cfg.setdefault(str(r["config_id"]), {})
        if cond in slot:
            raise ValueError(f"duplicate row for {r['config_id']}/{direction}/{cond}")
        slot[cond] = r
    return [v for v in by_cfg.values() if set(v) == set(CONDITIONS)]


def direction_result(rows: Iterable[dict], direction: str) -> DirectionResult:
    triples = _triples(rows, direction)
    if not triples:
        raise ValueError(f"no complete RR/LR/RL/LL quadruples for direction={direction}")
    m = {
        c: float(np.mean([int(bool(t[c]["success"])) for t in triples]))
        for c in CONDITIONS
    }
    return DirectionResult(
        direction=direction,
        n_configs=len(triples),
        RR=m["RR"], LR=m["LR"], RL=m["RL"], LL=m["LL"],
        nav_effect_upper_replayed=m["LR"] - m["RR"],
        nav_effect_upper_live=m["LL"] - m["RL"],
        upper_effect_nav_replayed=m["RL"] - m["RR"],
        upper_effect_nav_live=m["LL"] - m["LR"],
    )


def clustered_bootstrap(
    rows: Iterable[dict], direction: str, *, n_boot: int = 10000, seed: int = 20260825
) -> dict[str, tuple[float, float, float]]:
    triples = _triples(rows, direction)
    configs = [f"c{i}" for i in range(len(triples))]
    point = direction_result(rows, direction)
    rng = np.random.default_rng(seed)
    keys = ("nav_effect_upper_replayed", "nav_effect_upper_live",
            "upper_effect_nav_replayed", "upper_effect_nav_live")
    boot: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(triples), size=len(triples))
        m = {
            c: float(np.mean([int(bool(triples[i][c]["success"])) for i in idx]))
            for c in CONDITIONS
        }
        boot["nav_effect_upper_replayed"].append(m["LR"] - m["RR"])
        boot["nav_effect_upper_live"].append(m["LL"] - m["RL"])
        boot["upper_effect_nav_replayed"].append(m["RL"] - m["RR"])
        boot["upper_effect_nav_live"].append(m["LL"] - m["LR"])

    def pack(pt: float, xs: list[float]) -> tuple[float, float, float]:
        lo, hi = np.quantile(xs, [0.025, 0.975])
        return float(pt), float(lo), float(hi)

    pt = asdict(point)
    return {k: pack(pt[k], boot[k]) for k in keys}


def structural_violations(rows: Iterable[dict]) -> list[str]:
    bad: list[str] = []
    for r in rows:
        cond = str(r["condition"])
        key = f"{r['config_id']}/{r['direction']}/{cond}"
        if float(r.get("force_n", -1)) != FORCE_N:
            bad.append(f"{key}: G1 only runs force_n={FORCE_N}, got {r.get('force_n')}")
        if cond in ("RR", "LR") and r.get("upper_replayed") is not True:
            bad.append(f"{key}: condition implies upper channel replayed but row says otherwise")
        if cond in ("RL", "LL") and r.get("upper_replayed") is not False:
            bad.append(f"{key}: condition implies upper channel live but row says otherwise")
        if cond in ("RR", "RL") and r.get("nav_replayed") is not True:
            bad.append(f"{key}: condition implies nav channel replayed but row says otherwise")
        if cond in ("LR", "LL") and r.get("nav_replayed") is not False:
            bad.append(f"{key}: condition implies nav channel live but row says otherwise")
        if cond == "RR" and r.get("server_queries", 0) not in (0, None):
            # RR is reused from G0's vla_replay, where the VLA is never queried.
            bad.append(f"{key}: RR contacted the policy server")
        # Newly-collected hybrid rows must show the intervention actually fired
        # on every control tick, not merely that it was requested. Rows reused
        # from G0 carry no overwrite counters and are exempt.
        if r.get("source") == "g1_new":
            steps = r.get("steps")
            nav_ovr = r.get("nav_overwrites")
            up_ovr = r.get("upper_overwrites")
            if cond in ("LR", "RL") and not r.get("server_queries"):
                bad.append(f"{key}: hybrid condition never queried the live VLA")
            if r.get("nav_replayed") and nav_ovr != steps:
                bad.append(
                    f"{key}: nav channel claimed replayed but overwrote "
                    f"{nav_ovr} of {steps} ticks")
            if not r.get("nav_replayed") and nav_ovr:
                bad.append(f"{key}: nav channel claimed live but was overwritten {nav_ovr} times")
            if r.get("upper_replayed") and up_ovr != steps:
                bad.append(
                    f"{key}: upper channel claimed replayed but overwrote "
                    f"{up_ovr} of {steps} ticks")
            if not r.get("upper_replayed") and up_ovr:
                bad.append(f"{key}: upper channel claimed live but was overwritten {up_ovr} times")
    return bad


def evaluate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    violations = structural_violations(rows)
    result: dict = {"structural_violations": violations}
    if violations:
        result["verdict"] = "PREREQUISITE_FAIL_STRUCTURAL"
        return result

    directions: dict[str, DirectionResult] = {}
    for d in ("left", "right"):
        try:
            directions[d] = direction_result(rows, d)
        except ValueError:
            pass

    if set(directions) != {"left", "right"}:
        result["verdict"] = "INSUFFICIENT_MATCHED_CONFIGS"
        result["directions"] = {d: asdict(r) for d, r in directions.items()}
        return result
    if any(r.n_configs < MIN_MATCHED_CONFIGS for r in directions.values()):
        result["verdict"] = "INSUFFICIENT_MATCHED_CONFIGS"
        result["directions"] = {d: asdict(r) for d, r in directions.items()}
        return result

    result["directions"] = {d: asdict(r) for d, r in directions.items()}
    result["bootstrap"] = {d: clustered_bootstrap(rows, d) for d in ("left", "right")}

    def real(effect_key: str, direction: str) -> bool:
        pt, lo, hi = result["bootstrap"][direction][effect_key]
        return (lo > 0 or hi < 0) and abs(pt) >= MIN_WORTHY_EFFECT

    nav_real = any(real(k, d) for k in ("nav_effect_upper_replayed", "nav_effect_upper_live")
                    for d in ("left", "right"))
    upper_real = any(real(k, d) for k in ("upper_effect_nav_replayed", "upper_effect_nav_live")
                      for d in ("left", "right"))

    if nav_real and not upper_real:
        result["verdict"] = "NAVIGATION_CHANNEL_CAUSES_REVERSAL"
    elif upper_real and not nav_real:
        result["verdict"] = "UPPER_BODY_CHANNEL_CAUSES_REVERSAL"
    elif nav_real and upper_real:
        result["verdict"] = "BOTH_CHANNELS_CONTRIBUTE"
    else:
        result["verdict"] = "CROSS_CHANNEL_INTERACTION_OR_NO_SINGLE_CHANNEL_EFFECT"
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
