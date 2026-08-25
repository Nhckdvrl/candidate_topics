"""Topic 24 G2: does the navigation-channel reversal hold across the force grid?

G1 factored the VLA command at 100N only and found the left/right sign flip
lives in the navigation/base channel. G2 asks whether that is a property of
100N specifically or holds across G0's whole frozen grid (50N, 150N; 100N is
already done and is reused, not re-run).

This module also carries a deliberate correction to G1's evaluator, recorded
as a correction rather than folded in silently (see G1_RESULTS.md's
post-result audit). `g1_core.py`'s verdict used

    nav_real = any(effect clears bar for k in two contrasts, d in two directions)

which never checked that the `left` and `right` effects were opposite in
sign, so it could not actually distinguish a real reversal from a same-signed
effect that just happened to be smaller in one direction. G2's predicate is
built directly on the single controlled contrast G1's own post-result audit
identified as the clean one:

    N_f,d = S(LR) - S(RR)          upper-body held at replay; navigation is
                                    the only thing that changes
    R_f   = N_f,left - N_f,right   the reversal magnitude at force f

A reversal is established at force f only if N_f,left and N_f,right are each
independently significant (bootstrap CI excludes zero, |point| >= minimum
worthy effect) AND have opposite sign. Same-signed or non-significant results
are reported as exactly that, not folded into a "reversal" label.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

CONDITIONS = ("RR", "LR", "RL", "LL")
FORCES_N = (50.0, 100.0, 150.0)
MIN_MATCHED_CONFIGS = 24
MIN_WORTHY_EFFECT = 0.10


@dataclass(frozen=True)
class ForceDirectionResult:
    force_n: float
    direction: str
    n_configs: int
    RR: float
    LR: float
    N: float   # LR - RR: the primary navigation contrast, upper held at replay


def _triples(rows: Iterable[dict], force_n: float, direction: str) -> list[dict[str, dict]]:
    by_cfg: dict[str, dict[str, dict]] = {}
    for r in rows:
        if str(r["direction"]) != direction or float(r["force_n"]) != force_n:
            continue
        cond = str(r["condition"])
        if cond not in ("RR", "LR"):
            continue
        slot = by_cfg.setdefault(str(r["config_id"]), {})
        if cond in slot:
            raise ValueError(f"duplicate row for {r['config_id']}/{direction}/{cond}")
        slot[cond] = r
    return [v for v in by_cfg.values() if set(v) == {"RR", "LR"}]


def force_direction_result(rows: Iterable[dict], force_n: float, direction: str) -> ForceDirectionResult:
    triples = _triples(rows, force_n, direction)
    if not triples:
        raise ValueError(f"no complete RR/LR pairs for force={force_n}/{direction}")
    RR = float(np.mean([int(bool(t["RR"]["success"])) for t in triples]))
    LR = float(np.mean([int(bool(t["LR"]["success"])) for t in triples]))
    return ForceDirectionResult(
        force_n=force_n, direction=direction, n_configs=len(triples), RR=RR, LR=LR, N=LR - RR,
    )


def bootstrap_N(
    rows: Iterable[dict], force_n: float, direction: str, *, n_boot: int = 10000, seed: int = 20260825
) -> tuple[float, float, float]:
    triples = _triples(rows, force_n, direction)
    point = force_direction_result(rows, force_n, direction).N
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(triples), size=len(triples))
        RR = np.mean([int(bool(triples[i]["RR"]["success"])) for i in idx])
        LR = np.mean([int(bool(triples[i]["LR"]["success"])) for i in idx])
        boot.append(LR - RR)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def reversal_at_force(rows: Iterable[dict], force_n: float) -> dict:
    """R_f and whether the frozen reversal predicate is satisfied at this force."""
    rows = list(rows)
    left = force_direction_result(rows, force_n, "left")
    right = force_direction_result(rows, force_n, "right")
    left_ci = bootstrap_N(rows, force_n, "left")
    right_ci = bootstrap_N(rows, force_n, "right")

    def significant(ci: tuple[float, float, float]) -> bool:
        pt, lo, hi = ci
        return (lo > 0 or hi < 0) and abs(pt) >= MIN_WORTHY_EFFECT

    left_sig, right_sig = significant(left_ci), significant(right_ci)
    opposite_sign = (left_ci[0] > 0) != (right_ci[0] > 0)
    reversal_established = left_sig and right_sig and opposite_sign

    return {
        "force_n": force_n,
        "n_configs": {"left": left.n_configs, "right": right.n_configs},
        "N_left": left_ci, "N_right": right_ci,
        "left_significant": left_sig, "right_significant": right_sig,
        "opposite_sign": opposite_sign,
        "R_f": left_ci[0] - right_ci[0],
        "reversal_established": reversal_established,
    }


def structural_violations(rows: Iterable[dict]) -> list[str]:
    bad: list[str] = []
    for r in rows:
        cond = str(r["condition"])
        force_n = float(r.get("force_n", -1))
        key = f"{r['config_id']}/f{force_n}/{r['direction']}/{cond}"
        if force_n not in FORCES_N:
            bad.append(f"{key}: force_n must be one of {FORCES_N}")
        if cond in ("RR", "LR") and r.get("upper_replayed") is not True:
            bad.append(f"{key}: implies upper channel replayed but row says otherwise")
        if cond in ("RL", "LL") and r.get("upper_replayed") is not False:
            bad.append(f"{key}: implies upper channel live but row says otherwise")
        if cond in ("RR", "RL") and r.get("nav_replayed") is not True:
            bad.append(f"{key}: implies nav channel replayed but row says otherwise")
        if cond in ("LR", "LL") and r.get("nav_replayed") is not False:
            bad.append(f"{key}: implies nav channel live but row says otherwise")
        if cond == "RR" and r.get("server_queries", 0) not in (0, None):
            bad.append(f"{key}: RR contacted the policy server")
        if r.get("source") == "g2_new":
            steps = r.get("steps")
            nav_ovr, up_ovr = r.get("nav_overwrites"), r.get("upper_overwrites")
            if cond in ("LR", "RL") and not r.get("server_queries"):
                bad.append(f"{key}: hybrid condition never queried the live VLA")
            if r.get("nav_replayed") and nav_ovr != steps:
                bad.append(f"{key}: nav channel claimed replayed but overwrote {nav_ovr} of {steps} ticks")
            if not r.get("nav_replayed") and nav_ovr:
                bad.append(f"{key}: nav channel claimed live but was overwritten {nav_ovr} times")
            if r.get("upper_replayed") and up_ovr != steps:
                bad.append(f"{key}: upper channel claimed replayed but overwrote {up_ovr} of {steps} ticks")
            if not r.get("upper_replayed") and up_ovr:
                bad.append(f"{key}: upper channel claimed live but was overwritten {up_ovr} times")
    return bad


def evaluate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    violations = structural_violations(rows)
    if violations:
        return {"structural_violations": violations, "verdict": "PREREQUISITE_FAIL_STRUCTURAL"}

    per_force: dict[float, dict] = {}
    for f in FORCES_N:
        try:
            r = reversal_at_force(rows, f)
        except ValueError:
            continue
        if min(r["n_configs"].values()) < MIN_MATCHED_CONFIGS:
            continue
        per_force[f] = r

    if len(per_force) < len(FORCES_N):
        missing = [f for f in FORCES_N if f not in per_force]
        return {
            "structural_violations": [],
            "verdict": "INSUFFICIENT_MATCHED_CONFIGS",
            "missing_forces": missing,
            "per_force": per_force,
        }

    established = [f for f, r in per_force.items() if r["reversal_established"]]
    if len(established) == len(FORCES_N):
        verdict = "REVERSAL_CONFIRMED_ACROSS_FORCE_GRID"
    elif len(established) > 0:
        verdict = "REVERSAL_CONFIRMED_AT_SOME_FORCES_ONLY"
    else:
        verdict = "REVERSAL_NOT_ESTABLISHED_OUTSIDE_100N"

    return {
        "structural_violations": [],
        "verdict": verdict,
        "reversal_established_at": established,
        "per_force": {f: r for f, r in per_force.items()},
    }


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("records", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    result = evaluate(load_jsonl(args.records))
    print(json.dumps(result, indent=2, default=str))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
