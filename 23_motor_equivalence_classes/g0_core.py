"""Outcome-space G0 utilities for Topic 23.

The scientific question is whether a pretrained robot policy preserves a task effect
after the demonstrator's canonical right-side motor route is removed.

Important: the primary endpoint is environment success/effect, not a projection in
joint space. This is deliberate after the Topic 19 identification failure.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


class Condition(str, Enum):
    """The frozen condition panel.

    Revised 2026-08-24 after reading the upstream task and running a route probe.
    The original four-condition panel could not identify motor substitution, because
    a `right_disabled` success is consistent with three different worlds and only one
    of them is motor equivalence:

      W1  the policy re-planned the task onto another effector       (the claim)
      W2  the right arm was never articulating; the hand was a passive
          bumper carried into the object by locomotion               (RIGHT_FROZEN)
      W3  no arm was needed at all; the torso/base does the work     (BOTH_ARMS_DISABLED)

    RIGHT_FROZEN and BOTH_ARMS_DISABLED are the conditions that separate them.
    """

    CANONICAL = "canonical"
    # Locked-joint fault: right arm+hand held at the pose it already had. Removes the
    # arm's *articulation* but not the arm as an effector.
    RIGHT_FROZEN = "right_frozen"
    # Effector removal: right arm+hand retracted to the neutral at-side pose and held.
    RIGHT_DISABLED = "right_disabled"
    LEFT_DISABLED = "left_disabled"
    BOTH_ARMS_DISABLED = "both_arms_disabled"
    FULL_HOLD = "full_hold"
    ORACLE_RIGHT_DISABLED = "oracle_right_disabled"


#: Conditions that define a matched unit. The oracle is deliberately excluded.
POLICY_CONDITIONS = (
    Condition.CANONICAL,
    Condition.RIGHT_FROZEN,
    Condition.RIGHT_DISABLED,
    Condition.LEFT_DISABLED,
    Condition.BOTH_ARMS_DISABLED,
    Condition.FULL_HOLD,
)




@dataclass(frozen=True)
class GateConfig:
    min_matched_configs: int = 20
    canonical_min_success: float = 0.70
    # Freezing the right arm in place must actually cost the policy something.
    # If it does not, the canonical route contains no right-arm motor program and
    # there is nothing for an equivalent route to substitute for.
    min_arm_program_cost: float = 0.20
    # The canonical route must run through the right side in the first place.
    canonical_right_route_min: float = 0.70
    oracle_min_success: float = 0.70
    full_hold_max_success: float = 0.10
    # If the task survives removing both arms, success is a body/base route and
    # says nothing about substituting one arm for another.
    both_arms_disabled_max_success: float = 0.10
    # The retract-and-hold clamp must actually hold (max joint deviation, rad).
    max_clamp_leak_rad: float = 0.20
    min_substitution_rate: float = 0.20
    min_substitution_events: int = 5
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260824


# NOTE: the pre-controller action-group intervention that used to live here was
# removed on 2026-08-24. The clamp is now applied at the actuator boundary, after
# the GR00T whole-body controller, in `topic23_runner.MotorClamp`. Editing the
# policy's action groups before the WBC let the controller re-solve around the
# constraint, so the limb was not reliably held.


def task_effect_success(task: str, effect_qpos: float) -> bool:
    """Frozen SIMPLE object-state success checks used by the first two tasks."""
    name = task.lower()
    q = float(effect_qpos)
    if "close_door" in name or name == "close_door":
        return q < -0.16
    if "open_faucet" in name or name == "open_faucet":
        return q > 0.7 or q < -0.7
    raise KeyError(f"unsupported frozen task: {task}")


def realized_motion_l2(states: Iterable[Mapping[str, Any]], group: str) -> float:
    """Secondary route diagnostic from realized proprioception, not action targets."""
    seq = [np.asarray(s[group], dtype=float).reshape(-1) for s in states]
    if len(seq) < 2:
        return 0.0
    return float(sum(np.linalg.norm(b - a) for a, b in zip(seq[:-1], seq[1:])))


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _bootstrap_mean_ci(values: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def analyze_records(rows: list[dict[str, Any]], cfg: GateConfig = GateConfig()) -> dict[str, Any]:
    """Analyze matched-condition JSON rows.

    Required row fields:
      config_id, task, condition, success
    Optional:
      effect_qpos, route_verified, canonical_right_route, right_arm_clamp_leak_rad,
      left_arm_motion_l2, torso_motion_l2

    Only configs present in all six frozen *policy* conditions are primary units;
    the scripted oracle is a separate prerequisite and may be absent.
    """
    required = {"config_id", "task", "condition", "success"}
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {i} missing {sorted(missing)}")
        Condition(row["condition"])

    by_cfg: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["config_id"]))
        cond = str(row["condition"])
        if cond in by_cfg.setdefault(key, {}):
            raise ValueError(f"duplicate row for {key} / {cond}")
        by_cfg[key][cond] = row

    # Matched units are defined by the *policy* conditions. The oracle is a
    # scripted feasibility prerequisite that is run separately and later, so a
    # missing oracle must not silently zero out the matched-config count.
    needed = {c.value for c in POLICY_CONDITIONS}
    matched = {k: v for k, v in by_cfg.items() if needed <= set(v)}
    n = len(matched)

    cond_success: dict[str, float] = {}
    for c in POLICY_CONDITIONS:
        cond_success[c.value] = _mean(
            [float(bool(v[c.value]["success"])) for v in matched.values()]
        )

    canonical = cond_success[Condition.CANONICAL.value]
    oracle_rows = [
        v[Condition.ORACLE_RIGHT_DISABLED.value]
        for v in matched.values()
        if Condition.ORACLE_RIGHT_DISABLED.value in v
    ]
    oracle = (
        _mean([float(bool(r["success"])) for r in oracle_rows]) if oracle_rows else None
    )
    full_hold = cond_success[Condition.FULL_HOLD.value]
    right_disabled = cond_success[Condition.RIGHT_DISABLED.value]
    right_frozen = cond_success[Condition.RIGHT_FROZEN.value]
    both_arms = cond_success[Condition.BOTH_ARMS_DISABLED.value]

    # How much does removing only the right arm's *articulation* cost?
    arm_program_cost = canonical - right_frozen

    # Among canonical successes, did the right side actually touch the object when
    # the task predicate was first satisfied?
    canonical_hits = [
        v[Condition.CANONICAL.value] for v in matched.values()
        if bool(v[Condition.CANONICAL.value]["success"])
    ]
    right_route_flags = [
        bool(r.get("canonical_right_route")) for r in canonical_hits
        if r.get("canonical_right_route") is not None
    ]
    canonical_right_route_rate = (
        _mean([float(x) for x in right_route_flags]) if right_route_flags else None
    )

    leaks = [
        float(v[Condition.RIGHT_DISABLED.value].get("right_arm_clamp_leak_rad", 0.0))
        for v in matched.values()
    ]
    max_clamp_leak = max(leaks) if leaks else 0.0

    paired_sub = np.asarray(
        [
            float(bool(v[Condition.RIGHT_DISABLED.value]["success"]))
            - float(bool(v[Condition.FULL_HOLD.value]["success"]))
            for v in matched.values()
        ],
        dtype=float,
    )
    diff_mean, diff_lo, diff_hi = _bootstrap_mean_ci(
        paired_sub, cfg.bootstrap_samples, cfg.bootstrap_seed
    )
    substitution_events = int(
        sum(
            bool(v[Condition.RIGHT_DISABLED.value]["success"])
            and not bool(v[Condition.FULL_HOLD.value]["success"])
            for v in matched.values()
        )
    )

    route_flags = [
        v[Condition.RIGHT_DISABLED.value].get("route_verified")
        for v in matched.values()
        if bool(v[Condition.RIGHT_DISABLED.value]["success"])
        and v[Condition.RIGHT_DISABLED.value].get("route_verified") is not None
    ]
    route_verified_rate = (
        _mean([float(bool(x)) for x in route_flags]) if route_flags else None
    )

    if n < cfg.min_matched_configs:
        verdict = "INSUFFICIENT_MATCHED_CONFIGS"
    elif canonical < cfg.canonical_min_success:
        verdict = "PREREQUISITE_FAIL_CANONICAL"
    elif arm_program_cost < cfg.min_arm_program_cost:
        # Locking the right arm in place changes nothing, so the canonical solution
        # does not contain a right-arm motor program. Nothing to substitute.
        verdict = "PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM"
    elif (
        canonical_right_route_rate is not None
        and canonical_right_route_rate < cfg.canonical_right_route_min
    ):
        verdict = "PREREQUISITE_FAIL_ROUTE_NOT_RIGHT_SIDE"
    elif max_clamp_leak > cfg.max_clamp_leak_rad:
        verdict = "PREREQUISITE_FAIL_INTERVENTION_LEAK"
    elif both_arms > cfg.both_arms_disabled_max_success:
        # The task survives losing both arms, so any right_disabled success is a
        # body/base route, not one arm standing in for the other.
        verdict = "PREREQUISITE_FAIL_BODY_ONLY_ROUTE"
    elif oracle is None:
        # Everything the policy conditions can decide has passed; the scripted
        # feasibility oracle has not been run yet, so no verdict is available.
        verdict = "PREREQUISITE_PENDING_ALTERNATIVE_FEASIBILITY"
    elif oracle < cfg.oracle_min_success:
        verdict = "PREREQUISITE_FAIL_ALTERNATIVE_FEASIBILITY"
    elif full_hold > cfg.full_hold_max_success:
        verdict = "PREREQUISITE_FAIL_NEGATIVE_CONTROL"
    elif (
        right_disabled >= cfg.min_substitution_rate
        and substitution_events >= cfg.min_substitution_events
        and diff_mean >= cfg.min_substitution_rate
    ):
        verdict = (
            "PROMISING_MOTOR_SUBSTITUTION"
            if route_verified_rate is not None and route_verified_rate >= 0.8
            else "PROMISING_NEEDS_ROUTE_VERIFICATION"
        )
    else:
        verdict = "NO_EVIDENCE_IN_PSI0_G0"

    return {
        "n_input_rows": len(rows),
        "n_matched_configs": n,
        "success_rate": cond_success,
        "oracle_success_rate": oracle,
        "n_oracle_rows": len(oracle_rows),
        "arm_program_cost": arm_program_cost,
        "canonical_right_route_rate": canonical_right_route_rate,
        "max_clamp_leak_rad": max_clamp_leak,
        "paired_right_disabled_minus_full_hold": {
            "mean": diff_mean,
            "bootstrap_95_ci": [diff_lo, diff_hi],
        },
        "substitution_events": substitution_events,
        "route_verified_rate_among_successes": route_verified_rate,
        "gate_config": cfg.__dict__,
        "verdict": verdict,
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("records")
    p.add_argument("--out", default="g0_result.json")
    args = p.parse_args()
    report = analyze_records(load_jsonl(args.records))
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
