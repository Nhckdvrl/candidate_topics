"""Outcome-space G0 utilities for Topic 23.

The scientific question is whether a pretrained robot policy preserves a task effect
after the demonstrator's canonical right-side motor route is removed.

Important: the primary endpoint is environment success/effect, not a projection in
joint space. This is deliberate after the Topic 19 identification failure.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


class Condition(str, Enum):
    CANONICAL = "canonical"
    RIGHT_DISABLED = "right_disabled"
    FULL_HOLD = "full_hold"
    ORACLE_RIGHT_DISABLED = "oracle_right_disabled"


STATE_MATCH_GROUPS = ("left_hand", "right_hand", "left_arm", "right_arm", "rpy", "height")
RIGHT_GROUPS = ("right_hand", "right_arm")
ZERO_VELOCITY_GROUPS = ("torso_vx", "torso_vy", "torso_vyaw")


@dataclass(frozen=True)
class GateConfig:
    min_matched_configs: int = 20
    canonical_min_success: float = 0.70
    oracle_min_success: float = 0.70
    full_hold_max_success: float = 0.10
    min_substitution_rate: float = 0.20
    min_substitution_events: int = 5
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260824


def _copy_value(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.copy()
    return copy.deepcopy(x)


def intervene_absolute_action(
    action: Mapping[str, Any],
    state: Mapping[str, Any],
    condition: Condition | str,
) -> dict[str, Any]:
    """Apply a transparent post-policy motor intervention.

    Psi0/GR00T G1 loco-manip action groups are absolute for arm/hand/body pose
    groups. RIGHT_DISABLED holds the canonical right arm+hand at the observed
    state while leaving left arm, torso and locomotion available. FULL_HOLD
    removes all intentional body motion and is a negative control.

    This function intentionally operates *after* policy inference. The policy
    still sees the real constrained state/consequences on subsequent steps.
    """
    c = Condition(condition)
    out = {k: _copy_value(v) for k, v in action.items()}

    if c in (Condition.CANONICAL, Condition.ORACLE_RIGHT_DISABLED):
        if c is Condition.CANONICAL:
            return out
        # Oracle trajectories are evaluated under the same physical right-side
        # intervention as the policy; fall through to RIGHT_DISABLED behavior.
        c = Condition.RIGHT_DISABLED

    if c is Condition.RIGHT_DISABLED:
        for key in RIGHT_GROUPS:
            if key not in out:
                raise KeyError(f"action missing required group {key!r}")
            if key not in state:
                raise KeyError(f"state missing required group {key!r}")
            out[key] = _copy_value(state[key])
        return out

    if c is Condition.FULL_HOLD:
        for key in STATE_MATCH_GROUPS:
            if key in out:
                if key not in state:
                    raise KeyError(f"state missing group {key!r} required by FULL_HOLD")
                out[key] = _copy_value(state[key])
        for key in ZERO_VELOCITY_GROUPS:
            if key in out:
                out[key] = np.zeros_like(np.asarray(out[key], dtype=float))
        if "target_yaw" in out:
            if "rpy" not in state:
                raise KeyError("state missing 'rpy' needed to hold target_yaw")
            rpy = np.asarray(state["rpy"]).reshape(-1)
            if rpy.size < 3:
                raise ValueError("state['rpy'] must contain roll, pitch, yaw")
            target = np.asarray(out["target_yaw"])
            out["target_yaw"] = np.full_like(target, rpy[-1], dtype=float)
        return out

    raise AssertionError(c)


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
      effect_qpos, route_verified, left_arm_motion_l2, torso_motion_l2

    Only configs present in all four frozen conditions are primary units.
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

    needed = {c.value for c in Condition}
    matched = {k: v for k, v in by_cfg.items() if needed <= set(v)}
    n = len(matched)

    cond_success: dict[str, float] = {}
    for c in Condition:
        cond_success[c.value] = _mean(
            [float(bool(v[c.value]["success"])) for v in matched.values()]
        )

    canonical = cond_success[Condition.CANONICAL.value]
    oracle = cond_success[Condition.ORACLE_RIGHT_DISABLED.value]
    full_hold = cond_success[Condition.FULL_HOLD.value]
    right_disabled = cond_success[Condition.RIGHT_DISABLED.value]

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
