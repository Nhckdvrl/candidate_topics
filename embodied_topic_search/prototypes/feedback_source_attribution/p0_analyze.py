"""Frozen P0 replay-fidelity gate for the feedback-source attribution candidate.

The gate is the one written down before any rollout was run (see README):

    fresh                    >= 0.90
    vla_replay               >= 0.90
    actuator_replay          >= 0.90
    fresh - vla_replay       <= 0.10
    fresh - actuator_replay  <= 0.10

A pass licenses registering the question as a root Topic and freezing the
physical-disturbance G0. A fail stops the candidate: without it, a later
`fresh > actuator_replay` could not be read as a low-level feedback
contribution, because the replay instrument would already break the task with no
disturbance present.

Two structural checks run first and are not thresholds to be tuned:

  * every replay condition must consume the whole recorded tape at the recorded
    cadence (`steps == tape_len`, `tape_exhausted_early == False`);
  * a replay condition must never contact the policy server (`server_queries == 0`).

Trajectory fidelity is reported alongside the success gate but is descriptive:
success is what the frozen contract is written on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from g0_core import CONDITIONS, p0_fidelity


def structural_violations(rows: Iterable[dict]) -> list[str]:
    bad: list[str] = []
    for r in rows:
        key = f"{r['config_id']}/{r['condition']}"
        if r["condition"] != "fresh":
            if r.get("server_queries", 0) != 0:
                bad.append(f"{key}: replay contacted the policy server "
                           f"({r['server_queries']} queries)")
            if r.get("tape_exhausted_early"):
                bad.append(f"{key}: tape exhausted before the episode ended")
            if r.get("steps") != r.get("tape_len"):
                bad.append(f"{key}: consumed {r.get('steps')} of {r.get('tape_len')} tape rows")
        if r.get("force_n", 0.0) != 0.0:
            bad.append(f"{key}: P0 rows must be unperturbed, got force_n={r['force_n']}")
    return bad


def trajectory_fidelity(rows: list[dict], tape_dir: Path | None) -> dict:
    """Descriptive divergence of each replay from its own fresh rollout."""
    fresh = {r["config_id"]: r for r in rows if r["condition"] == "fresh"}
    out: dict[str, dict[str, float]] = {}
    for cond in ("vla_replay", "actuator_replay"):
        d_door, d_base = [], []
        for r in rows:
            if r["condition"] != cond or r["config_id"] not in fresh:
                continue
            f = fresh[r["config_id"]]
            if r.get("effect_qpos") is not None and f.get("effect_qpos") is not None:
                d_door.append(abs(r["effect_qpos"] - f["effect_qpos"]))
            if r.get("terminal_base_xyz") and f.get("terminal_base_xyz"):
                d_base.append(float(np.linalg.norm(
                    np.asarray(r["terminal_base_xyz"]) - np.asarray(f["terminal_base_xyz"])
                )))
        if tape_dir is not None:
            for r in rows:
                if r["condition"] != cond:
                    continue
                cfg = r["config_id"].replace(":", "_cfg")
                a = tape_dir / f"{cfg}_{cond}_trace.json"
                b = tape_dir / f"{cfg}_fresh_trace.json"
                if a.exists() and b.exists():
                    ta = json.loads(a.read_text())["door_trace"]
                    tb = json.loads(b.read_text())["door_trace"]
                    n = min(len(ta), len(tb))
                    out.setdefault(cond, {})["max_door_dev_rad"] = max(
                        out.get(cond, {}).get("max_door_dev_rad", 0.0),
                        float(np.max(np.abs(np.asarray(ta[:n]) - np.asarray(tb[:n])))) if n else 0.0,
                    )
        e = out.setdefault(cond, {})
        e["max_terminal_door_dev_rad"] = max(d_door) if d_door else None
        e["max_terminal_base_dev_m"] = max(d_base) if d_base else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("records", type=Path)
    p.add_argument("--tape-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min-configs", type=int, default=10)
    args = p.parse_args()

    rows = [json.loads(x) for x in args.records.read_text().splitlines() if x.strip()]
    violations = structural_violations(rows)
    gate = p0_fidelity(rows)
    configs = sorted({r["config_id"] for r in rows})
    complete = [
        c for c in configs
        if {r["condition"] for r in rows if r["config_id"] == c} >= set(CONDITIONS)
    ]

    verdict = "PASS"
    if violations:
        verdict = "FAIL_STRUCTURAL"
    elif len(complete) < args.min_configs:
        verdict = "INSUFFICIENT_CONFIGS"
    elif not gate["pass"]:
        verdict = "FAIL_REPLAY_FIDELITY"

    result = {
        "verdict": verdict,
        "n_complete_configs": len(complete),
        "gate": gate,
        "structural_violations": violations,
        "trajectory_fidelity": trajectory_fidelity(rows, args.tape_dir),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
