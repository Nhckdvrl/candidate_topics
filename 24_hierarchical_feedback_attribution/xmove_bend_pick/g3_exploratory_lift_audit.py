"""Topic 24 G3 -- POST-HOC EXPLORATORY audit.

Does binary task success hide continuous effects of VLA feedback under
floor conditions?

G3's confirmatory verdict is `PREREQUISITE_FAIL_REPLAY_FIDELITY` and this
script does not change it, does not re-run the gate, and does not select
any force/direction/config after seeing an outcome. It answers a different,
explicitly exploratory question against the same frozen 392-row panel:

    Delta_lift(f, d)   = final_target_lift_m[fresh] - final_target_lift_m[vla_replay]
    Delta_reward(f, d) = clip(lift_fresh/0.1, 0, 1) - clip(lift_replay/0.1, 0, 1)

`Delta_reward` uses the task's own published reward normalization
(`XMoveBendPickTeleop`'s `compute_reward`: r = clip((z - z0)/0.1, 0, 1),
thresholded at r >= 0.8 for the binary `success` G3 already reports), so
this is not an invented metric -- it is the continuous quantity success
is itself thresholded from.

All six force x direction cells are reported. None are selected after
seeing a result. The 0N control cell is reported too, as a sanity check
only (it should be ~0; a nonzero control delta would itself be a flag
about replay fidelity at the continuous level, independent of the binary
gate already reported in G3_RESULTS.md).

Pre-declared kill rule (set by the user before this script ran): if the
full six-cell Delta_lift/Delta_reward grid shows no structure -- deltas
indistinguishable from zero with signs that do not track force magnitude
-- Topic 24 stops on XMoveBendPickTeleop. This script does not decide
that call; it only produces the grid.
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
REWARD_NORM_M = 0.1


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


def _lift(row: dict) -> float:
    return float(row["final_target_lift_m"])


def _reward(row: dict) -> float:
    return float(np.clip(_lift(row) / REWARD_NORM_M, 0.0, 1.0))


@dataclass(frozen=True)
class ContinuousCellResult:
    force_n: float
    direction: str
    n_configs: int
    mean_lift_fresh: float
    mean_lift_replay: float
    delta_lift_point: float
    delta_lift_ci_lo: float
    delta_lift_ci_hi: float
    mean_reward_fresh: float
    mean_reward_replay: float
    delta_reward_point: float
    delta_reward_ci_lo: float
    delta_reward_ci_hi: float
    # binary success carried alongside for direct comparison to G3_RESULTS.md
    success_fresh: float
    success_replay: float


def cell_result(
    rows: Iterable[dict], force_n: float, direction: str, *, n_boot: int = 10000, seed: int = 20260826
) -> ContinuousCellResult:
    pairs = _pairs(rows, force_n, direction)
    if not pairs:
        raise ValueError(f"no complete fresh/vla_replay pairs for {force_n}N/{direction}")

    lift_fresh = np.array([_lift(p["fresh"]) for p in pairs])
    lift_replay = np.array([_lift(p["vla_replay"]) for p in pairs])
    reward_fresh = np.array([_reward(p["fresh"]) for p in pairs])
    reward_replay = np.array([_reward(p["vla_replay"]) for p in pairs])
    succ_fresh = np.array([int(bool(p["fresh"]["success"])) for p in pairs])
    succ_replay = np.array([int(bool(p["vla_replay"]["success"])) for p in pairs])

    delta_lift_point = float(np.mean(lift_fresh - lift_replay))
    delta_reward_point = float(np.mean(reward_fresh - reward_replay))

    rng = np.random.default_rng(seed)
    n = len(pairs)
    boot_lift = np.empty(n_boot)
    boot_reward = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_lift[b] = np.mean(lift_fresh[idx] - lift_replay[idx])
        boot_reward[b] = np.mean(reward_fresh[idx] - reward_replay[idx])
    lift_lo, lift_hi = np.quantile(boot_lift, [0.025, 0.975])
    reward_lo, reward_hi = np.quantile(boot_reward, [0.025, 0.975])

    return ContinuousCellResult(
        force_n=force_n,
        direction=direction,
        n_configs=n,
        mean_lift_fresh=float(np.mean(lift_fresh)),
        mean_lift_replay=float(np.mean(lift_replay)),
        delta_lift_point=delta_lift_point,
        delta_lift_ci_lo=float(lift_lo),
        delta_lift_ci_hi=float(lift_hi),
        mean_reward_fresh=float(np.mean(reward_fresh)),
        mean_reward_replay=float(np.mean(reward_replay)),
        delta_reward_point=delta_reward_point,
        delta_reward_ci_lo=float(reward_lo),
        delta_reward_ci_hi=float(reward_hi),
        success_fresh=float(np.mean(succ_fresh)),
        success_replay=float(np.mean(succ_replay)),
    )


def full_grid(rows: Iterable[dict]) -> list[ContinuousCellResult]:
    rows = list(rows)
    out = []
    for force_n, direction in [(0.0, "none")] + [(f, d) for f in FORCES_N for d in DIRECTIONS]:
        try:
            out.append(cell_result(rows, force_n, direction))
        except ValueError:
            continue
    return out


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("records", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = load_jsonl(args.records)
    grid = full_grid(rows)
    result = {
        "label": "POST_HOC_EXPLORATORY_LIFT_AUDIT",
        "does_not_alter_g3_verdict": True,
        "g3_confirmatory_verdict": "PREREQUISITE_FAIL_REPLAY_FIDELITY",
        "grid": [asdict(c) for c in grid],
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
