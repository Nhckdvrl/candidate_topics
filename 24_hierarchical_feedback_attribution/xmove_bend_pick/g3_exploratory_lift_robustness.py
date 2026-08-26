"""Topic 24 G3 -- second-pass robustness audit on the POST-HOC EXPLORATORY
lift analysis (`g3_exploratory_lift_audit.py`).

Three checks, all zero-GPU, all against the same frozen 392-row panel:

1. A studentized max-T sign-flip permutation test across the six
   force/direction cells, clustered by config (one random +-1 sign per
   config, applied to that config's Delta_lift in all six cells at once, so
   the cross-cell correlation structure induced by sharing configs is
   preserved). This gives a family-wise-error-controlled omnibus p-value and
   per-cell maxT-adjusted p-values / simultaneous CIs, replacing the plain
   per-cell 95% CIs that do not account for having scanned six cells.

2. Leave-one-config-out fragility: for each of the 28 configs, drop it and
   recompute the cell mean and (for the two cells whose plain CI excluded
   zero) the clustered bootstrap CI on the remaining 27. Reports the range of
   point estimates, whether the sign ever flips, and which single-config
   removals cause the CI to re-cross zero.

3. Catastrophic-swing threshold sensitivity: repeats the fresh-worse vs
   replay-worse tail count at |Delta_lift| > {0.10, 0.15, 0.20, 0.25} instead
   of the single 0.20 cutoff used in the first pass, to check whether the
   asymmetric-tail description is a threshold artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from g3_exploratory_lift_audit import _pairs, _lift

CELLS = [(50.0, "left"), (50.0, "right"), (100.0, "left"), (100.0, "right"),
         (150.0, "left"), (150.0, "right")]


def load_deltas(rows: list[dict]) -> tuple[list[str], dict[tuple[float, str], np.ndarray]]:
    """Return (config_order, {cell: array of per-config Delta_lift in that order})."""
    per_cell_pairs = {cell: _pairs(rows, *cell) for cell in CELLS}
    config_order = sorted(p["fresh"]["config_id"] for p in per_cell_pairs[CELLS[0]])
    out = {}
    for cell, pairs in per_cell_pairs.items():
        by_cfg = {p["fresh"]["config_id"]: _lift(p["fresh"]) - _lift(p["vla_replay"]) for p in pairs}
        assert set(by_cfg) == set(config_order), f"config set mismatch at {cell}"
        out[cell] = np.array([by_cfg[c] for c in config_order])
    return config_order, out


# ---------------------------------------------------------------------------
# 1. studentized max-T sign-flip permutation, clustered by config
# ---------------------------------------------------------------------------

def studentized_t(x: np.ndarray) -> float:
    n = len(x)
    sd = np.std(x, ddof=1)
    if sd == 0.0:
        return 0.0
    return float(np.mean(x) / (sd / np.sqrt(n)))


def maxT_permutation(deltas: dict[tuple[float, str], np.ndarray], *, n_perm: int = 100_000,
                      seed: int = 20260826) -> dict:
    n = len(next(iter(deltas.values())))
    obs_t = {cell: studentized_t(arr) for cell, arr in deltas.items()}
    obs_mean = {cell: float(np.mean(arr)) for cell, arr in deltas.items()}
    obs_se = {cell: float(np.std(arr, ddof=1) / np.sqrt(n)) for cell, arr in deltas.items()}
    T_obs_max = max(abs(t) for t in obs_t.values())

    rng = np.random.default_rng(seed)
    stacked = np.stack([deltas[c] for c in CELLS], axis=0)  # (6, n_configs)
    null_Tmax = np.empty(n_perm)

    batch = 2000
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        signs = rng.choice([-1.0, 1.0], size=(b, n))  # (b, n_configs), shared across cells per draw
        # permuted values: (b, 6, n_configs) = signs[:,None,:] * stacked[None,:,:]
        perm = signs[:, None, :] * stacked[None, :, :]
        means = perm.mean(axis=2)  # (b, 6)
        sds = perm.std(axis=2, ddof=1)  # (b, 6)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(sds > 0, means / (sds / np.sqrt(n)), 0.0)  # (b, 6)
        abs_t = np.abs(t)
        Tmax_batch = abs_t.max(axis=1)  # (b,) -- max over the 6 cells, per permutation
        null_Tmax[done:done + b] = Tmax_batch
        done += b

    p_omnibus = float((np.sum(null_Tmax >= T_obs_max) + 1) / (n_perm + 1))
    q95_Tmax = float(np.quantile(null_Tmax, 0.95))

    # Single-step maxT adjusted p-value (Westfall-Young): each cell's OWN
    # observed |t| is compared against the JOINT max-over-cells null
    # distribution, not that cell's own marginal null -- this is what makes
    # it multiplicity-corrected. Using each cell's own marginal null instead
    # would just reproduce the uncorrected per-cell p-value.
    per_cell = {}
    for cell in CELLS:
        p_adj = float((np.sum(null_Tmax >= abs(obs_t[cell])) + 1) / (n_perm + 1))
        sim_lo = obs_mean[cell] - q95_Tmax * obs_se[cell]
        sim_hi = obs_mean[cell] + q95_Tmax * obs_se[cell]
        per_cell[f"{cell[0]}N/{cell[1]}"] = {
            "point": obs_mean[cell],
            "t_obs": obs_t[cell],
            "maxT_adjusted_p": p_adj,
            "simultaneous_95CI_lo": sim_lo,
            "simultaneous_95CI_hi": sim_hi,
            "multiplicity_robust_signal": bool(sim_lo > 0 or sim_hi < 0),
        }

    return {
        "n_perm": n_perm,
        "n_configs": n,
        "T_obs_max": T_obs_max,
        "q95_null_Tmax": q95_Tmax,
        "omnibus_p": p_omnibus,
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# 2. leave-one-config-out fragility
# ---------------------------------------------------------------------------

def clustered_bootstrap_ci(x: np.ndarray, *, n_boot: int = 10_000, seed: int = 20260826) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = np.mean(x[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def leave_one_out(config_order: list[str], deltas: dict[tuple[float, str], np.ndarray],
                   *, recompute_ci_cells: list[tuple[float, str]]) -> dict:
    out = {}
    n = len(config_order)
    for cell, arr in deltas.items():
        loo_means = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            loo_means.append(float(np.mean(arr[mask])))
        loo_means = np.array(loo_means)
        full_mean = float(np.mean(arr))
        sign_full = np.sign(full_mean)
        sign_flips = [config_order[i] for i in range(n) if np.sign(loo_means[i]) != sign_full and sign_full != 0]

        entry = {
            "full_point": full_mean,
            "loo_point_min": float(loo_means.min()),
            "loo_point_max": float(loo_means.max()),
            "loo_point_range": float(loo_means.max() - loo_means.min()),
            "sign_ever_flips": len(sign_flips) > 0,
            "configs_causing_sign_flip": sign_flips,
        }

        if cell in recompute_ci_cells:
            crossing_configs = []
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                lo, hi = clustered_bootstrap_ci(arr[mask])
                if lo <= 0 <= hi:
                    crossing_configs.append({
                        "removed_config": config_order[i],
                        "loo_point": float(loo_means[i]),
                        "loo_ci_lo": lo, "loo_ci_hi": hi,
                    })
            full_lo, full_hi = clustered_bootstrap_ci(arr)
            entry["full_ci"] = [full_lo, full_hi]
            entry["configs_whose_removal_crosses_zero"] = crossing_configs
            entry["fragile"] = len(crossing_configs) > 0

        out[f"{cell[0]}N/{cell[1]}"] = entry
    return out


# ---------------------------------------------------------------------------
# 3. catastrophic-swing threshold sensitivity
# ---------------------------------------------------------------------------

def threshold_sensitivity(deltas: dict[tuple[float, str], np.ndarray],
                           thresholds: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)) -> dict:
    out = {}
    for cell, arr in deltas.items():
        rows = {}
        for th in thresholds:
            fresh_worse = int(np.sum(arr < -th))
            replay_worse = int(np.sum(arr > th))
            rows[f"gt_{th:.2f}m"] = {"fresh_worse_tail": fresh_worse, "replay_worse_tail": replay_worse}
        out[f"{cell[0]}N/{cell[1]}"] = rows
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("records", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-perm", type=int, default=100_000)
    args = p.parse_args()

    rows = [json.loads(x) for x in args.records.read_text().splitlines() if x.strip()]
    config_order, deltas = load_deltas(rows)

    maxT = maxT_permutation(deltas, n_perm=args.n_perm)
    loo = leave_one_out(config_order, deltas, recompute_ci_cells=[(100.0, "right"), (150.0, "left")])
    thresh = threshold_sensitivity(deltas)

    result = {
        "label": "POST_HOC_EXPLORATORY_LIFT_ROBUSTNESS_AUDIT",
        "n_configs": len(config_order),
        "maxT_permutation": maxT,
        "leave_one_config_out": loo,
        "threshold_sensitivity": thresh,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
