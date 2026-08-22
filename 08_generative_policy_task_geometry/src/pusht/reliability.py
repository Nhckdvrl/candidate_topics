"""Finite-B sanity checks.

ACE and the measured outcome dispersion are both estimated from the *same* B sampled
action chunks at a state. If a chunk of the state-to-state residual were just Monte Carlo
noise in those estimators, it would manufacture exactly the pattern this topic is looking
for: two states with the same ACE and different measured outcome dispersion. So the
residual has to be shown to be reliable before it is interpreted.

Both checks reuse the stored per-sample data; neither needs any new simulation.

  * split-half: score each state twice on disjoint halves of its B samples and correlate
    across states. This is the reliability ceiling of the measurement.
  * noise-to-signal: median |half A - half B| against the between-state IQR.
  * estimator sensitivity: ACE recomputed at several `cellsize_factor` values. FIPER's
    released 0.03 is calibrated to the whole workspace range, which may be far too coarse
    for a policy whose per-state spread is a couple of pixels -- if the topic's effect
    only exists at 0.03, it is a statement about the constant, not about task geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .ace import ace
from .collect_e1 import keypoint_dispersion


def load_raw(run_dir: Path, df: pd.DataFrame):
    z = np.load(run_dir / "raw_samples.npz")
    rolls = sorted({int(r) for r in df.rollout.unique()})
    chunks = np.concatenate([z[f"chunks_{r}"] for r in rolls], axis=0)
    kps = np.concatenate([z[f"keypoints_{r}"] for r in rolls], axis=0)
    return chunks, kps


def _rho(x, y) -> float:
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def split_half(chunks, kps, cell, seed: int = 0) -> dict:
    p, b = kps.shape[0], kps.shape[1]
    rng = np.random.default_rng(seed)
    perm = np.array([rng.permutation(b) for _ in range(p)])
    a_idx, b_idx = perm[:, : b // 2], perm[:, b // 2 :]

    d_a = np.array([keypoint_dispersion(kps[i][a_idx[i]]) for i in range(p)])
    d_b = np.array([keypoint_dispersion(kps[i][b_idx[i]]) for i in range(p)])
    ace_a = np.array([ace(chunks[i][a_idx[i]], cell) for i in range(p)])
    ace_b = np.array([ace(chunks[i][b_idx[i]], cell) for i in range(p)])
    return {
        "n_states": int(p),
        "samples_per_half": int(b // 2),
        "reliability_outcome_dispersion": _rho(d_a, d_b),
        "reliability_ace": _rho(ace_a, ace_b),
        "median_half_vs_half_outcome_diff_px": float(np.median(np.abs(d_a - d_b))),
        "_d_a": d_a,
        "_d_b": d_b,
    }


def estimator_sweep(chunks, outcome, contact, cell, factors=(0.03, 0.01, 0.003, 0.001)) -> list[dict]:
    rows = []
    for f in factors:
        c = cell / 0.03 * f
        a = np.array([ace(x, c) for x in chunks])
        rows.append(
            {
                "cellsize_factor": f,
                "is_released_fiper_value": bool(abs(f - 0.03) < 1e-12),
                "ace_p10": float(np.quantile(a, 0.10)),
                "ace_p50": float(np.quantile(a, 0.50)),
                "ace_p90": float(np.quantile(a, 0.90)),
                "frac_ace_exactly_zero": float((a == 0).mean()),
                "spearman_ace_outcome_all": _rho(a, outcome),
                "spearman_ace_outcome_in_contact": _rho(a[contact], outcome[contact]),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.run_dir / "probe_states.csv")
    cell = np.load(args.run_dir / "ace_cell_size.npy")
    chunks, kps = load_raw(args.run_dir, df)

    outcome = df.outcome_kp_dispersion_px.to_numpy(float)
    contact = (df.frac_samples_with_contact > 0.5).to_numpy()

    sh = split_half(chunks, kps, cell, seed=args.seed)
    d_a, d_b = sh.pop("_d_a"), sh.pop("_d_b")
    iqr_contact = float(np.subtract(*np.percentile(outcome[contact], [75, 25])))
    sh["between_state_iqr_in_contact_px"] = iqr_contact
    sh["noise_to_between_state_iqr"] = float(
        np.median(np.abs(d_a - d_b)[contact]) / max(iqr_contact, 1e-12)
    )

    # B-stability of the point estimates actually used in the analysis
    half = kps.shape[1] // 2
    d_half = np.array([keypoint_dispersion(kps[i][:half]) for i in range(kps.shape[0])])
    sh["spearman_outcome_halfB_vs_fullB"] = _rho(d_half, outcome)

    report = {
        "run_dir": str(args.run_dir),
        "split_half": sh,
        "estimator_sweep": estimator_sweep(chunks, outcome, contact, cell),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
