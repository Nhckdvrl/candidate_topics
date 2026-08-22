"""E1 analysis: does a scalar action-diversity score already determine the true
task-outcome dispersion?

Reports, for each scalar diversity score (FIPER ACE and two estimator-free dispersion
measures) against each measured outcome-dispersion variable:

  * pooled Spearman, and Spearman computed *within* rollouts (states inside a rollout are
    dependent, so the pooled number can be a between-rollout artefact);
  * how wide the outcome distribution still is inside narrow quantile bins of the score;
  * matched-score pairs drawn from different rollouts;

with rollout-level bootstrap CIs on the headline quantities.

This script deliberately does **not** hard-code a pass/fail verdict. Discovery runs are
descriptive; the decision thresholds are frozen separately in `gate_e1.json` and applied
by `--gate`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    binned_spread,
    matched_pairs,
    rollout_bootstrap,
    spearman,
    within_group_spearman,
)

SCORES = ["ace", "act_rms_dispersion", "act_trace_cov_mean"]
OUTCOMES = ["outcome_kp_dispersion_px", "outcome_kp_pairwise_px", "outcome_cov_std"]


def describe(df: pd.DataFrame) -> dict:
    return {
        "n_probe_states": int(len(df)),
        "n_rollouts": int(df["rollout"].nunique()),
        "probes_per_rollout_mean": float(df.groupby("rollout").size().mean()),
        "action_dispersion_px": {
            "p10": float(df.act_rms_dispersion.quantile(0.10)),
            "p50": float(df.act_rms_dispersion.median()),
            "p90": float(df.act_rms_dispersion.quantile(0.90)),
        },
        "outcome_kp_dispersion_px": {
            "p10": float(df.outcome_kp_dispersion_px.quantile(0.10)),
            "p50": float(df.outcome_kp_dispersion_px.median()),
            "p90": float(df.outcome_kp_dispersion_px.quantile(0.90)),
        },
        "ace_bits": {
            "p10": float(df.ace.quantile(0.10)),
            "p50": float(df.ace.median()),
            "p90": float(df.ace.quantile(0.90)),
        },
        # Q2 of the brief: how much sampled diversity is goal-equivalent? Measured as the
        # share of probe states whose actions spread widely but whose outcomes do not.
        "frac_states_high_action_low_outcome": float(
            (
                (df.act_rms_dispersion > df.act_rms_dispersion.median())
                & (df.outcome_kp_dispersion_px < df.outcome_kp_dispersion_px.median())
            ).mean()
        ),
        "frac_states_in_contact": float((df.frac_samples_with_contact > 0.5).mean()),
        "rollout_success_rate": float(
            df.groupby("rollout").rollout_terminated.max().mean()
        ),
    }


def score_outcome_table(df: pd.DataFrame, boot: int, seed: int) -> list[dict]:
    rows = []
    for s in SCORES:
        for o in OUTCOMES:
            pooled = rollout_bootstrap(df, lambda d, s=s, o=o: spearman(d[s], d[o]), n=boot, seed=seed)
            within = rollout_bootstrap(
                df, lambda d, s=s, o=o: within_group_spearman(d, s, o), n=boot, seed=seed
            )
            rows.append(
                {
                    "score": s,
                    "outcome": o,
                    "spearman_pooled": pooled["point"],
                    "spearman_pooled_ci95": pooled["ci95"],
                    "spearman_within_rollout": within["point"],
                    "spearman_within_rollout_ci95": within["ci95"],
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--match-tol-z", type=float, default=0.10)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--gate", type=Path, default=None, help="frozen gate JSON; applies a verdict")
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    # rollout ids must stay unique across merged files
    if df.duplicated(subset=["rollout", "probe"]).any():
        raise ValueError("duplicate (rollout, probe) across inputs -- use disjoint seed bases")

    report = {"inputs": [str(c) for c in args.csv], "descriptive": describe(df)}
    report["score_vs_outcome"] = score_outcome_table(df, args.bootstrap, args.seed)

    primary_score, primary_outcome = "ace", "outcome_kp_dispersion_px"
    bs = binned_spread(df, primary_score, primary_outcome, n_bins=args.bins)
    report["binned_spread_ace"] = bs.to_dict(orient="records")
    report["median_p90_over_p10_within_ace_bin"] = (
        float(bs.p90_over_p10.median()) if len(bs) else None
    )

    mp = matched_pairs(df, primary_score, primary_outcome, tol_z=args.match_tol_z)
    report["matched_pairs"] = {
        "n_pairs": int(len(mp)),
        "mean_abs_score_z_gap": float(mp.dz.mean()) if len(mp) else None,
        "median_outcome_ratio_hi_over_lo": float(mp.ratio.median()) if len(mp) else None,
        "median_outcome_diff_px": float(mp["diff"].median()) if len(mp) else None,
    }

    def _matched_ratio(d):
        m = matched_pairs(d, primary_score, primary_outcome, tol_z=args.match_tol_z)
        return float(m.ratio.median()) if len(m) >= 10 else np.nan

    report["matched_pairs"]["median_ratio_ci95"] = rollout_bootstrap(
        df, _matched_ratio, n=args.bootstrap, seed=args.seed
    )["ci95"]

    # The mundane alternative explanation: "diversity is harmless exactly when the pusher
    # is not touching the block". If contact state alone explains the residual, the
    # geometry story adds nothing.
    contact = df.frac_samples_with_contact > 0.5
    report["contact_stratified"] = {}
    for name, sub in (("in_contact", df[contact]), ("no_contact", df[~contact])):
        if len(sub) < 20:
            report["contact_stratified"][name] = {"n": int(len(sub)), "note": "too few states"}
            continue
        b = binned_spread(sub, primary_score, primary_outcome, n_bins=max(3, args.bins // 2))
        report["contact_stratified"][name] = {
            "n": int(len(sub)),
            "spearman_ace_outcome": spearman(sub[primary_score], sub[primary_outcome]),
            "median_p90_over_p10_within_ace_bin": float(b.p90_over_p10.median()) if len(b) else None,
        }

    if args.gate is not None:
        gate = json.loads(args.gate.read_text())
        fails = []
        sp = next(
            r for r in report["score_vs_outcome"]
            if r["score"] == primary_score and r["outcome"] == primary_outcome
        )
        if sp["spearman_pooled"] >= gate["kill_if_spearman_at_least"]:
            fails.append("KILL_scalar_entropy_already_determines_outcome_dispersion")
        if (report["median_p90_over_p10_within_ace_bin"] or 0) < gate["min_median_p90_over_p10"]:
            fails.append("KILL_no_outcome_spread_at_matched_entropy")
        if report["matched_pairs"]["n_pairs"] < gate["min_matched_pairs"]:
            fails.append("KILL_insufficient_matched_pairs")
        lo = report["matched_pairs"]["median_ratio_ci95"][0]
        if lo is None or lo < gate["min_matched_ratio_ci_lower"]:
            fails.append("KILL_matched_ratio_not_robust_to_rollout_bootstrap")
        report["gate"] = gate
        report["failed_clauses"] = fails
        report["verdict"] = "CONTINUE" if not fails else "KILL"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
