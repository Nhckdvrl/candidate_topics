"""E1 analysis: does a scalar action-diversity score determine true task-outcome
dispersion?

The headline statistic is `matched_pair_reduction`: pair up states that share a scalar
diversity score (the outcome plays **no** part in forming the pairs) and ask whether
their true task outcomes are any more alike than two states picked at random.

    reduction ~ 1   knowing the entropy tells you nothing about outcome uncertainty
    reduction << 1  the entropy largely determines it, and the topic is dead

An earlier version of this file gated on `matched_pairs_descriptive`, which selects pair
members from the top and bottom outcome quartiles. That ratio is large by construction --
bounded below by Q75/Q25 whatever the score does -- so it cannot be evidence. It is kept
here as an illustration and is explicitly excluded from the gate.

Every documented kill criterion is implemented in `--gate`, not just the pooled ones:
the effect must survive *within* rollouts, must survive inside the in-contact stratum
alone, must not depend on the ACE cell-size constant, and must be large relative to the
finite-B noise floor measured by `reliability.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    binned_spread,
    matched_pair_reduction,
    matched_pairs_descriptive,
    rollout_bootstrap,
    spearman,
    within_group_spearman,
)

SCORES = ["ace", "ace_executed", "act_rms_dispersion", "act_trace_cov_mean"]
OUTCOMES = ["outcome_kp_dispersion_px", "outcome_kp_pairwise_px", "outcome_cov_std"]
PRIMARY_SCORE = "ace"
PRIMARY_OUTCOME = "outcome_kp_dispersion_px"


def describe(df: pd.DataFrame) -> dict:
    contact = df.frac_samples_with_contact > 0.5
    q = lambda c, s=None: {
        "p10": float((s if s is not None else df)[c].quantile(0.10)),
        "p50": float((s if s is not None else df)[c].median()),
        "p90": float((s if s is not None else df)[c].quantile(0.90)),
        "max": float((s if s is not None else df)[c].max()),
    }
    return {
        "n_probe_states": int(len(df)),
        "n_rollouts": int(df["rollout"].nunique()),
        "probes_per_rollout_mean": float(df.groupby("rollout").size().mean()),
        "rollout_success_rate": float(df.groupby("rollout").rollout_terminated.max().mean()),
        "frac_states_in_contact": float(contact.mean()),
        "action_dispersion_px": q("act_rms_dispersion"),
        "ace_bits": q("ace"),
        "outcome_kp_dispersion_px": q("outcome_kp_dispersion_px"),
        "block_mean_shift_px": q("outcome_kp_mean_shift_px"),
        "in_contact": {
            "n": int(contact.sum()),
            "action_dispersion_px": q("act_rms_dispersion", df[contact]),
            "outcome_kp_dispersion_px": q("outcome_kp_dispersion_px", df[contact]),
            "block_mean_shift_px": q("outcome_kp_mean_shift_px", df[contact]),
            # how much of the block's actual displacement is *not* shared across the
            # sampled chunks -- i.e. how goal-equivalent the sampled diversity is
            "dispersion_over_mean_shift_median": float(
                (
                    df[contact].outcome_kp_dispersion_px
                    / df[contact].outcome_kp_mean_shift_px.clip(lower=1e-9)
                ).median()
            ),
        },
        "frac_states_zero_outcome_dispersion": float((df.outcome_kp_dispersion_px < 1e-9).mean()),
        "frac_states_ace_exactly_zero": float((df.ace == 0).mean()),
    }


def score_outcome_table(df: pd.DataFrame, boot: int, seed: int) -> list[dict]:
    rows = []
    for s in SCORES:
        if s not in df.columns:
            continue
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


def within_rollout_reduction(df: pd.DataFrame, tol_z: float, seed: int) -> dict:
    """`matched_pair_reduction` computed inside each rollout and pooled by pair count.

    Probe states in one rollout share an episode, a block, and a goal. If the conflation
    only appears when pooling across rollouts, it is a between-episode effect, not a
    property of the policy's conditional action distribution.
    """
    nums, dens, npairs = [], [], 0
    for _, g in df.groupby("rollout"):
        if len(g) < 8:
            continue
        r = matched_pair_reduction(
            g, PRIMARY_SCORE, PRIMARY_OUTCOME, tol_z=tol_z, seed=seed,
            require_different_rollout=False,
        )
        if not np.isfinite(r.get("reduction", np.nan)):
            continue
        nums.append(r["median_abs_outcome_diff_matched_px"] * r["n_pairs"])
        dens.append(r["median_abs_outcome_diff_random_px"] * r["n_pairs"])
        npairs += r["n_pairs"]
    if not nums:
        return {"n_pairs": 0, "reduction": float("nan")}
    return {"n_pairs": npairs, "reduction": float(sum(nums) / max(sum(dens), 1e-12))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reliability", type=Path, default=None,
                   help="reliability.py output; required by the gate's noise-floor clause")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--match-tol-z", type=float, default=0.10)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--gate", type=Path, default=None)
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    if df.duplicated(subset=["rollout", "probe"]).any():
        raise ValueError("duplicate (rollout, probe) across inputs -- use disjoint seed bases")

    contact = df.frac_samples_with_contact > 0.5
    report: dict = {
        "inputs": [str(c) for c in args.csv],
        "descriptive": describe(df),
        "score_vs_outcome": score_outcome_table(df, args.bootstrap, args.seed),
    }

    # ---- primary, non-circular ------------------------------------------------------
    mpr = matched_pair_reduction(df, PRIMARY_SCORE, PRIMARY_OUTCOME,
                                 tol_z=args.match_tol_z, seed=args.seed)
    mpr["reduction_ci95"] = rollout_bootstrap(
        df,
        lambda d: matched_pair_reduction(d, PRIMARY_SCORE, PRIMARY_OUTCOME,
                                         tol_z=args.match_tol_z, seed=args.seed)["reduction"],
        n=args.bootstrap, seed=args.seed,
    )["ci95"]
    report["matched_pair_reduction"] = mpr
    report["matched_pair_reduction_within_rollout"] = within_rollout_reduction(
        df, args.match_tol_z, args.seed
    )
    report["matched_pair_reduction_in_contact"] = (
        matched_pair_reduction(df[contact], PRIMARY_SCORE, PRIMARY_OUTCOME,
                               tol_z=args.match_tol_z, seed=args.seed)
        if contact.sum() >= 30 else {"n_pairs": 0, "reduction": float("nan")}
    )

    # ---- supporting descriptives ----------------------------------------------------
    bs = binned_spread(df, PRIMARY_SCORE, PRIMARY_OUTCOME, n_bins=args.bins)
    report["binned_spread_ace"] = bs.to_dict(orient="records")
    report["median_outcome_iqr_within_ace_bin_px"] = (
        float(bs.outcome_iqr_px.median()) if len(bs) else None
    )

    mpd = matched_pairs_descriptive(df, PRIMARY_SCORE, PRIMARY_OUTCOME, tol_z=args.match_tol_z)
    report["matched_pairs_descriptive_NOT_EVIDENCE"] = {
        "note": "pairs are selected by outcome quartile; the ratio is large by "
                "construction and is never gated on",
        "n_pairs": int(len(mpd)),
        "median_outcome_diff_px": float(mpd["diff"].median()) if len(mpd) else None,
    }

    report["contact_stratified"] = {}
    for name, sub in (("in_contact", df[contact]), ("no_contact", df[~contact])):
        if len(sub) < 20:
            report["contact_stratified"][name] = {"n": int(len(sub)), "note": "too few states"}
            continue
        b = binned_spread(sub, PRIMARY_SCORE, PRIMARY_OUTCOME, n_bins=max(3, args.bins // 2))
        report["contact_stratified"][name] = {
            "n": int(len(sub)),
            "spearman_ace_outcome": spearman(sub[PRIMARY_SCORE], sub[PRIMARY_OUTCOME]),
            "median_outcome_iqr_within_ace_bin_px": float(b.outcome_iqr_px.median()) if len(b) else None,
        }

    if args.reliability is not None:
        report["reliability"] = json.loads(args.reliability.read_text())

    # ---- gate ------------------------------------------------------------------------
    if args.gate is not None:
        report.update(apply_gate(report, json.loads(args.gate.read_text())))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


def apply_gate(report: dict, gate: dict) -> dict:
    """Every kill criterion written in PUSHT_EXISTENCE_TEST.md, implemented."""
    fails = []

    sp = next(r for r in report["score_vs_outcome"]
              if r["score"] == PRIMARY_SCORE and r["outcome"] == PRIMARY_OUTCOME)

    # K1 the scalar already determines the outcome (pooled)
    if sp["spearman_pooled"] >= gate["kill_if_spearman_at_least"]:
        fails.append("KILL_scalar_entropy_determines_outcome_pooled")

    # K2 ... or within rollouts
    if sp["spearman_within_rollout"] >= gate["kill_if_spearman_at_least"]:
        fails.append("KILL_scalar_entropy_determines_outcome_within_rollout")

    # K3 primary non-circular statistic: matching on the scalar must not shrink the
    #    outcome difference much
    mpr = report["matched_pair_reduction"]
    if mpr["n_pairs"] < gate["min_matched_pairs"]:
        fails.append("KILL_insufficient_matched_pairs")
    if not np.isfinite(mpr["reduction"]) or mpr["reduction"] < gate["min_matched_pair_reduction"]:
        fails.append("KILL_matching_on_entropy_removes_the_outcome_difference")
    lo = mpr.get("reduction_ci95", [None, None])[0]
    if lo is None or lo < gate["min_matched_pair_reduction_ci_lower"]:
        fails.append("KILL_reduction_not_robust_to_rollout_bootstrap")

    # K4 documented: effect present pooled but gone within rollouts
    wr = report["matched_pair_reduction_within_rollout"]
    if wr["n_pairs"] < gate["min_within_rollout_pairs"]:
        fails.append("KILL_insufficient_within_rollout_pairs")
    elif not np.isfinite(wr["reduction"]) or wr["reduction"] < gate["min_matched_pair_reduction"]:
        fails.append("KILL_effect_disappears_within_rollouts")

    # K5 documented: residual fully explained by contact state
    ic = report["matched_pair_reduction_in_contact"]
    if ic["n_pairs"] < gate["min_in_contact_pairs"]:
        fails.append("KILL_insufficient_in_contact_pairs")
    elif not np.isfinite(ic["reduction"]) or ic["reduction"] < gate["min_matched_pair_reduction"]:
        fails.append("KILL_residual_explained_by_contact_state")
    ic_spread = report["contact_stratified"].get("in_contact", {}).get(
        "median_outcome_iqr_within_ace_bin_px")
    if ic_spread is None or ic_spread < gate["min_in_contact_outcome_iqr_px"]:
        fails.append("KILL_no_outcome_spread_at_matched_entropy_within_contact")

    # K6 documented: the effect must not depend on the ACE cell-size constant
    rel = report.get("reliability")
    if rel is None:
        fails.append("KILL_reliability_report_missing")
    else:
        for row in rel["estimator_sweep"]:
            if row["spearman_ace_outcome_in_contact"] >= gate["kill_if_spearman_at_least"]:
                fails.append(
                    f"KILL_scalar_determines_outcome_at_cellsize_factor_{row['cellsize_factor']}"
                )
        # K7 the residual must be signal, not finite-B estimator noise
        sh = rel["split_half"]
        if sh["reliability_outcome_dispersion"] < gate["min_outcome_reliability"]:
            fails.append("KILL_outcome_dispersion_not_reliably_measured")
        if sh["noise_to_between_state_iqr"] > gate["max_noise_to_iqr"]:
            fails.append("KILL_residual_spread_is_within_finite_B_noise")

    return {"gate": gate, "failed_clauses": fails,
            "verdict": "CONTINUE" if not fails else "KILL"}


if __name__ == "__main__":
    main()
