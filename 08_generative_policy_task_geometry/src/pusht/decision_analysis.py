"""Does a runtime entropy monitor actually make wrong calls because of this?

"Action entropy and task-outcome uncertainty are not perfectly correlated" is not a
finding. Robot dynamics are nonlinear and state-dependent, so of course a pusher flailing
in free space is diverse-and-harmless while a millimetre near contact matters. Stated
that way the phenomenon is unsurprising and the topic should be archived.

The version that would matter is stronger and operational:

    a deployed uncertainty monitor that thresholds scalar action entropy fires on states
    where every sampled action leads to the same place, and stays quiet on states where
    the sampled actions genuinely diverge.

That is a semantic mismatch in a mechanism people actually run, not a geometric curiosity.
This module measures it directly against episode-level branch outcomes from `collect_e1b`.

FIPER's operating points are taken from its released config (`configs/eval/base.yaml`):
constant thresholds set at calibration quantiles 0.90-0.99 of the score. We report:

  * AUC of each score for ranking states by true functional uncertainty. 0.5 is chance --
    a monitor at chance is not a conservative monitor, it is a broken one.
  * precision at each operating point against the base rate. If precision ~ base rate,
    the alarms carry no information.
  * the two error types in plain terms: alarms raised on states where the branches all
    end in the same place, and silence on states where they do not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .analysis import rollout_bootstrap, spearman

SCORES = ["ace", "act_rms_dispersion", "act_trace_cov_mean"]
OUTCOME = "branch_final_kp_dispersion_px"
FIPER_QUANTILES = [0.90, 0.95, 0.99]


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    if label.min() == label.max():
        return float("nan")
    return float(roc_auc_score(label, score))


def monitor_table(df: pd.DataFrame, outcome: str = OUTCOME) -> list[dict]:
    """One row per (score, operating point): what the monitor would actually do."""
    o = df[outcome].to_numpy(float)
    truly_uncertain = o >= np.quantile(o, 0.75)      # top-quartile functional uncertainty
    base_rate = float(truly_uncertain.mean())
    # "the branches all ended in the same place": dispersion in the bottom quartile
    benign = o <= np.quantile(o, 0.25)

    rows = []
    for s in SCORES:
        if s not in df.columns:
            continue
        v = df[s].to_numpy(float)
        auc = _auc(v, truly_uncertain)
        for q in FIPER_QUANTILES:
            thr = float(np.quantile(v, q))
            flag = v > thr
            n_flag = int(flag.sum())
            rows.append(
                {
                    "score": s,
                    "quantile": q,
                    "threshold": thr,
                    "auc_vs_true_uncertainty": auc,
                    "base_rate_top_quartile": base_rate,
                    "n_flagged": n_flag,
                    # of the states the monitor stops on, how many actually mattered
                    "precision": float(truly_uncertain[flag].mean()) if n_flag else float("nan"),
                    "precision_over_base_rate": (
                        float(truly_uncertain[flag].mean() / base_rate)
                        if n_flag and base_rate > 0 else float("nan")
                    ),
                    # of the states it stops on, how many were completely benign
                    "frac_alarms_on_benign_states": (
                        float(benign[flag].mean()) if n_flag else float("nan")
                    ),
                    # of the genuinely uncertain states, how many it never flags
                    "miss_rate_on_truly_uncertain": float((~flag)[truly_uncertain].mean()),
                }
            )
    return rows


def describe(df: pd.DataFrame, outcome: str = OUTCOME) -> dict:
    o = df[outcome]
    q = lambda c: {
        "p10": float(df[c].quantile(0.10)),
        "p50": float(df[c].median()),
        "p90": float(df[c].quantile(0.90)),
        "max": float(df[c].max()),
    }
    hi_ace = df.ace >= df.ace.quantile(0.75)
    lo_ace = df.ace <= df.ace.quantile(0.25)
    lo_out = o <= o.quantile(0.25)
    hi_out = o >= o.quantile(0.75)
    return {
        "n_branch_states": int(len(df)),
        "n_rollouts": int(df.rollout.nunique()),
        "branches_per_state": int(df.branches.iloc[0]),
        "extra_steps": int(df.extra_steps.iloc[0]),
        "ace_bits": q("ace"),
        "action_dispersion_px": q("act_rms_dispersion"),
        "branch_final_kp_dispersion_px": q(outcome),
        "branch_final_cov_std": q("branch_final_cov_std"),
        # the two cells of the 2x2 that the topic lives or dies on
        "frac_high_entropy_low_outcome": float((hi_ace & lo_out).mean()),
        "frac_low_entropy_high_outcome": float((lo_ace & hi_out).mean()),
        "frac_states_branches_disagree_on_goal": float((df.branch_goal_disagreement > 0).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    if df.duplicated(subset=["rollout", "probe"]).any():
        raise ValueError("duplicate (rollout, probe) -- shards must use disjoint seed bases")

    o = df[OUTCOME].to_numpy(float)
    truly = o >= np.quantile(o, 0.75)

    report = {
        "inputs": [str(c) for c in args.csv],
        "descriptive": describe(df),
        "monitor": monitor_table(df),
        "spearman_ace_vs_branch_outcome": rollout_bootstrap(
            df, lambda d: spearman(d.ace, d[OUTCOME]), n=args.bootstrap, seed=args.seed
        ),
        "auc_ace_vs_true_uncertainty": rollout_bootstrap(
            df,
            lambda d: _auc(
                d.ace.to_numpy(float),
                d[OUTCOME].to_numpy(float) >= np.quantile(d[OUTCOME].to_numpy(float), 0.75),
            ),
            n=args.bootstrap, seed=args.seed,
        ),
    }
    # contact stratification: is the whole thing just free space vs contact?
    contact = df.agent_block_gap_px < 20.0
    report["contact_stratified"] = {}
    for name, sub in (("near_block", df[contact]), ("far_from_block", df[~contact])):
        if len(sub) < 30:
            report["contact_stratified"][name] = {"n": int(len(sub)), "note": "too few states"}
            continue
        so = sub[OUTCOME].to_numpy(float)
        report["contact_stratified"][name] = {
            "n": int(len(sub)),
            "spearman_ace_outcome": spearman(sub.ace, sub[OUTCOME]),
            "auc_ace_vs_true_uncertainty": _auc(
                sub.ace.to_numpy(float), so >= np.quantile(so, 0.75)
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
