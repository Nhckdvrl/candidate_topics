from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--eval-json", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--match-z", type=float, default=0.15,
                   help="max absolute ACE difference after z-scoring")
    return p.parse_args()


def corr(x: np.ndarray, y: np.ndarray) -> float:
    r = spearmanr(x, y).statistic
    return float(0.0 if np.isnan(r) else r)


def state_auc(score: np.ndarray, risk: np.ndarray) -> float:
    y = risk > np.median(risk)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, score))


def matched_entropy_effect(df: pd.DataFrame, max_z: float = 0.15) -> dict:
    ace = df["ace"].to_numpy(float)
    tf = df["task_fraction"].to_numpy(float)
    risk = df["risk"].to_numpy(float)
    z = (ace - ace.mean()) / max(ace.std(ddof=1), 1e-8)
    lo_idx = np.where(tf <= np.quantile(tf, 0.25))[0]
    hi_idx = np.where(tf >= np.quantile(tf, 0.75))[0]
    candidates = []
    for hi in hi_idx:
        for lo in lo_idx:
            dz = abs(z[hi] - z[lo])
            if dz <= max_z:
                candidates.append((dz, hi, lo))
    candidates.sort(key=lambda x: x[0])
    used_hi, used_lo, pairs = set(), set(), []
    for dz, hi, lo in candidates:
        if hi in used_hi or lo in used_lo:
            continue
        used_hi.add(hi)
        used_lo.add(lo)
        pairs.append((dz, hi, lo))
    if not pairs:
        return {
            "n_pairs": 0,
            "mean_abs_ace_z_gap": None,
            "risk_diff_high_minus_low": None,
            "task_fraction_diff": None,
        }
    return {
        "n_pairs": len(pairs),
        "mean_abs_ace_z_gap": float(np.mean([p[0] for p in pairs])),
        "risk_diff_high_minus_low": float(np.mean([risk[h] - risk[l] for _, h, l in pairs])),
        "task_fraction_diff": float(np.mean([tf[h] - tf[l] for _, h, l in pairs])),
    }


def bootstrap(df: pd.DataFrame, n: int, seed: int, max_z: float) -> dict:
    rng = np.random.default_rng(seed)
    nrow = len(df)
    diffs, corrdiffs, ratios = [], [], []
    for _ in range(n):
        b = df.iloc[rng.integers(0, nrow, size=nrow)].reset_index(drop=True)
        m = matched_entropy_effect(b, max_z=max_z)
        if m["n_pairs"] >= 10:
            diffs.append(m["risk_diff_high_minus_low"])
        corrdiffs.append(
            corr(b.task_var.to_numpy(), b.risk.to_numpy())
            - corr(b.ace.to_numpy(), b.risk.to_numpy())
        )
        ratios.append(float(np.median(b.null_var / np.maximum(b.task_var, 1e-12))))

    def ci(x):
        x = np.asarray(x, dtype=float)
        return [float(np.quantile(x, .025)), float(np.quantile(x, .975))] if len(x) else [None, None]

    return {
        "matched_risk_diff_ci95": ci(diffs),
        "corr_advantage_ci95": ci(corrdiffs),
        "median_null_task_ratio_ci95": ci(ratios),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    required = {"ace", "task_var", "null_var", "task_fraction", "risk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    match = matched_entropy_effect(df, max_z=args.match_z)
    eval_summary = {}
    if args.eval_json is not None:
        eval_summary = json.loads(args.eval_json.read_text())

    metrics = {
        "n_states": int(len(df)),
        "rollout_success": eval_summary.get("rollout_success"),
        "median_risk": float(df.risk.median()),
        "median_task_var": float(df.task_var.median()),
        "median_null_var": float(df.null_var.median()),
        "median_null_task_ratio": float(np.median(df.null_var / np.maximum(df.task_var, 1e-12))),
        "spearman_ace_risk": corr(df.ace.to_numpy(), df.risk.to_numpy()),
        "spearman_taskvar_risk": corr(df.task_var.to_numpy(), df.risk.to_numpy()),
        "spearman_nullvar_risk": corr(df.null_var.to_numpy(), df.risk.to_numpy()),
        "auc_ace_highrisk": state_auc(df.ace.to_numpy(), df.risk.to_numpy()),
        "auc_taskvar_highrisk": state_auc(df.task_var.to_numpy(), df.risk.to_numpy()),
        "matched_entropy": match,
    }
    metrics["spearman_task_minus_ace"] = (
        metrics["spearman_taskvar_risk"] - metrics["spearman_ace_risk"]
    )
    metrics.update(bootstrap(df, args.bootstrap, args.seed, args.match_z))

    reasons = []
    if metrics["rollout_success"] is not None and metrics["rollout_success"] < 0.80:
        reasons.append("G0_FAIL_policy_rollout_success_below_0.80")
    if metrics["median_null_task_ratio"] < 0.75:
        reasons.append("G1_FAIL_policy_diversity_not_preferentially_task_null")
    if match["n_pairs"] < 30:
        reasons.append("G2_FAIL_insufficient_matched_entropy_pairs")
    if match["n_pairs"] >= 30 and match["risk_diff_high_minus_low"] < 0.10:
        reasons.append("G3_FAIL_matched_entropy_geometry_has_small_risk_effect")
    ci = metrics["matched_risk_diff_ci95"]
    if ci[0] is not None and ci[0] <= 0.0:
        reasons.append("G3_FAIL_matched_entropy_risk_effect_not_bootstrap_positive")

    metrics["verdict"] = "GO_TO_FRANKA" if not reasons else "STOP_OR_REDESIGN_G0"
    metrics["failed_clauses"] = reasons
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
