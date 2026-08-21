from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_subjects(path: str) -> List[dict]:
    subjects = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                subjects.append(json.loads(line))
    return subjects


def subject_metrics(subjects: List[dict]) -> pd.DataFrame:
    rows = []
    for s in subjects:
        steps = pd.DataFrame(s["steps"])
        train = steps[steps.phase == "train"]
        test = steps[steps.phase == "test"]
        max_epi = int(train.episode_idx.max())
        late = train[train.episode_idx >= max(0, max_epi - 2)]
        late_first = (
            late.sort_values(["episode_idx", "step_idx"])
            .groupby("episode_idx", as_index=False)
            .first()
        )
        first = test.sort_values(["episode_idx", "step_idx"]).iloc[0]
        first3 = test.sort_values("step_idx").head(3)
        valid_active = test[test["active"]]
        rows.append(
            {
                "base_seed": s["base_seed"],
                "diversity": s["diversity"],
                "history_controllability": s["history_controllability"],
                "uncontrollable": int(s["history_controllability"] == "uncontrollable"),
                "distributed": int(s["diversity"] == "distributed"),
                "test_family": s["test_family"],
                "invalid_rate": 1.0 - float(steps.format_valid.mean()),
                "train_late_active": float(late.active.mean()),
                "train_late_episode_first_active": float(late_first.active.mean()),
                "train_mean_abs_state": float(train.state_after.abs().mean()),
                "test_step1_active": float(first.active),
                "test_first3_active": float(first3.active.mean()),
                "test_active_rate": float(test.active.mean()),
                "test_mean_abs_state": float(test.state_after.abs().mean()),
                "test_active_improvement_rate": float(valid_active.improved.mean()) if len(valid_active) else np.nan,
                "test_time_to_first_active": int(test.loc[test.active, "step_idx"].min() + 1) if test.active.any() else int(len(test) + 1),
            }
        )
    return pd.DataFrame(rows)


def cell_mean(df: pd.DataFrame, metric: str, u: int, d: int) -> float:
    x = df[(df.uncontrollable == u) & (df.distributed == d)][metric]
    return float(x.mean())


def contrasts(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    c1 = cell_mean(df, metric, 0, 0)
    c10 = cell_mean(df, metric, 0, 1)
    u1 = cell_mean(df, metric, 1, 0)
    u10 = cell_mean(df, metric, 1, 1)
    return {
        "C_concentrated": c1,
        "C_distributed": c10,
        "U_concentrated": u1,
        "U_distributed": u10,
        "pooled_U_minus_C": 0.5 * (u1 + u10) - 0.5 * (c1 + c10),
        "diversity_interaction": (u10 - u1) - (c10 - c1),
    }


def bootstrap_by_seed(df: pd.DataFrame, metric: str, n_boot: int = 5000, seed: int = 0) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(df.base_seed.unique()))
    pooled, inter = [], []
    for _ in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        chunks = []
        for new_id, s in enumerate(sampled):
            tmp = df[df.base_seed == s].copy()
            tmp["boot_seed"] = new_id
            chunks.append(tmp)
        b = pd.concat(chunks, ignore_index=True)
        c = contrasts(b, metric)
        pooled.append(c["pooled_U_minus_C"])
        inter.append(c["diversity_interaction"])
    return {
        "pooled_U_minus_C_ci95": (float(np.quantile(pooled, 0.025)), float(np.quantile(pooled, 0.975))),
        "diversity_interaction_ci95": (float(np.quantile(inter, 0.025)), float(np.quantile(inter, 0.975))),
    }


def fit_cluster_logit(df: pd.DataFrame) -> Dict[str, float]:
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"available": False, "reason": "statsmodels not installed"}
    try:
        X = df[["uncontrollable", "distributed"]].copy()
        X["interaction"] = X.uncontrollable * X.distributed
        X = sm.add_constant(X)
        model = sm.GLM(df.test_step1_active, X, family=sm.families.Binomial())
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": df.base_seed})
        out = {"available": True}
        for name in X.columns:
            out[f"coef_{name}"] = float(fit.params[name])
            out[f"se_{name}"] = float(fit.bse[name])
            out[f"p_{name}"] = float(fit.pvalues[name])
        return out
    except Exception as exc:
        # A perfectly separated first-action response is itself informative and
        # should not make the primary bootstrap analysis crash.
        return {"available": False, "reason": f"cluster logit failed: {type(exc).__name__}: {exc}"}


def decide(df: pd.DataFrame, summary: Dict[str, dict]) -> Dict[str, str]:
    invalid = float(df.invalid_rate.mean())
    train = contrasts(df, "train_late_episode_first_active")
    test = summary["test_step1_active"]
    train_c_minus_u = -train["pooled_U_minus_C"]
    transfer = test["pooled_U_minus_C"]
    interaction = test["diversity_interaction"]
    ci_transfer = summary["test_step1_active_ci"]["pooled_U_minus_C_ci95"]
    ci_inter = summary["test_step1_active_ci"]["diversity_interaction_ci95"]

    if invalid > 0.02:
        status = "PLUMBING_FAIL"
        reason = "More than 2% of model responses were unparsable; fix action I/O before scientific interpretation."
    elif train_c_minus_u < 0.03:
        status = "LOCAL_ACQUISITION_WEAK"
        reason = "On the first action of late training episodes, controllable histories did not produce at least a 3-point increase in intervention over yoked uncontrollable histories."
    elif transfer > -0.02 and ci_transfer[0] > -0.05:
        status = "TRANSFER_KILL"
        reason = "Little evidence that uncontrollable history suppresses the first intervention in the novel controllable task."
    elif interaction <= -0.05 and ci_inter[1] < 0:
        status = "DIVERSITY_EFFECT_SUPPORTED"
        reason = "Distributed uncontrollability suppresses novel-task intervention beyond the generic effect of task diversity."
    elif interaction >= 0.05 and ci_inter[0] > 0:
        status = "DIVERSITY_WRONG_DIRECTION"
        reason = "The diversity interaction is reliably opposite to the proposed worldview-generalization hypothesis."
    elif ci_inter[0] >= -0.02 and ci_inter[1] <= 0.02:
        status = "DIVERSITY_NEGLIGIBLE"
        reason = "The locked 95% interval lies inside a +/-2 point ROPE; the diversity effect is too small to support this candidate."
    else:
        status = "INCONCLUSIVE_SCALE_OR_REPLICATE"
        reason = "Core transfer is alive, but the diversity interaction is not yet decisively separated from zero."
    return {"status": status, "reason": reason}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--outdir", default="results/analysis")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = subject_metrics(load_subjects(args.input))
    df.to_csv(outdir / "subject_metrics.csv", index=False)

    summary: Dict[str, dict] = {
        "n_subjects": int(len(df)),
        "n_base_seeds": int(df.base_seed.nunique()),
        "mean_invalid_rate": float(df.invalid_rate.mean()),
    }
    for metric in ("train_late_active", "train_late_episode_first_active", "test_step1_active", "test_first3_active", "test_active_rate", "test_mean_abs_state"):
        summary[metric] = contrasts(df, metric)
        summary[f"{metric}_ci"] = bootstrap_by_seed(df, metric, args.bootstrap, args.seed)
    summary["cluster_logit_test_step1"] = fit_cluster_logit(df)
    summary["decision"] = decide(df, summary)

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
