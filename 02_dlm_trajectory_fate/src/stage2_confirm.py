from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from fate_labels import fate_from_correctness
from io_utils import load_shards

SPECS = {
    "llada": {
        "transient_recovery": {"step": 16, "layer": 25, "lead": 4},
        "transient_overwrite": {"step": 4, "layer": 28, "lead": 16},
        "positive_control_layers": [25, 28],
    },
    # Relative-depth mapping from LLaDA 32 blocks to Dream 28 blocks:
    # 25/32 -> round(.78125*28)=22, 28/32 -> round(.875*28)=25.
    "dream": {
        "transient_recovery": {"step": 16, "layer": 22, "lead": 4},
        "transient_overwrite": {"step": 4, "layer": 25, "lead": 16},
        "positive_control_layers": [22, 25],
    },
}


def hidden_pipeline(dim: int, n_train: int):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=min(64, dim, max(1, n_train - 1))),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def surface_pipeline():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def cv_splits(y: np.ndarray, folds: int):
    counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(folds, int(counts.min()))
    if n_splits < 2:
        return []
    return list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
            np.zeros(len(y)), y
        )
    )


def oof_predict(X: np.ndarray, y: np.ndarray, splits, hidden: bool) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=np.float64)
    for tr, te in splits:
        clf = hidden_pipeline(X.shape[1], len(tr)) if hidden else surface_pipeline()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(p)
    if ok.sum() < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], p[ok]))


def surface_features(data: dict, labels: dict, si: int, idx: np.ndarray) -> np.ndarray:
    current_correct = labels["current_correct"][idx, si].astype(np.float32)
    current_correct = np.where(current_correct < 0, 0.0, current_correct)
    return np.column_stack(
        [
            np.asarray(data["entropy"])[idx, si],
            np.asarray(data["selected_prob"])[idx, si],
            np.asarray(data["clean_maxprob"])[idx, si],
            np.asarray(data["frac_unmasked"])[idx, si],
            np.asarray(data["prompt_tokens"])[idx],
            labels["current_observed"][idx, si].astype(np.float32),
            current_correct,
        ]
    ).astype(np.float32)


def bootstrap_confirmation(
    y: np.ndarray,
    current: np.ndarray,
    surface: np.ndarray,
    initial: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    auc = safe_auc(y, current)
    surface_auc = safe_auc(y, surface)
    initial_auc = safe_auc(y, initial)
    delta_surface = auc - surface_auc
    delta_initial = auc - initial_auc
    margin = min(auc - 0.55, delta_surface - 0.03, delta_initial - 0.03)

    rng = np.random.default_rng(seed)
    vals = {"auc": [], "delta_surface": [], "delta_initial": [], "margin": []}
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(y), len(y))
        if np.unique(y[sample]).size < 2:
            continue
        a = safe_auc(y[sample], current[sample])
        s = safe_auc(y[sample], surface[sample])
        i = safe_auc(y[sample], initial[sample])
        if not all(np.isfinite(v) for v in (a, s, i)):
            continue
        ds = a - s
        di = a - i
        vals["auc"].append(a)
        vals["delta_surface"].append(ds)
        vals["delta_initial"].append(di)
        vals["margin"].append(min(a - 0.55, ds - 0.03, di - 0.03))

    out = {
        "auc": auc,
        "surface_auc": surface_auc,
        "initial_auc": initial_auc,
        "delta_vs_surface": delta_surface,
        "delta_vs_initial": delta_initial,
        "confirmation_margin": margin,
    }
    for key, samples in vals.items():
        name = "confirmation_margin" if key == "margin" else key
        if len(samples) < 100:
            out[name + "_lo_97_5"] = float("nan")
            out[name + "_hi_97_5"] = float("nan")
        else:
            # One-sided 97.5% lower bound: Bonferroni alpha=.025 for two locked tasks.
            out[name + "_lo_97_5"] = float(np.percentile(samples, 2.5))
            out[name + "_hi_97_5"] = float(np.percentile(samples, 97.5))
    return out


def evaluate_locked_task(
    data: dict,
    labels: dict,
    task: str,
    spec: dict,
    folds: int,
    bootstrap: int,
) -> dict:
    steps = np.asarray(data["capture_steps"]).tolist()
    layers = np.asarray(data["hidden_indices"]).tolist()
    if spec["step"] not in steps:
        raise ValueError(f"missing locked step {spec['step']}")
    if spec["layer"] not in layers:
        raise ValueError(f"missing locked layer {spec['layer']}")
    si = steps.index(spec["step"])
    zero_i = steps.index(0)
    li = layers.index(spec["layer"])

    y = labels[task][:, si]
    lead_key = "recovery_lead" if task == "transient_recovery" else "overwrite_lead"
    lead = labels[lead_key][:, si]
    valid = y >= 0
    keep = valid & ((y == 0) | ((y == 1) & (lead >= spec["lead"])))
    idx = np.flatnonzero(keep)
    yy = y[idx].astype(int)
    counts = np.bincount(yy, minlength=2) if len(yy) else np.zeros(2, dtype=int)

    row = {
        "task": task,
        "step": spec["step"],
        "layer": spec["layer"],
        "min_lead": spec["lead"],
        "n": int(len(yy)),
        "positive": int(counts[1]),
        "negative": int(counts[0]),
    }
    if counts.min() < 2:
        row.update({"status": "LOW_SUPPORT"})
        return row

    hidden = np.asarray(data["hidden"])
    X = hidden[idx, si, li].astype(np.float32).mean(axis=1)
    X0 = hidden[idx, zero_i, li].astype(np.float32).mean(axis=1)
    Xs = surface_features(data, labels, si, idx)
    splits = cv_splits(yy, folds)
    current_pred = oof_predict(X, yy, splits, hidden=True)
    initial_pred = oof_predict(X0, yy, splits, hidden=True)
    surface_pred = oof_predict(Xs, yy, splits, hidden=False)
    row.update(
        bootstrap_confirmation(
            yy,
            current_pred,
            surface_pred,
            initial_pred,
            bootstrap,
            seed=2003 + spec["step"] * 19 + spec["layer"],
        )
    )
    return row


def evaluate_positive_control(data: dict, labels: dict, model_family: str, folds: int) -> list[dict]:
    steps = np.asarray(data["capture_steps"]).tolist()
    layers = np.asarray(data["hidden_indices"]).tolist()
    if 63 not in steps or 0 not in steps:
        return []
    si = steps.index(63)
    zero_i = steps.index(0)
    valid = labels["final_observed"][:, si] == 1
    idx = np.flatnonzero(valid)
    y = labels["final_correct"][idx, si].astype(int)
    splits = cv_splits(y, folds)
    rows = []
    hidden = np.asarray(data["hidden"])
    for layer in SPECS[model_family]["positive_control_layers"]:
        if layer not in layers:
            continue
        li = layers.index(layer)
        X = hidden[idx, si, li].astype(np.float32).mean(axis=1)
        X0 = hidden[idx, zero_i, li].astype(np.float32).mean(axis=1)
        p = oof_predict(X, y, splits, hidden=True)
        p0 = oof_predict(X0, y, splits, hidden=True)
        auc = safe_auc(y, p)
        init_auc = safe_auc(y, p0)
        rows.append(
            {
                "layer": layer,
                "step": 63,
                "n": int(len(y)),
                "auc": auc,
                "initial_auc": init_auc,
                "delta_vs_initial": auc - init_auc,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Locked G1 confirmation; no step/layer/lead search")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-family", choices=sorted(SPECS), required=True)
    p.add_argument("--mode", choices=["audit", "confirm", "model_replication"], default="confirm")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--min-class-count", type=int, default=25)
    args = p.parse_args()

    data = load_shards(Path(args.input_dir), require_hidden=True)
    labels = fate_from_correctness(
        np.asarray(data["correct_strict"]),
        np.asarray(data["capture_steps"]),
        np.asarray(data["observed_strict"]),
    )

    rows = []
    for task in ["transient_recovery", "transient_overwrite"]:
        row = evaluate_locked_task(
            data, labels, task, SPECS[args.model_family][task], args.folds, args.bootstrap
        )
        enough = min(row["positive"], row["negative"]) >= args.min_class_count
        if row.get("status") != "LOW_SUPPORT":
            row["support_ok"] = enough
            if args.mode == "audit":
                row["confirmed"] = bool(
                    enough
                    and row["auc"] > 0.5
                    and row["delta_vs_surface"] > 0
                    and row["delta_vs_initial"] > 0
                )
            else:
                row["confirmed"] = bool(
                    enough and row["confirmation_margin_lo_97_5"] > 0
                )
        else:
            row["support_ok"] = False
            row["confirmed"] = False
        rows.append(row)

    pc = evaluate_positive_control(data, labels, args.model_family, args.folds)
    positive_control_ok = any(
        r["auc"] >= 0.65 and r["delta_vs_initial"] >= 0.03 for r in pc
    )
    n_confirmed = sum(bool(r["confirmed"]) for r in rows)
    n_supported = sum(bool(r["support_ok"]) for r in rows)
    if args.mode == "audit":
        if n_supported == 0:
            status = "AUDIT_LOW_SUPPORT"
        elif n_confirmed == 2:
            status = "AUDIT_BOTH_DIRECTIONAL"
        elif n_confirmed == 1:
            status = "AUDIT_ONE_DIRECTIONAL"
        else:
            status = "AUDIT_NONE_DIRECTIONAL"
    elif not positive_control_ok:
        status = "GEOMETRY_NOT_VALIDATED"
    elif n_confirmed == 2:
        status = "CONFIRM_BOTH"
    elif n_confirmed == 1:
        status = "CONFIRM_ONE"
    elif n_supported == 1:
        status = "LOW_SUPPORT_ONE_TASK"
    elif n_supported == 0:
        status = "LOW_SUPPORT_BOTH"
    else:
        status = "FAIL_BOTH"

    result = {
        "status": status,
        "mode": args.mode,
        "model_family": args.model_family,
        "metadata": data["metadata"],
        "criterion": {
            "point_effect_floor": {
                "auc": 0.55,
                "delta_vs_surface": 0.03,
                "delta_vs_initial": 0.03,
            },
            "confirmation": "one-sided 97.5% bootstrap lower bound of min(AUC-.55, dSurface-.03, dStep0-.03) > 0",
            "multiplicity": "Bonferroni alpha=.025 across the two locked G0 hypotheses",
            "min_class_count": args.min_class_count,
        },
        "positive_control_ok": positive_control_ok,
        "positive_control": pc,
        "tasks": rows,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "locked_confirmation.csv", index=False)
    (out / "locked_confirmation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
