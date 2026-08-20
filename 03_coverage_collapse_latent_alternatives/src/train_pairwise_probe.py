from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description="Confirm branch-specific latent viability at a locked layer across early/late checkpoints.")
    p.add_argument("--input-dir", default="artifacts/states")
    p.add_argument("--reference-tag", required=True, help="Early high-coverage checkpoint selected by behavior preflight.")
    p.add_argument("--late-tag", default="e16")
    p.add_argument("--control-tag", default=None, help="Optional target-blind state file stem, e.g. e02_target_blind.")
    p.add_argument("--output", default="artifacts/latent_gate_metrics.csv")
    p.add_argument("--predictions-output", default="artifacts/latent_gate_predictions.csv")
    p.add_argument("--discovery-frac", type=float, default=0.6)
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_state(path: Path):
    d = np.load(path, allow_pickle=False)
    return {
        "tag": str(d["tag"].item()),
        "pid": d["problem_id"].astype(int),
        "y": d["label_a_viable"].astype(int),
        "hidden": d["hidden"].astype(np.float32),
        "de": d["candidate_embedding_diff"].astype(np.float32),
        "margin": d["output_logprob_margin_a_minus_b"].astype(np.float32),
        "true_margin": d["output_true_viable_margin"].astype(np.float32),
    }


def candidate_features(d, layer):
    return d["hidden"][:, layer, :] * d["de"]


def make_probe(dim: int, n_train: int, pca_dim: int):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=min(pca_dim, dim, max(1, n_train - 1)), random_state=0),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
    )


def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def cv_auc(X, y, pca_dim, seed):
    counts = np.bincount(y, minlength=2)
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y):
        clf = make_probe(X.shape[1], len(tr), pca_dim)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return safe_auc(y, pred)


def bootstrap_ci(y, score, fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(y[idx], score[idx])
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def acc_fn(y, score):
    return float(accuracy_score(y, score >= 0.5))


def align(reference, other):
    if not np.array_equal(reference["pid"], other["pid"]):
        raise ValueError(f"problem_id mismatch between {reference['tag']} and {other['tag']}")
    if not np.array_equal(reference["y"], other["y"]):
        raise ValueError(f"label mismatch between {reference['tag']} and {other['tag']}")
    if reference["hidden"].shape[1:] != other["hidden"].shape[1:]:
        raise ValueError("hidden-state shape mismatch")


def main():
    args = parse_args()
    root = Path(args.input_dir)
    ref = load_state(root / f"{args.reference_tag}.npz")
    late = load_state(root / f"{args.late_tag}.npz")
    align(ref, late)

    y = ref["y"]
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=args.discovery_frac, random_state=args.seed)
    discovery, confirm = next(splitter.split(np.zeros(len(y)), y))

    layer_rows = []
    for layer in range(ref["hidden"].shape[1]):
        X = candidate_features(ref, layer)[discovery]
        auc = cv_auc(X, y[discovery], args.pca_dim, args.seed)
        layer_rows.append((layer, auc))
    selected_layer, discovery_auc = max(layer_rows, key=lambda x: (-np.inf if np.isnan(x[1]) else x[1]))

    rows = []
    preds = []
    for tag, d in ((args.reference_tag, ref), (args.late_tag, late)):
        X = candidate_features(d, selected_layer)
        clf = make_probe(X.shape[1], len(discovery), args.pca_dim)
        clf.fit(X[discovery], y[discovery])
        p = clf.predict_proba(X[confirm])[:, 1]
        yc = y[confirm]
        hidden_auc = safe_auc(yc, p)
        hidden_acc = acc_fn(yc, p)
        output_score = d["margin"][confirm]
        output_auc = safe_auc(yc, output_score)
        output_acc = float((d["true_margin"][confirm] > 0).mean())

        wrong = d["true_margin"][confirm] < 0
        strong_wrong = d["true_margin"][confirm] < -2.0
        pred_true = np.where(yc == 1, p, 1.0 - p)
        rescue_acc = float((pred_true[wrong] > 0.5).mean()) if wrong.any() else float("nan")
        strong_rescue_acc = float((pred_true[strong_wrong] > 0.5).mean()) if strong_wrong.any() else float("nan")

        auc_lo, auc_hi = bootstrap_ci(yc, p, safe_auc, args.bootstrap, args.seed + 1)
        acc_lo, acc_hi = bootstrap_ci(yc, p, acc_fn, args.bootstrap, args.seed + 2)
        if wrong.any():
            rr = (pred_true[wrong] > 0.5).astype(float)
            rng = np.random.default_rng(args.seed + 3)
            boot = [rr[rng.integers(0, len(rr), len(rr))].mean() for _ in range(args.bootstrap)]
            rescue_lo, rescue_hi = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
        else:
            rescue_lo = rescue_hi = float("nan")

        rows.append({
            "tag": tag,
            "n_discovery": len(discovery),
            "n_confirm": len(confirm),
            "selected_layer": selected_layer,
            "reference_discovery_cv_auc": discovery_auc,
            "hidden_auc": hidden_auc,
            "hidden_auc_ci_lo": auc_lo,
            "hidden_auc_ci_hi": auc_hi,
            "hidden_accuracy": hidden_acc,
            "hidden_accuracy_ci_lo": acc_lo,
            "hidden_accuracy_ci_hi": acc_hi,
            "output_margin_auc": output_auc,
            "output_choice_accuracy": output_acc,
            "output_wrong_n": int(wrong.sum()),
            "hidden_rescue_acc_on_output_wrong": rescue_acc,
            "hidden_rescue_ci_lo": rescue_lo,
            "hidden_rescue_ci_hi": rescue_hi,
            "strong_output_wrong_n": int(strong_wrong.sum()),
            "hidden_rescue_acc_on_strong_output_wrong": strong_rescue_acc,
        })
        for pid, yi, pi, m, tm in zip(d["pid"][confirm], yc, p, d["margin"][confirm], d["true_margin"][confirm]):
            preds.append({
                "tag": tag,
                "problem_id": int(pid),
                "a_is_viable": int(yi),
                "latent_p_a_viable": float(pi),
                "latent_p_true_viable": float(pi if yi else 1 - pi),
                "output_margin_a_minus_b": float(m),
                "output_true_viable_margin": float(tm),
                "output_wrong": int(tm < 0),
                "strong_output_wrong": int(tm < -2.0),
            })

    if args.control_tag:
        ctrl = load_state(root / f"{args.control_tag}.npz")
        align(ref, ctrl)
        X = candidate_features(ctrl, selected_layer)
        clf = make_probe(X.shape[1], len(discovery), args.pca_dim)
        clf.fit(X[discovery], y[discovery])
        p = clf.predict_proba(X[confirm])[:, 1]
        rows.append({
            "tag": args.control_tag,
            "n_discovery": len(discovery),
            "n_confirm": len(confirm),
            "selected_layer": selected_layer,
            "reference_discovery_cv_auc": discovery_auc,
            "hidden_auc": safe_auc(y[confirm], p),
            "hidden_auc_ci_lo": np.nan,
            "hidden_auc_ci_hi": np.nan,
            "hidden_accuracy": acc_fn(y[confirm], p),
            "hidden_accuracy_ci_lo": np.nan,
            "hidden_accuracy_ci_hi": np.nan,
            "output_margin_auc": safe_auc(y[confirm], ctrl["margin"][confirm]),
            "output_choice_accuracy": float((ctrl["true_margin"][confirm] > 0).mean()),
            "output_wrong_n": int((ctrl["true_margin"][confirm] < 0).sum()),
            "hidden_rescue_acc_on_output_wrong": np.nan,
            "hidden_rescue_ci_lo": np.nan,
            "hidden_rescue_ci_hi": np.nan,
            "strong_output_wrong_n": int((ctrl["true_margin"][confirm] < -2).sum()),
            "hidden_rescue_acc_on_strong_output_wrong": np.nan,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.DataFrame(preds).to_csv(args.predictions_output, index=False)

    ref_row = df[df.tag == args.reference_tag].iloc[0]
    late_row = df[df.tag == args.late_tag].iloc[0]
    reasons = []
    if ref_row.hidden_auc_ci_lo <= 0.55:
        reasons.append("reference latent viability is not robustly decodable")
    if late_row.output_wrong_n < 30:
        reasons.append("too few late output-wrong confirmation examples")
    if np.isfinite(late_row.hidden_rescue_ci_lo) and late_row.hidden_rescue_ci_lo <= 0.50:
        reasons.append("hidden probe does not rescue late output-wrong cases above chance")
    if args.control_tag:
        ctrl_row = df[df.tag == args.control_tag].iloc[0]
        if ctrl_row.hidden_auc >= 0.60:
            reasons.append("target-blind control remains decodable; likely shortcut/leakage")
    decision = {
        "status": "continue" if not reasons else "stop_or_redesign",
        "reference_tag": args.reference_tag,
        "late_tag": args.late_tag,
        "selected_layer": int(selected_layer),
        "reasons": reasons,
        "note": "Behavior preflight must separately establish pass@k shrinkage and first-fork polarization before this latent gate is interpreted mechanistically.",
    }
    out.with_suffix(".json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
