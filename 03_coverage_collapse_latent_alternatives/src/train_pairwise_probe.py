from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate branch-specific latent viability across one or more checkpoints.")
    p.add_argument("--input-dir", default="artifacts/states")
    p.add_argument("--output", default="artifacts/branch_probe_metrics.csv")
    p.add_argument("--predictions-output", default=None, help="OOF per-problem predictions; defaults next to --output.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--primary-layer-frac", type=float, default=0.5, help="Preregistered primary block depth; all layers are still reported diagnostically.")
    return p.parse_args()


def make_probe(dim: int, n_train: int):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=min(64, dim, max(1, n_train - 1))),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def probe_oof(X, y, folds=5):
    counts = np.bincount(y, minlength=2)
    n_splits = min(folds, int(counts.min()))
    if n_splits < 2:
        raise ValueError(f"Need at least 2 samples in each class, got {counts.tolist()}")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        clf = make_probe(X.shape[1], len(tr))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        fold_ids[te] = fold
    if np.isnan(oof).any() or np.any(fold_ids < 0):
        raise RuntimeError("OOF predictions were not filled for every example")
    auc = roc_auc_score(y, oof)
    acc = accuracy_score(y, oof >= 0.5)
    fold_aucs = [roc_auc_score(y[fold_ids == f], oof[fold_ids == f]) for f in np.unique(fold_ids)]
    return float(auc), float(np.std(fold_aucs)), float(acc), oof


def orient_margin_auc(margin, y):
    auc = roc_auc_score(y, margin)
    acc = accuracy_score(y, margin >= 0)
    return float(auc), float(acc)


def main():
    args = parse_args()
    files = sorted(Path(args.input_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No state files in {args.input_dir}")
    rows = []
    pred_rows = []
    for path in files:
        d = np.load(path, allow_pickle=False)
        y = d["label_a_viable"].astype(int)
        problem_ids = d["problem_id"].astype(int)
        hidden = d["hidden"].astype(np.float32)
        de = d["candidate_embedding_diff"].astype(np.float32)
        margin = d["output_logprob_margin_a_minus_b"].astype(np.float32)
        tag = str(d["tag"].item())
        baseline_auc, baseline_acc = orient_margin_auc(margin, y)
        primary = int(round((hidden.shape[1] - 1) * args.primary_layer_frac))

        for li in range(hidden.shape[1]):
            X = hidden[:, li, :] * de
            auc, auc_std, acc, oof = probe_oof(X, y, args.folds)
            rows.append({
                "tag": tag,
                "layer": li,
                "primary_layer": int(li == primary),
                "n": len(y),
                "positive_a_viable": int(y.sum()),
                "latent_auc": auc,
                "latent_auc_fold_std": auc_std,
                "latent_accuracy": acc,
                "output_margin_auc": baseline_auc,
                "output_margin_accuracy": baseline_acc,
                "latent_minus_output_auc": auc - baseline_auc,
            })
            for pid, yi, p, m in zip(problem_ids, y, oof, margin):
                pred_rows.append({
                    "tag": tag,
                    "problem_id": int(pid),
                    "layer": li,
                    "primary_layer": int(li == primary),
                    "a_is_viable": int(yi),
                    "latent_oof_p_a_viable": float(p),
                    "latent_oof_p_true_viable": float(p if yi else 1.0 - p),
                    "output_margin_a_minus_b": float(m),
                    "output_margin_true_viable": float(m if yi else -m),
                })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    pred_out = Path(args.predictions_output) if args.predictions_output else out.with_name(out.stem + "_oof.csv")
    pd.DataFrame(pred_rows).to_csv(pred_out, index=False)
    print(df[df["primary_layer"] == 1].to_string(index=False))
    print(f"\nSaved all-layer diagnostics to {out}")
    print(f"Saved per-problem OOF scores to {pred_out}")


if __name__ == "__main__":
    main()
