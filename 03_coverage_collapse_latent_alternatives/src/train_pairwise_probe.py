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
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--primary-layer-frac", type=float, default=0.5, help="Preregistered primary block depth; all layers are still reported diagnostically.")
    return p.parse_args()


def probe_auc(X, y, folds=5):
    counts = np.bincount(y, minlength=2)
    n_splits = min(folds, int(counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, accs = [], []
    for tr, te in cv.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(64, X.shape[1])),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], p >= 0.5))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs))


def orient_margin_auc(margin, y):
    # label=1 means alphabetically first candidate A is viable, so positive margin should favor y=1.
    auc = roc_auc_score(y, margin)
    acc = accuracy_score(y, margin >= 0)
    return float(auc), float(acc)


def main():
    args = parse_args()
    files = sorted(Path(args.input_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No state files in {args.input_dir}")
    rows = []
    for path in files:
        d = np.load(path, allow_pickle=False)
        y = d["label_a_viable"].astype(int)
        hidden = d["hidden"].astype(np.float32)
        de = d["candidate_embedding_diff"].astype(np.float32)
        margin = d["output_logprob_margin_a_minus_b"].astype(np.float32)
        tag = str(d["tag"].item())
        baseline_auc, baseline_acc = orient_margin_auc(margin, y)
        primary = int(round((hidden.shape[1]-1) * args.primary_layer_frac))

        for li in range(hidden.shape[1]):
            # Diagonal bilinear candidate-conditioned feature. This is deliberately low capacity:
            # z = h ⊙ (e_A - e_B). A linear probe on z asks whether the state contains information
            # that aligns with which concrete candidate branch is viable.
            X = hidden[:,li,:] * de
            auc, auc_std, acc = probe_auc(X, y, args.folds)
            rows.append({
                "tag":tag,
                "layer":li,
                "primary_layer":int(li==primary),
                "n":len(y),
                "positive_a_viable":int(y.sum()),
                "latent_auc":auc,
                "latent_auc_std":auc_std,
                "latent_accuracy":acc,
                "output_margin_auc":baseline_auc,
                "output_margin_accuracy":baseline_acc,
                "latent_minus_output_auc":auc-baseline_auc,
            })

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df[df["primary_layer"]==1].to_string(index=False))
    print(f"\nSaved all-layer diagnostics to {out}")


if __name__ == "__main__":
    main()
