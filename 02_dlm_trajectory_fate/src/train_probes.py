from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from fate_labels import fate_from_correctness


def parse_args():
    p = argparse.ArgumentParser(description="Train conditional trajectory-fate probes.")
    p.add_argument("--input-dir", default="artifacts/raw")
    p.add_argument("--output-dir", default="artifacts/probes")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--label-mode", choices=["strict", "fallback"], default="strict", help="strict requires the requested answer marker at intermediate steps; fallback matches the public probe parser fallback.")
    return p.parse_args()


def load_shards(input_dir: Path) -> dict[str, np.ndarray]:
    files = sorted(input_dir.glob("shard_??_of_??.npz"))
    if not files:
        raise FileNotFoundError(f"No shard NPZ files in {input_dir}")
    blobs = [np.load(f, allow_pickle=False) for f in files]
    for key in ["capture_steps", "hidden_indices"]:
        for b in blobs[1:]:
            if not np.array_equal(blobs[0][key], b[key]):
                raise ValueError(f"Shard mismatch for {key}")
    cat_keys = ["problem_id", "gold", "correct_all", "correct_all_fallback", "answer_all", "hidden", "entropy", "maxprob", "frac_unmasked", "final_correct"]
    out = {k: np.concatenate([b[k] for b in blobs], axis=0) for k in cat_keys}
    out["capture_steps"] = blobs[0]["capture_steps"]
    out["hidden_indices"] = blobs[0]["hidden_indices"]
    return out


def pipeline(dim: int, n_train: int):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=min(64, dim, max(1, n_train - 1))),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def cv_auc(X, y, folds=5) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    n_splits = min(folds, int(counts.min()))
    if n_splits < 2:
        return float("nan"), float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in cv.split(X, y):
        clf = pipeline(X.shape[1], len(tr))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def fixed_step_results(data, labels, task: str, min_class_count: int, folds: int):
    target = labels["recoverable"] if task == "recover" else labels["will_overwrite"]
    current_should_be = 0 if task == "recover" else 1
    rows = []
    for si, step in enumerate(data["capture_steps"].tolist()):
        mask = labels["current_correct"][:, si] == current_should_be
        y = target[mask, si]
        valid = y >= 0
        mask_idx = np.flatnonzero(mask)[valid]
        y = y[valid].astype(int)
        counts = np.bincount(y, minlength=2)
        if counts.min() < min_class_count:
            rows.append({"task": task, "step": step, "layer": "SKIP", "n": len(y), "positive": int(y.sum()), "auc": np.nan, "auc_std": np.nan, "baseline_auc": np.nan})
            continue

        baseline = np.column_stack([
            data["entropy"][mask_idx, si],
            data["maxprob"][mask_idx, si],
            data["frac_unmasked"][mask_idx, si],
        ]).astype(np.float32)
        b_auc, b_std = cv_auc(baseline, y, folds)

        for li, hs_idx in enumerate(data["hidden_indices"].tolist()):
            X = data["hidden"][mask_idx, si, li].astype(np.float32).mean(axis=1)
            auc, auc_std = cv_auc(X, y, folds)
            rows.append({
                "task": task,
                "step": step,
                "layer": int(hs_idx),
                "n": len(y),
                "positive": int(y.sum()),
                "auc": auc,
                "auc_std": auc_std,
                "baseline_auc": b_auc,
                "baseline_auc_std": b_std,
                "delta_vs_baseline": auc - b_auc,
            })
    return rows


def lead_time_results(data, labels, task: str, folds: int, min_class_count: int):
    target = labels["recoverable"] if task == "recover" else labels["will_overwrite"]
    lead = labels["recovery_lead"] if task == "recover" else labels["overwrite_lead"]
    current_should_be = 0 if task == "recover" else 1
    bins = [(1,4), (5,8), (9,16), (17,32), (33,127)]
    rows = []

    # Pool saved steps, but enforce problem-ID grouping so states from the same problem never cross folds.
    for lo, hi in bins:
        examples = []
        for si, step in enumerate(data["capture_steps"].tolist()):
            current_mask = labels["current_correct"][:, si] == current_should_be
            pos = current_mask & (target[:, si] == 1) & (lead[:, si] >= lo) & (lead[:, si] <= hi)
            neg = current_mask & (target[:, si] == 0)
            keep = pos | neg
            for pi in np.flatnonzero(keep):
                examples.append((pi, si, int(pos[pi])))
        if not examples:
            continue
        y = np.array([e[2] for e in examples], dtype=int)
        if np.bincount(y, minlength=2).min() < min_class_count:
            continue
        groups = np.array([data["problem_id"][e[0]] for e in examples])
        baseline = np.array([[data["entropy"][e[0],e[1]], data["maxprob"][e[0],e[1]], data["frac_unmasked"][e[0],e[1]], data["capture_steps"][e[1]]] for e in examples], dtype=np.float32)
        for li, hs_idx in enumerate(data["hidden_indices"].tolist()):
            X = np.stack([data["hidden"][e[0],e[1],li].astype(np.float32).mean(axis=0) for e in examples])
            n_splits = min(folds, int(np.bincount(y, minlength=2).min()))
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            h_aucs, b_aucs = [], []
            for tr, te in cv.split(X, y, groups=groups):
                hc = pipeline(X.shape[1], len(tr)); hc.fit(X[tr], y[tr])
                bc = pipeline(baseline.shape[1], len(tr)); bc.fit(baseline[tr], y[tr])
                h_aucs.append(roc_auc_score(y[te], hc.predict_proba(X[te])[:,1]))
                b_aucs.append(roc_auc_score(y[te], bc.predict_proba(baseline[te])[:,1]))
            rows.append({"task":task,"lead_lo":lo,"lead_hi":hi,"layer":int(hs_idx),"n":len(y),"positive":int(y.sum()),"auc":float(np.mean(h_aucs)),"baseline_auc":float(np.mean(b_aucs)),"delta_vs_baseline":float(np.mean(h_aucs)-np.mean(b_aucs))})
    return rows


def main():
    args = parse_args()
    data = load_shards(Path(args.input_dir))
    correctness_key = "correct_all" if args.label_mode == "strict" else "correct_all_fallback"
    labels = fate_from_correctness(data[correctness_key], data["capture_steps"])
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "fate_labels.npz", problem_id=data["problem_id"], capture_steps=data["capture_steps"], **labels)

    rows = []
    for task in ["recover", "overwrite"]:
        rows.extend(fixed_step_results(data, labels, task, args.min_class_count, args.folds))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "step_layer_auc.csv", index=False)

    lead_rows = []
    for task in ["recover", "overwrite"]:
        lead_rows.extend(lead_time_results(data, labels, task, args.folds, args.min_class_count))
    pd.DataFrame(lead_rows).to_csv(out_dir / "lead_time_auc.csv", index=False)

    counts = []
    for si, step in enumerate(data["capture_steps"].tolist()):
        cc = labels["current_correct"][:,si]
        r = labels["recoverable"][:,si]
        o = labels["will_overwrite"][:,si]
        counts.append({"step":step,"wrong":int((cc==0).sum()),"wrong_recoverable":int((r==1).sum()),"wrong_doomed":int((r==0).sum()),"correct":int((cc==1).sum()),"correct_overwrite":int((o==1).sum()),"correct_stable":int((o==0).sum())})
    pd.DataFrame(counts).to_csv(out_dir / "class_counts.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
