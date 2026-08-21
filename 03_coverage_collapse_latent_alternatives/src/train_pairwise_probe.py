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
    p = argparse.ArgumentParser(
        description="Counterfactual latent gate with a frozen reference probe and matched target flips."
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--reference-tag", required=True)
    p.add_argument("--late-tag", default="e16")
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--discovery-frac", type=float, default=0.6)
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--commit-margin", type=float, default=2.0)
    p.add_argument("--min-disagreement-events", type=int, default=30)
    return p.parse_args()


def load_state(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {
        "tag": str(d["tag"].item()),
        "condition": str(d["condition"].item()),
        "pid": d["problem_id"].astype(int),
        "y": d["label_a_viable"].astype(int),
        "y_original": d["original_label_a_viable"].astype(int),
        "hidden": d["hidden"].astype(np.float32),
        "de": d["candidate_embedding_diff"].astype(np.float32),
        "margin": d["output_logprob_margin_a_minus_b"].astype(np.float32),
        "true_margin": d["output_true_viable_margin"].astype(np.float32),
    }


def align(reference: dict, other: dict, expect_flipped: bool = False) -> None:
    if not np.array_equal(reference["pid"], other["pid"]):
        raise ValueError(f"problem_id mismatch between {reference['tag']}:{reference['condition']} and {other['tag']}:{other['condition']}")
    if reference["hidden"].shape[1:] != other["hidden"].shape[1:]:
        raise ValueError("hidden-state shape mismatch")
    if not np.array_equal(reference["y_original"], other["y_original"]):
        raise ValueError("original viability labels do not align")
    expected = 1 - reference["y"] if expect_flipped else reference["y"]
    if not np.array_equal(expected, other["y"]):
        relation = "flipped" if expect_flipped else "same"
        raise ValueError(f"condition labels are not {relation} as expected")


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


def locked_features(d: dict, layer: int, reference_de: np.ndarray) -> np.ndarray:
    # Important: the candidate-alignment basis is frozen at the behavior-selected
    # reference checkpoint. We do NOT use late checkpoint embeddings here.
    return d["hidden"][:, layer, :] * reference_de


def paired_training_arrays(X_orig, X_flip, y_orig, ids):
    return (
        np.concatenate([X_orig[ids], X_flip[ids]], axis=0),
        np.concatenate([y_orig[ids], 1 - y_orig[ids]], axis=0),
    )


def paired_cv_auc(X_orig, X_flip, y_orig, ids, pca_dim, seed):
    y_problem = y_orig[ids]
    counts = np.bincount(y_problem, minlength=2)
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred_all, y_all = [], []
    for tr_local, te_local in cv.split(ids, y_problem):
        tr = ids[tr_local]
        te = ids[te_local]
        Xtr, ytr = paired_training_arrays(X_orig, X_flip, y_orig, tr)
        Xte, yte = paired_training_arrays(X_orig, X_flip, y_orig, te)
        clf = make_probe(Xtr.shape[1], len(Xtr), pca_dim)
        clf.fit(Xtr, ytr)
        pred_all.append(clf.predict_proba(Xte)[:, 1])
        y_all.append(yte)
    return safe_auc(np.concatenate(y_all), np.concatenate(pred_all))


def _bootstrap_problem(n, fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ids = rng.integers(0, n, n)
        v = fn(ids)
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def pair_metrics(y_orig, p_orig, p_flip, margin_orig, margin_flip, n_boot, seed):
    y_pair = np.concatenate([y_orig, 1 - y_orig])
    p_pair = np.concatenate([p_orig, p_flip])
    out_pair = np.concatenate([margin_orig, margin_flip])
    hidden_pair_auc = safe_auc(y_pair, p_pair)
    hidden_pair_acc = float(accuracy_score(y_pair, p_pair >= 0.5))
    output_pair_auc = safe_auc(y_pair, out_pair)
    output_pair_acc = float(accuracy_score(y_pair, out_pair > 0))
    signed_flip = (p_orig - p_flip) * (2 * y_orig - 1)
    flip_direction_acc = float((signed_flip > 0).mean())
    original_hidden_acc = float(accuracy_score(y_orig, p_orig >= 0.5))
    original_output_acc = float(accuracy_score(y_orig, margin_orig > 0))

    n = len(y_orig)
    auc_lo, auc_hi = _bootstrap_problem(
        n,
        lambda ids: safe_auc(
            np.concatenate([y_orig[ids], 1 - y_orig[ids]]),
            np.concatenate([p_orig[ids], p_flip[ids]]),
        ),
        n_boot,
        seed,
    )
    flip_lo, flip_hi = _bootstrap_problem(
        n,
        lambda ids: float((((p_orig[ids] - p_flip[ids]) * (2 * y_orig[ids] - 1)) > 0).mean()),
        n_boot,
        seed + 1,
    )
    return {
        "pair_hidden_auc": hidden_pair_auc,
        "pair_hidden_auc_ci_lo": auc_lo,
        "pair_hidden_auc_ci_hi": auc_hi,
        "pair_hidden_accuracy": hidden_pair_acc,
        "pair_output_auc": output_pair_auc,
        "pair_output_accuracy": output_pair_acc,
        "target_flip_direction_accuracy": flip_direction_acc,
        "target_flip_direction_ci_lo": flip_lo,
        "target_flip_direction_ci_hi": flip_hi,
        "original_hidden_accuracy": original_hidden_acc,
        "original_output_accuracy": original_output_acc,
    }


def disagreement_hidden_win(
    y_orig,
    p_orig,
    p_flip,
    margin_orig,
    margin_flip,
    commit_margin,
    n_boot,
    seed,
):
    y = np.stack([y_orig, 1 - y_orig], axis=1)
    p = np.stack([p_orig, p_flip], axis=1)
    m = np.stack([margin_orig, margin_flip], axis=1)
    hidden_pred = p >= 0.5
    output_pred = m > 0
    committed = np.abs(m) >= commit_margin
    disagree = committed & (hidden_pred != output_pred)
    hidden_wins = hidden_pred == y
    n_events = int(disagree.sum())
    n_problems = int(disagree.any(axis=1).sum())
    if n_events == 0:
        return {
            "committed_disagreement_events": 0,
            "committed_disagreement_problems": 0,
            "hidden_win_rate_on_committed_disagreement": float("nan"),
            "hidden_win_ci_lo": float("nan"),
            "hidden_win_ci_hi": float("nan"),
        }
    point = float(hidden_wins[disagree].mean())
    n = len(y_orig)

    def stat(ids):
        mask = disagree[ids]
        if not mask.any():
            return float("nan")
        return float(hidden_wins[ids][mask].mean())

    lo, hi = _bootstrap_problem(n, stat, n_boot, seed)
    return {
        "committed_disagreement_events": n_events,
        "committed_disagreement_problems": n_problems,
        "hidden_win_rate_on_committed_disagreement": point,
        "hidden_win_ci_lo": lo,
        "hidden_win_ci_hi": hi,
    }


def main():
    args = parse_args()
    root = Path(args.input_dir)

    ref_orig = load_state(root / f"{args.reference_tag}_original.npz")
    ref_flip = load_state(root / f"{args.reference_tag}_target_flip.npz")
    late_orig = load_state(root / f"{args.late_tag}_original.npz")
    late_flip = load_state(root / f"{args.late_tag}_target_flip.npz")
    ref_blind = load_state(root / f"{args.reference_tag}_target_blind.npz")

    align(ref_orig, ref_flip, expect_flipped=True)
    align(ref_orig, late_orig, expect_flipped=False)
    align(ref_orig, late_flip, expect_flipped=True)
    align(ref_orig, ref_blind, expect_flipped=False)

    y = ref_orig["y"]
    if len(y) < 200:
        raise ValueError(f"Need at least 200 non-behavior problems for latent confirmation; got {len(y)}")

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=args.discovery_frac, random_state=args.seed)
    discovery, confirm = next(splitter.split(np.zeros(len(y)), y))

    # Freeze the candidate basis at the reference checkpoint. This prevents the late
    # probe measurement from inheriting a changed tied embedding/output head.
    ref_de = ref_orig["de"]
    layer_rows = []
    for layer in range(ref_orig["hidden"].shape[1]):
        Xo = locked_features(ref_orig, layer, ref_de)
        Xf = locked_features(ref_flip, layer, ref_de)
        auc = paired_cv_auc(Xo, Xf, y, discovery, args.pca_dim, args.seed)
        layer_rows.append((layer, auc))
    selected_layer, discovery_auc = max(
        layer_rows, key=lambda x: (-np.inf if np.isnan(x[1]) else x[1])
    )

    X_ref_o = locked_features(ref_orig, selected_layer, ref_de)
    X_ref_f = locked_features(ref_flip, selected_layer, ref_de)
    Xtrain, ytrain = paired_training_arrays(X_ref_o, X_ref_f, y, discovery)
    clf = make_probe(Xtrain.shape[1], len(Xtrain), args.pca_dim)
    clf.fit(Xtrain, ytrain)

    rows = []
    predictions = []
    eval_cache = {}
    for tag, orig, flip in (
        (args.reference_tag, ref_orig, ref_flip),
        (args.late_tag, late_orig, late_flip),
    ):
        Xo = locked_features(orig, selected_layer, ref_de)
        Xf = locked_features(flip, selected_layer, ref_de)
        po = clf.predict_proba(Xo[confirm])[:, 1]
        pf = clf.predict_proba(Xf[confirm])[:, 1]
        yc = y[confirm]
        mo = orig["margin"][confirm]
        mf = flip["margin"][confirm]

        metrics = pair_metrics(yc, po, pf, mo, mf, args.bootstrap, args.seed + (0 if tag == args.reference_tag else 10))
        metrics.update({
            "tag": tag,
            "n_discovery_problems": int(len(discovery)),
            "n_confirm_problems": int(len(confirm)),
            "selected_layer": int(selected_layer),
            "reference_discovery_cv_auc": float(discovery_auc),
            "probe_basis": "reference_checkpoint_candidate_embeddings",
            "probe_training": "reference_checkpoint_original+matched_target_flip_discovery_only",
        })
        rows.append(metrics)
        eval_cache[tag] = (yc, po, pf, mo, mf)

        for condition, probs, margins, labels in (
            ("original", po, mo, yc),
            ("target_flip", pf, mf, 1 - yc),
        ):
            for pid, yi, pi, mi in zip(orig["pid"][confirm], labels, probs, margins):
                predictions.append({
                    "tag": tag,
                    "condition": condition,
                    "problem_id": int(pid),
                    "a_is_viable": int(yi),
                    "latent_p_a_viable": float(pi),
                    "native_margin_a_minus_b": float(mi),
                    "latent_choice_a": int(pi >= 0.5),
                    "native_choice_a": int(mi > 0),
                })

    # Target-blind negative control uses the SAME frozen reference probe. No control-specific
    # fitting is allowed, because fitting a new probe could itself learn generator shortcuts.
    Xb = locked_features(ref_blind, selected_layer, ref_de)
    pb = clf.predict_proba(Xb[confirm])[:, 1]
    yb = y[confirm]
    blind_auc = safe_auc(yb, pb)
    blind_lo, blind_hi = _bootstrap_problem(
        len(yb), lambda ids: safe_auc(yb[ids], pb[ids]), args.bootstrap, args.seed + 20
    )

    late_y, late_po, late_pf, late_mo, late_mf = eval_cache[args.late_tag]
    disagreement = disagreement_hidden_win(
        late_y,
        late_po,
        late_pf,
        late_mo,
        late_mf,
        args.commit_margin,
        args.bootstrap,
        args.seed + 30,
    )

    df = pd.DataFrame(rows)
    ref_row = df[df.tag == args.reference_tag].iloc[0]
    late_row = df[df.tag == args.late_tag].iloc[0]

    reasons = []
    if ref_row.pair_hidden_auc_ci_lo <= 0.70:
        reasons.append("reference paired counterfactual viability is not robustly decodable (AUC lower CI <= 0.70)")
    if ref_row.target_flip_direction_ci_lo <= 0.75:
        reasons.append("reference hidden score does not reliably flip when only the query target flips")
    if late_row.pair_hidden_auc_ci_lo <= 0.60:
        reasons.append("frozen reference probe does not transfer robustly to the late checkpoint")
    if late_row.target_flip_direction_ci_lo <= 0.60:
        reasons.append("late hidden score is not reliably target-sensitive under matched target flips")
    if abs(blind_auc - 0.5) >= 0.10:
        reasons.append(f"target-blind frozen-probe AUC={blind_auc:.3f} is too far from chance; shortcut/leakage risk")
    if disagreement["committed_disagreement_events"] < args.min_disagreement_events:
        reasons.append(
            f"only {disagreement['committed_disagreement_events']} label-free committed output/probe disagreement events; "
            f"need >= {args.min_disagreement_events} to distinguish access suppression"
        )
    elif disagreement["hidden_win_ci_lo"] <= 0.55:
        reasons.append(
            "on label-free high-confidence output/probe disagreements, hidden signal does not beat native output "
            "with lower 95% CI > 0.55"
        )

    decision = {
        "status": "continue_full_confirmation" if not reasons else "stop_or_redesign",
        "reference_tag": args.reference_tag,
        "late_tag": args.late_tag,
        "selected_layer": int(selected_layer),
        "reference_discovery_cv_auc": float(discovery_auc),
        "target_blind_frozen_probe_auc": float(blind_auc),
        "target_blind_frozen_probe_auc_ci95": [blind_lo, blind_hi],
        "commit_margin_abs_logprob": args.commit_margin,
        **disagreement,
        "thresholds": {
            "reference_pair_auc_ci_lo_gt": 0.70,
            "reference_flip_direction_ci_lo_gt": 0.75,
            "late_transfer_pair_auc_ci_lo_gt": 0.60,
            "late_flip_direction_ci_lo_gt": 0.60,
            "target_blind_abs_auc_minus_half_lt": 0.10,
            "min_committed_disagreement_events": args.min_disagreement_events,
            "hidden_win_ci_lo_on_disagreements_gt": 0.55,
        },
        "reasons": reasons,
        "important_note": (
            "The old 'hidden rescue on output-wrong examples' is intentionally NOT a gate: in a binary fork, "
            "conditioning on native output being wrong makes the opposite branch deterministically correct. "
            "The replacement subset is selected without labels: high-confidence native/probe disagreements."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.DataFrame(predictions).to_csv(args.predictions_output, index=False)
    out.with_suffix(".json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
