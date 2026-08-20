from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

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


TASKS = (
    "final_correct_replication",
    "recover_any",
    "overwrite_any",
    "transient_recovery",
    "transient_overwrite",
    "finish_correct_from_wrong",
    "finish_wrong_from_correct",
)
NOVEL_TASKS = {"transient_recovery", "transient_overwrite"}
TRANSITION_TASKS = {
    "recover_any",
    "overwrite_any",
    "transient_recovery",
    "transient_overwrite",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train conditional trajectory-transition probes with novelty controls."
    )
    p.add_argument("--input-dir", default="artifacts/g0/raw")
    p.add_argument("--output-dir", default="artifacts/g0/probes")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-class-count", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument(
        "--label-mode",
        choices=["strict", "fallback"],
        default="strict",
        help=(
            "strict requires an explicit requested answer marker; fallback uses the "
            "public probing parser's last-number fallback."
        ),
    )
    p.add_argument(
        "--lead-thresholds",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Minimum positive lead times for same-step pre-transition analyses.",
    )
    return p.parse_args()


def hidden_pipeline(dim: int, n_train: int):
    n_components = min(64, dim, max(1, n_train - 1))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def surface_pipeline():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def _splits(y: np.ndarray, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    n_splits = min(folds, int(counts.min()))
    if n_splits < 2:
        return []
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )
    return list(cv.split(np.zeros(len(y)), y))


def _oof_hidden(
    X: np.ndarray,
    y: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=np.float64)
    for tr, te in splits:
        clf = hidden_pipeline(X.shape[1], len(tr))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def _oof_surface(
    X: np.ndarray,
    y: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=np.float64)
    for tr, te in splits:
        clf = surface_pipeline()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    valid = np.isfinite(p)
    if valid.sum() < 2 or np.unique(y[valid]).size < 2:
        return float("nan")
    return float(roc_auc_score(y[valid], p[valid]))


def _bootstrap_metrics(
    y: np.ndarray,
    current: np.ndarray,
    surface: np.ndarray,
    initial: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    point = {
        "auc": _safe_auc(y, current),
        "surface_auc": _safe_auc(y, surface),
        "initial_auc": _safe_auc(y, initial),
    }
    point["delta_vs_surface"] = point["auc"] - point["surface_auc"]
    point["delta_vs_initial"] = point["auc"] - point["initial_auc"]

    if n_bootstrap <= 0:
        for key in list(point):
            point[f"{key}_lo"] = float("nan")
            point[f"{key}_hi"] = float("nan")
        return point

    rng = np.random.default_rng(seed)
    samples = {k: [] for k in point}
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).size < 2:
            continue
        a = _safe_auc(y[idx], current[idx])
        s = _safe_auc(y[idx], surface[idx])
        i = _safe_auc(y[idx], initial[idx])
        if not (np.isfinite(a) and np.isfinite(s) and np.isfinite(i)):
            continue
        samples["auc"].append(a)
        samples["surface_auc"].append(s)
        samples["initial_auc"].append(i)
        samples["delta_vs_surface"].append(a - s)
        samples["delta_vs_initial"].append(a - i)

    for key, vals in samples.items():
        if len(vals) < 20:
            point[f"{key}_lo"] = float("nan")
            point[f"{key}_hi"] = float("nan")
        else:
            point[f"{key}_lo"] = float(np.percentile(vals, 2.5))
            point[f"{key}_hi"] = float(np.percentile(vals, 97.5))
    return point


def _surface_features(
    data: dict,
    labels: dict[str, np.ndarray],
    si: int,
    idx: np.ndarray,
) -> np.ndarray:
    current_observed = labels["current_observed"][idx, si].astype(np.float32)
    current_correct = labels["current_correct"][idx, si].astype(np.float32)
    current_correct = np.where(current_correct < 0, 0.0, current_correct)

    return np.column_stack(
        [
            np.asarray(data["entropy"])[idx, si],
            np.asarray(data["selected_prob"])[idx, si],
            np.asarray(data["clean_maxprob"])[idx, si],
            np.asarray(data["frac_unmasked"])[idx, si],
            np.asarray(data["prompt_tokens"])[idx],
            current_observed,
            current_correct,
        ]
    ).astype(np.float32)


def _task_target(
    labels: dict[str, np.ndarray],
    task: str,
    si: int,
) -> tuple[np.ndarray, np.ndarray]:
    if task == "final_correct_replication":
        y = labels["final_correct"][:, si]
        valid = labels["final_observed"][:, si] == 1
    elif task == "recover_any":
        y = labels["recoverable"][:, si]
        valid = y >= 0
    elif task == "overwrite_any":
        y = labels["will_overwrite"][:, si]
        valid = y >= 0
    elif task == "transient_recovery":
        y = labels["transient_recovery"][:, si]
        valid = y >= 0
    elif task == "transient_overwrite":
        y = labels["transient_overwrite"][:, si]
        valid = y >= 0
    elif task == "finish_correct_from_wrong":
        y = labels["finish_correct_from_wrong"][:, si]
        valid = y >= 0
    elif task == "finish_wrong_from_correct":
        y = labels["finish_wrong_from_correct"][:, si]
        valid = y >= 0
    else:
        raise ValueError(f"unknown task: {task}")
    return y.astype(np.int8), valid


def _lead_for_task(
    labels: dict[str, np.ndarray],
    task: str,
) -> np.ndarray:
    if task in {"recover_any", "transient_recovery"}:
        return labels["recovery_lead"]
    if task in {"overwrite_any", "transient_overwrite"}:
        return labels["overwrite_lead"]
    raise ValueError(f"task has no transition lead: {task}")


def _probe_row(
    data: dict,
    labels: dict[str, np.ndarray],
    *,
    task: str,
    si: int,
    layer_i: int,
    layer_id: int,
    idx: np.ndarray,
    y: np.ndarray,
    folds: int,
    n_bootstrap: int,
    suffix: dict | None = None,
) -> dict:
    capture_steps = np.asarray(data["capture_steps"])
    zero_candidates = np.flatnonzero(capture_steps == 0)
    if zero_candidates.size != 1:
        raise ValueError("capture_steps must include step 0 exactly once")
    zero_i = int(zero_candidates[0])

    hidden = np.asarray(data["hidden"])
    current_X = hidden[idx, si, layer_i].astype(np.float32).mean(axis=1)
    initial_X = hidden[idx, zero_i, layer_i].astype(np.float32).mean(axis=1)
    surface_X = _surface_features(data, labels, si, idx)

    splits = _splits(y, folds)
    if not splits:
        raise ValueError("not enough samples for CV")

    current_pred = _oof_hidden(current_X, y, splits)
    initial_pred = _oof_hidden(initial_X, y, splits)
    surface_pred = _oof_surface(surface_X, y, splits)

    metrics = _bootstrap_metrics(
        y,
        current_pred,
        surface_pred,
        initial_pred,
        n_bootstrap=n_bootstrap,
        seed=1009 + int(capture_steps[si]) * 17 + layer_i,
    )

    row = {
        "task": task,
        "step": int(capture_steps[si]),
        "layer": int(layer_id),
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        **metrics,
    }
    if suffix:
        row.update(suffix)
    return row


def fixed_step_results(
    data: dict,
    labels: dict[str, np.ndarray],
    *,
    min_class_count: int,
    folds: int,
    n_bootstrap: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    counts: list[dict] = []
    hidden_indices = np.asarray(data["hidden_indices"])

    for si, step in enumerate(np.asarray(data["capture_steps"]).tolist()):
        for task in TASKS:
            target, valid = _task_target(labels, task, si)
            idx = np.flatnonzero(valid)
            y = target[idx].astype(int)
            class_counts = np.bincount(y, minlength=2) if len(y) else np.zeros(2, dtype=int)
            counts.append(
                {
                    "task": task,
                    "step": int(step),
                    "n": int(len(y)),
                    "positive": int(class_counts[1]),
                    "negative": int(class_counts[0]),
                    "min_class": int(class_counts.min()) if len(y) else 0,
                }
            )
            if len(y) == 0 or class_counts.min() < min_class_count:
                continue

            for li, layer_id in enumerate(hidden_indices.tolist()):
                rows.append(
                    _probe_row(
                        data,
                        labels,
                        task=task,
                        si=si,
                        layer_i=li,
                        layer_id=layer_id,
                        idx=idx,
                        y=y,
                        folds=folds,
                        n_bootstrap=n_bootstrap,
                    )
                )
    return rows, counts


def pretransition_results(
    data: dict,
    labels: dict[str, np.ndarray],
    *,
    thresholds: list[int],
    min_class_count: int,
    folds: int,
    n_bootstrap: int,
) -> list[dict]:
    rows: list[dict] = []
    hidden_indices = np.asarray(data["hidden_indices"])

    for si, _step in enumerate(np.asarray(data["capture_steps"]).tolist()):
        for task in sorted(TRANSITION_TASKS):
            target, valid = _task_target(labels, task, si)
            lead = _lead_for_task(labels, task)[:, si]

            for min_lead in thresholds:
                # Same absolute denoising step for positives and negatives.
                # Near-transition positives are removed; negatives are "never flips".
                keep = valid & (
                    (target == 0)
                    | ((target == 1) & (lead >= min_lead))
                )
                idx = np.flatnonzero(keep)
                y = target[idx].astype(int)
                if len(y) == 0:
                    continue
                class_counts = np.bincount(y, minlength=2)
                if class_counts.min() < min_class_count:
                    continue

                for li, layer_id in enumerate(hidden_indices.tolist()):
                    rows.append(
                        _probe_row(
                            data,
                            labels,
                            task=task,
                            si=si,
                            layer_i=li,
                            layer_id=layer_id,
                            idx=idx,
                            y=y,
                            folds=folds,
                            n_bootstrap=n_bootstrap,
                            suffix={"min_lead": int(min_lead)},
                        )
                    )
    return rows


def _decision(
    step_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    count_df: pd.DataFrame,
    min_class_count: int,
) -> dict:
    support = count_df[
        count_df["task"].isin(sorted(NOVEL_TASKS))
    ]
    max_support = int(support["min_class"].max()) if len(support) else 0

    if "task" in step_df.columns:
        ref = step_df[step_df["task"] == "final_correct_replication"]
    else:
        ref = pd.DataFrame()
    # A valid geometry should reproduce the *emergent* final-correctness signal,
    # not merely classify static problem difficulty from step 0.
    later_ref = ref[ref["step"] > 0] if len(ref) else ref
    reference_signal_ok = bool(
        len(later_ref)
        and np.any(
            (later_ref["auc"].to_numpy() >= 0.65)
            & (later_ref["delta_vs_initial"].to_numpy() >= 0.03)
        )
    )

    pre = pre_df[
        pre_df["task"].isin(sorted(NOVEL_TASKS))
        & (pre_df["min_lead"] >= 4)
    ] if len(pre_df) else pre_df

    strong = pd.DataFrame()
    if len(pre):
        strong = pre[
            (pre["auc"] >= 0.65)
            & (pre["delta_vs_surface"] >= 0.03)
            & (pre["delta_vs_initial"] >= 0.03)
            & (pre["auc_lo"] > 0.55)
        ]

    if max_support < min_class_count:
        status = "STOP_LOW_NOVEL_CLASS_SUPPORT"
    elif not reference_signal_ok:
        status = "GEOMETRY_NOT_VALIDATED_RUN_REFERENCE_GEOMETRY"
    elif len(strong):
        status = "CONTINUE"
    else:
        status = "STOP_NO_NOVEL_PRETRANSITION_SIGNAL"

    return {
        "status": status,
        "max_novel_min_class_support": max_support,
        "reference_final_correctness_probe_replicated": reference_signal_ok,
        "strong_pretransition_rows": int(len(strong)),
        "gate": {
            "min_class_count": int(min_class_count),
            "auc": 0.65,
            "delta_vs_surface": 0.03,
            "delta_vs_initial": 0.03,
            "auc_lower_ci": 0.55,
            "min_lead": 4,
        },
    }


def main() -> None:
    args = parse_args()
    data = load_shards(Path(args.input_dir), require_hidden=True)

    if args.label_mode == "strict":
        correct = np.asarray(data["correct_strict"])
        observed = np.asarray(data["observed_strict"])
    else:
        correct = np.asarray(data["correct_fallback"])
        observed = np.asarray(data["observed_fallback"])

    labels = fate_from_correctness(
        correct,
        np.asarray(data["capture_steps"]),
        observed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "fate_labels.npz",
        problem_id=np.asarray(data["problem_id"]),
        capture_steps=np.asarray(data["capture_steps"]),
        **labels,
    )

    rows, counts = fixed_step_results(
        data,
        labels,
        min_class_count=args.min_class_count,
        folds=args.folds,
        n_bootstrap=args.bootstrap,
    )
    step_df = pd.DataFrame(rows)
    count_df = pd.DataFrame(counts)
    step_df.to_csv(out_dir / "step_layer_auc.csv", index=False)
    count_df.to_csv(out_dir / "task_class_counts.csv", index=False)

    pre_rows = pretransition_results(
        data,
        labels,
        thresholds=sorted(set(args.lead_thresholds)),
        min_class_count=args.min_class_count,
        folds=args.folds,
        n_bootstrap=args.bootstrap,
    )
    pre_df = pd.DataFrame(pre_rows)
    pre_df.to_csv(out_dir / "pretransition_auc.csv", index=False)

    decision = _decision(
        step_df,
        pre_df,
        count_df,
        min_class_count=args.min_class_count,
    )
    decision["label_mode"] = args.label_mode
    decision["metadata"] = data["metadata"]
    (out_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(decision, indent=2, sort_keys=True))
    if len(step_df):
        print(
            step_df.sort_values(
                ["task", "step", "auc"],
                ascending=[True, True, False],
            ).to_string(index=False)
        )


if __name__ == "__main__":
    main()
