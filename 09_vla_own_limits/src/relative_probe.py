"""Shared linear probe and paired same-state relative-success analysis."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class SharedLinearProbe:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SharedLinearProbe":
        x = np.asarray(x, float)
        y = np.asarray(y, int)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be [N,D] and aligned with labels")
        if len(np.unique(y)) < 2:
            raise ValueError("training labels contain only one class")
        self.model.fit(self.scaler.fit_transform(x), y)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Return the shared decoder logit; no per-checkpoint recalibration."""
        z = self.scaler.transform(np.asarray(x, float))
        return self.model.decision_function(z)


def fit_shared_probe(
    features: np.ndarray,
    success: np.ndarray,
    state_ids: np.ndarray,
    train_state_ids: np.ndarray,
) -> SharedLinearProbe:
    features = np.asarray(features, float)
    success = np.asarray(success, int)
    state_ids = np.asarray(state_ids)
    train_state_ids = set(np.asarray(train_state_ids).tolist())
    if len(features) != len(success) or len(success) != len(state_ids):
        raise ValueError("features, success and state_ids must align")
    mask = np.array([sid in train_state_ids for sid in state_ids], dtype=bool)
    if not mask.any():
        raise ValueError("no rows belong to train_state_ids")
    return SharedLinearProbe().fit(features[mask], success[mask])


def paired_relative_metrics(
    state_ids: np.ndarray,
    checkpoints: np.ndarray,
    success: np.ndarray,
    scores: np.ndarray,
    checkpoint_a: str,
    checkpoint_b: str,
) -> dict:
    """Evaluate whether q_A-q_B predicts which checkpoint wins on crossover states."""
    state_ids = np.asarray(state_ids)
    checkpoints = np.asarray(checkpoints).astype(str)
    success = np.asarray(success, int)
    scores = np.asarray(scores, float)
    if not (len(state_ids) == len(checkpoints) == len(success) == len(scores)):
        raise ValueError("all arrays must align")

    rows = []
    for sid in np.unique(state_ids):
        idx = np.where(state_ids == sid)[0]
        ia = idx[checkpoints[idx] == str(checkpoint_a)]
        ib = idx[checkpoints[idx] == str(checkpoint_b)]
        if len(ia) != 1 or len(ib) != 1:
            continue
        ia, ib = int(ia[0]), int(ib[0])
        if success[ia] == success[ib]:
            continue
        winner_a = int(success[ia] == 1 and success[ib] == 0)
        rows.append((winner_a, float(scores[ia] - scores[ib])))

    if not rows:
        raise ValueError("no crossover states")
    y = np.asarray([r[0] for r in rows], int)
    rel = np.asarray([r[1] for r in rows], float)
    if len(np.unique(y)) < 2:
        raise ValueError("crossover states are one-directional; relative test is not identifiable")

    return {
        "n_crossover": int(len(y)),
        "a_wins": int(y.sum()),
        "b_wins": int((1 - y).sum()),
        "relative_auc": float(roc_auc_score(y, rel)),
        "zero_threshold_balanced_accuracy": float(balanced_accuracy_score(y, (rel > 0).astype(int))),
        "median_relative_score_a_wins": float(np.median(rel[y == 1])),
        "median_relative_score_b_wins": float(np.median(rel[y == 0])),
    }


def bootstrap_relative_auc(
    state_ids: np.ndarray,
    checkpoints: np.ndarray,
    success: np.ndarray,
    scores: np.ndarray,
    checkpoint_a: str,
    checkpoint_b: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap whole physical states, keeping the paired checkpoint rows together."""
    state_ids = np.asarray(state_ids)
    unique = np.unique(state_ids)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        new_sid, cp, yy, ss = [], [], [], []
        for k, sid in enumerate(picked):
            idx = np.where(state_ids == sid)[0]
            # Give duplicated bootstrap draws unique ids so they remain distinct pairs.
            new_sid.extend([k] * len(idx))
            cp.extend(np.asarray(checkpoints)[idx])
            yy.extend(np.asarray(success)[idx])
            ss.extend(np.asarray(scores)[idx])
        try:
            m = paired_relative_metrics(
                np.asarray(new_sid), np.asarray(cp), np.asarray(yy), np.asarray(ss),
                checkpoint_a, checkpoint_b,
            )
            vals.append(m["relative_auc"])
        except ValueError:
            continue
    point = paired_relative_metrics(
        state_ids, checkpoints, success, scores, checkpoint_a, checkpoint_b
    )["relative_auc"]
    if not vals:
        return {"point": point, "ci95": [None, None], "n_ok": 0}
    return {
        "point": point,
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "n_ok": int(len(vals)),
    }
