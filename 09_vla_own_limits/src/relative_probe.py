"""Shared linear value probe and paired same-state confirmation analysis."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Frozen ridge grid. The action expert is 1024-wide while discovery supplies only
# 150 states x 2 checkpoints = 300 rows, so the fit is strongly overparameterized and the
# penalty cannot be left at an arbitrary constant.
RIDGE_ALPHA_GRID = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)


class SharedLinearProbe:
    """One fixed linear map shared by both checkpoints.

    Targets are Monte-Carlo success rates, not one stochastic success/failure draw.

    The ridge penalty is chosen by grouped cross-validation **inside the discovery split
    only**; confirmation states are never touched by the fit or the selection. Grouping is
    by physical `state_id`, which is not optional: `h_A(s)` and `h_B(s)` describe the same
    scene, so a plain K-fold would put two views of one state on both sides of the split,
    report an optimistic score, and select an alpha that is too small.

    Passing an explicit float disables selection and is used only by tests.
    """

    def __init__(self, alpha: float | str = "cv", n_splits: int = 5) -> None:
        self.scaler = StandardScaler()
        self.alpha_spec = alpha
        self.n_splits = int(n_splits)
        self.alpha_: float | None = None
        self.alpha_cv_: list[dict] | None = None
        self.model: Ridge | None = None

    def _select_alpha(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
        n_groups = len(np.unique(groups))
        splits = min(self.n_splits, n_groups)
        if splits < 2:
            raise ValueError("need at least two state groups to select a ridge penalty")
        cv = GroupKFold(n_splits=splits)
        self.alpha_cv_ = []
        for a in RIDGE_ALPHA_GRID:
            errs = []
            for tr, va in cv.split(x, y, groups=groups):
                sc = StandardScaler().fit(x[tr])
                m = Ridge(alpha=float(a)).fit(sc.transform(x[tr]), y[tr])
                errs.append(float(np.mean((m.predict(sc.transform(x[va])) - y[va]) ** 2)))
            self.alpha_cv_.append({"alpha": float(a), "cv_mse": float(np.mean(errs))})
        return min(self.alpha_cv_, key=lambda r: r["cv_mse"])["alpha"]

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> "SharedLinearProbe":
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be [N,D] and aligned with labels")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("non-finite feature or target")
        if np.min(y) < 0 or np.max(y) > 1:
            raise ValueError("success-rate target must lie in [0,1]")
        if np.std(y) == 0:
            raise ValueError("training targets are constant")

        if self.alpha_spec == "cv":
            if groups is None:
                raise ValueError("grouped alpha selection requires state_id groups")
            self.alpha_ = self._select_alpha(x, y, np.asarray(groups).astype(str))
        else:
            self.alpha_ = float(self.alpha_spec)
        self.model = Ridge(alpha=self.alpha_)
        self.model.fit(self.scaler.fit_transform(x), y)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("probe is not fitted")
        return np.asarray(self.model.predict(self.scaler.transform(np.asarray(x, float))), float)


def paired_relative_metrics(
    state_ids: np.ndarray,
    checkpoints: np.ndarray,
    winners: np.ndarray,
    scores: np.ndarray,
    checkpoint_a: str,
    checkpoint_b: str,
) -> dict:
    """Does q_A-q_B predict the robust Monte-Carlo winner on identical states?"""
    state_ids = np.asarray(state_ids).astype(str)
    checkpoints = np.asarray(checkpoints).astype(str)
    winners = np.asarray(winners).astype(str)
    scores = np.asarray(scores, float)
    if not (len(state_ids) == len(checkpoints) == len(winners) == len(scores)):
        raise ValueError("all arrays must align")

    rows = []
    for sid in np.unique(state_ids):
        idx = np.where(state_ids == sid)[0]
        ia = idx[checkpoints[idx] == str(checkpoint_a)]
        ib = idx[checkpoints[idx] == str(checkpoint_b)]
        if len(ia) != 1 or len(ib) != 1:
            continue
        ia, ib = int(ia[0]), int(ib[0])
        w = winners[ia]
        if winners[ib] != w:
            raise ValueError(f"winner label mismatch within state {sid}")
        if w not in {"A", "B"}:
            continue
        rows.append((sid, int(w == "A"), float(scores[ia] - scores[ib])))

    if not rows:
        raise ValueError("no robust crossover states")
    y = np.asarray([r[1] for r in rows], int)
    rel = np.asarray([r[2] for r in rows], float)
    if len(np.unique(y)) < 2:
        raise ValueError("robust crossover states are one-directional")
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
    winners: np.ndarray,
    scores: np.ndarray,
    checkpoint_a: str,
    checkpoint_b: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap physical states, preserving both checkpoint rows as one paired unit."""
    state_ids = np.asarray(state_ids).astype(str)
    unique = np.unique(state_ids)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        picked = rng.choice(unique, size=len(unique), replace=True)
        sid2, cp2, w2, sc2 = [], [], [], []
        for k, sid in enumerate(picked):
            idx = np.where(state_ids == sid)[0]
            sid2.extend([str(k)] * len(idx))
            cp2.extend(np.asarray(checkpoints)[idx])
            w2.extend(np.asarray(winners)[idx])
            sc2.extend(np.asarray(scores)[idx])
        try:
            vals.append(
                paired_relative_metrics(
                    np.asarray(sid2), np.asarray(cp2), np.asarray(w2), np.asarray(sc2),
                    checkpoint_a, checkpoint_b,
                )["relative_auc"]
            )
        except ValueError:
            continue
    point = paired_relative_metrics(
        state_ids, checkpoints, winners, scores, checkpoint_a, checkpoint_b
    )["relative_auc"]
    if not vals:
        return {"point": point, "ci95": [None, None], "n_ok": 0}
    return {
        "point": point,
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "n_ok": int(len(vals)),
    }


def absolute_success_metrics(
    state_ids: np.ndarray,
    checkpoints: np.ndarray,
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Pre-declared *power* control: does the readout track success at all?

    The paired test is designed so that a signal which is purely generic state difficulty
    cancels in `q_A - q_B`. That cancellation makes a null relative AUROC ambiguous: it
    happens both when the representation carries only state difficulty (an informative
    negative) and when the representation carries no success information whatsoever at
    the initial decision point (an uninformative measurement failure).

    This control separates those. It measures, *within* each checkpoint, the rank
    correlation between the shared readout and the Monte-Carlo success rate. It can only
    downgrade a negative result to "inconclusive"; it can never turn a negative into a
    positive, so it is not a rescue knob.
    """
    from scipy.stats import spearmanr

    checkpoints = np.asarray(checkpoints).astype(str)
    targets = np.asarray(targets, float)
    scores = np.asarray(scores, float)

    per_checkpoint = {}
    for cp in sorted(np.unique(checkpoints)):
        m = checkpoints == cp
        if m.sum() < 3 or np.std(targets[m]) == 0 or np.std(scores[m]) == 0:
            per_checkpoint[cp] = {"n": int(m.sum()), "spearman": None, "p_value": None}
            continue
        r = spearmanr(scores[m], targets[m])
        per_checkpoint[cp] = {
            "n": int(m.sum()),
            "spearman": float(r.statistic),
            "p_value": float(r.pvalue),
        }

    vals = [v["spearman"] for v in per_checkpoint.values() if v["spearman"] is not None]
    return {
        "per_checkpoint": per_checkpoint,
        "mean_within_checkpoint_spearman": float(np.mean(vals)) if vals else None,
    }
