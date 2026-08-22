from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import spearmanr, kendalltau


@dataclass
class RelationStats:
    spearman_rho: float
    kendall_tau: float
    depth_residual_spearman: float
    circular_shift_p: float
    topk_overlap: int
    topk_expected: float


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(kendalltau(x, y).statistic)


def quadratic_residual(y: np.ndarray) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, len(y))
    design = np.column_stack([np.ones(len(y)), x, x * x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def circular_shift_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Exact depth-preserving null over all non-zero circular layer shifts."""
    observed = abs(safe_spearman(x, y))
    null = np.array([abs(safe_spearman(x, np.roll(y, k))) for k in range(1, len(y))])
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def topk_overlap(x: np.ndarray, y: np.ndarray, k: int) -> tuple[int, float]:
    k = min(k, len(x))
    a = set(np.argsort(x)[-k:].tolist())
    b = set(np.argsort(y)[-k:].tolist())
    return len(a & b), k * k / len(x)


def relation_stats(necessity: np.ndarray, leverage: np.ndarray, topk: int = 5) -> RelationStats:
    overlap, expected = topk_overlap(necessity, leverage, topk)
    return RelationStats(
        spearman_rho=safe_spearman(necessity, leverage),
        kendall_tau=safe_kendall(necessity, leverage),
        depth_residual_spearman=safe_spearman(
            quadratic_residual(necessity), quadratic_residual(leverage)
        ),
        circular_shift_p=circular_shift_pvalue(necessity, leverage),
        topk_overlap=overlap,
        topk_expected=expected,
    )


def bootstrap_rho(
    baseline_by_task: Mapping[str, np.ndarray],
    layer_correct_by_task: Mapping[str, np.ndarray],
    leverage: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 20260822,
) -> np.ndarray:
    """Paired item bootstrap over the fixed evaluation ledger.

    `layer_correct_by_task[task]` has shape [num_layers, num_items].
    Each replicate resamples item indices *within each task*, then computes the
    equal-task-weight necessity curve before correlating it with fixed published
    RL leverage.
    """
    rng = np.random.default_rng(seed)
    task_names = sorted(baseline_by_task)
    out = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        damages = []
        for task in task_names:
            base = baseline_by_task[task]
            layer = layer_correct_by_task[task]
            idx = rng.integers(0, base.shape[0], size=base.shape[0])
            base_acc = float(base[idx].mean())
            layer_acc = layer[:, idx].mean(axis=1)
            damages.append(base_acc - layer_acc)
        necessity = np.mean(np.vstack(damages), axis=0)
        out[b] = safe_spearman(necessity, leverage)

    return out


def ci(values: np.ndarray, level: float = 0.90) -> tuple[float, float]:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - level
    return (
        float(np.quantile(vals, alpha / 2.0)),
        float(np.quantile(vals, 1.0 - alpha / 2.0)),
    )


def gate_label(rho: float, low: float, high: float, depth_resid: float) -> str:
    """Predeclared effect-size gate; never tune these after seeing Topic-12 data."""
    if np.isfinite(rho) and rho >= 0.50 and low >= 0.20:
        if np.isfinite(depth_resid) and depth_resid >= 0.25:
            return "STRONG_LAYER_LEVEL_ALIGNMENT"
        return "BROAD_DEPTH_ALIGNMENT_ONLY"
    if np.isfinite(rho) and rho <= -0.50 and high <= -0.20:
        return "STRONG_NEGATIVE_RELATION"
    if (
        np.isfinite(rho)
        and abs(rho) <= 0.20
        and low >= -0.35
        and high <= 0.35
    ):
        return "CREDIBLE_DISSOCIATION"
    return "INCONCLUSIVE_DO_NOT_TUNE"
