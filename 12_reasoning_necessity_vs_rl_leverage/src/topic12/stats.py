from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


@dataclass
class RelationStats:
    spearman_rho: float
    kendall_tau: float
    depth_residual_spearman: float
    depth_partial_rank: float
    circular_shift_p: float
    topk_overlap: int
    topk_expected: float


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(kendalltau(x, y).statistic)


def quadratic_residual(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.linspace(-1.0, 1.0, len(y))
    design = np.column_stack([np.ones(len(y)), x, x * x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def partial_spearman_depth(x: np.ndarray, y: np.ndarray) -> float:
    """Descriptive partial-rank diagnostic controlling linear+quadratic depth."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    depth = np.linspace(-1.0, 1.0, len(x))
    design = np.column_stack([np.ones(len(x)), depth, depth * depth])
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    bx, *_ = np.linalg.lstsq(design, rx, rcond=None)
    by, *_ = np.linalg.lstsq(design, ry, rcond=None)
    ex = rx - design @ bx
    ey = ry - design @ by
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def circular_shift_pvalue(x: np.ndarray, y: np.ndarray) -> float:
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


def competence_loss_curve(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Fraction of baseline-solved items broken by each layer intervention."""
    base = np.asarray(base, dtype=float)
    layer = np.asarray(layer, dtype=float)
    solved = base == 1.0
    if solved.sum() == 0:
        return np.full(layer.shape[0], np.nan)
    return np.mean(layer[:, solved] == 0.0, axis=1)


def net_accuracy_drop_curve(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    layer = np.asarray(layer, dtype=float)
    return float(base.mean()) - layer.mean(axis=1)


def relation_stats(necessity: np.ndarray, leverage: np.ndarray, topk: int = 5) -> RelationStats:
    overlap, expected = topk_overlap(necessity, leverage, topk)
    return RelationStats(
        spearman_rho=safe_spearman(necessity, leverage),
        kendall_tau=safe_kendall(necessity, leverage),
        depth_residual_spearman=safe_spearman(
            quadratic_residual(necessity), quadratic_residual(leverage)
        ),
        depth_partial_rank=partial_spearman_depth(necessity, leverage),
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
    metric: str = "conditional_loss",
) -> np.ndarray:
    """Paired item bootstrap over the frozen ledger."""
    rng = np.random.default_rng(seed)
    task_names = sorted(baseline_by_task)
    out = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        task_curves = []
        for task in task_names:
            base = baseline_by_task[task]
            layer = layer_correct_by_task[task]
            idx = rng.integers(0, base.shape[0], size=base.shape[0])
            if metric == "conditional_loss":
                curve = competence_loss_curve(base[idx], layer[:, idx])
            elif metric == "net_drop":
                curve = net_accuracy_drop_curve(base[idx], layer[:, idx])
            else:
                raise ValueError(metric)
            task_curves.append(curve)
        necessity = np.nanmean(np.vstack(task_curves), axis=0)
        out[b] = safe_spearman(necessity, leverage)
    return out


def ci(values: np.ndarray, level: float = 0.90) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - level
    return float(np.quantile(vals, alpha / 2.0)), float(np.quantile(vals, 1.0 - alpha / 2.0))


def intervention_label(
    necessity: np.ndarray,
    parser_fallback: np.ndarray | None = None,
    truncation: np.ndarray | None = None,
) -> str:
    """Detect when full deletion is too destructive to rank layers cleanly."""
    necessity = np.asarray(necessity, dtype=float)
    if np.mean(necessity >= 0.90) >= 0.25:
        return "TOO_DESTRUCTIVE_USE_MILD_SWEEP"
    if parser_fallback is not None and np.mean(np.asarray(parser_fallback) >= 0.50) >= 0.25:
        return "TOO_DESTRUCTIVE_USE_MILD_SWEEP"
    if truncation is not None and np.mean(np.asarray(truncation) >= 0.50) >= 0.25:
        return "TOO_DESTRUCTIVE_USE_MILD_SWEEP"
    return "INFORMATIVE"


def gate_label(
    rho: float,
    low: float,
    high: float,
    depth_residual: float,
    intervention: str = "INFORMATIVE",
) -> str:
    if intervention != "INFORMATIVE":
        return "INCONCLUSIVE_INTERVENTION"
    if np.isfinite(rho) and rho >= 0.50 and low >= 0.20:
        if np.isfinite(depth_residual) and depth_residual >= 0.25:
            return "STRONG_LAYER_LEVEL_ALIGNMENT"
        return "BROAD_DEPTH_ALIGNMENT_ONLY"
    if np.isfinite(rho) and rho <= -0.50 and high <= -0.20:
        return "STRONG_NEGATIVE_RELATION"
    if np.isfinite(rho) and abs(rho) <= 0.20 and low >= -0.35 and high <= 0.35:
        return "DISSOCIATION_CANDIDATE"
    return "INCONCLUSIVE_DO_NOT_TUNE"
