from __future__ import annotations

import numpy as np


def double_center(log_likelihoods: np.ndarray) -> np.ndarray:
    x = np.asarray(log_likelihoods, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected [checkpoints, examples], got {x.shape}")
    return x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + x.mean()


def lower_clip_per_checkpoint(log_likelihoods: np.ndarray, quantile: float = 0.02) -> np.ndarray:
    x = np.asarray(log_likelihoods, dtype=np.float64).copy()
    if not (0.0 <= quantile < 0.5):
        raise ValueError("quantile must be in [0, 0.5)")
    if quantile == 0.0:
        return x
    floors = np.quantile(x, quantile, axis=1, keepdims=True)
    return np.maximum(x, floors)


def pair_outlier_mask(
    log_likelihoods: np.ndarray,
    pairs: list[tuple[int, int]],
    trim_fraction: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep examples whose maximum absolute pairwise LL change is not in the top tail."""
    x = np.asarray(log_likelihoods, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("log_likelihoods must be [checkpoints, examples]")
    if not (0.0 <= trim_fraction < 0.5):
        raise ValueError("trim_fraction must be in [0, 0.5)")
    deltas = np.stack([np.abs(x[j] - x[i]) for i, j in pairs], axis=0)
    scores = deltas.max(axis=0)
    if trim_fraction == 0.0:
        return np.ones(x.shape[1], dtype=bool), scores
    n_trim = max(1, int(np.floor(trim_fraction * x.shape[1])))
    order = np.argsort(scores)
    keep = np.ones(x.shape[1], dtype=bool)
    keep[order[-n_trim:]] = False
    return keep, scores


def kl_proxy_bits_per_byte(q_i: np.ndarray, q_j: np.ndarray, mean_bytes: float) -> float:
    if mean_bytes <= 0:
        raise ValueError("mean_bytes must be positive")
    qi = np.asarray(q_i, dtype=np.float64)
    qj = np.asarray(q_j, dtype=np.float64)
    if qi.shape != qj.shape or qi.ndim != 1:
        raise ValueError("q_i and q_j must be 1-D arrays of equal shape")
    return float(np.square(qi - qj).mean() / (2.0 * mean_bytes * np.log(2.0)))


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("x and y must be 2-D arrays with identical shape")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = np.square(xy).sum()
    denominator = np.sqrt(np.square(xx).sum() * np.square(yy).sum())
    if denominator <= eps:
        return float("nan")
    return float(numerator / denominator)


def mean_cosine_drift(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be 2-D arrays with identical shape")
    xn = np.linalg.norm(x, axis=1)
    yn = np.linalg.norm(y, axis=1)
    denom = np.maximum(xn * yn, eps)
    cosine = np.sum(x * y, axis=1) / denom
    return float(np.mean(1.0 - np.clip(cosine, -1.0, 1.0)))


def pooled_standardized_drift(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Dimensionless matched-state displacement using moments pooled across the pair."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be 2-D arrays with identical shape")
    pooled = np.concatenate([x, y], axis=0)
    mu = pooled.mean(axis=0, keepdims=True)
    sd = pooled.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    zx = (x - mu) / sd
    zy = (y - mu) / sd
    return float(np.square(zx - zy).mean())


def mean_by_example(values: np.ndarray, example_ids: np.ndarray, n_examples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    example_ids = np.asarray(example_ids, dtype=np.int64)
    if values.ndim != 1 or example_ids.shape != values.shape:
        raise ValueError("values/example_ids must be 1-D arrays of equal shape")
    sums = np.bincount(example_ids, weights=values, minlength=n_examples)
    counts = np.bincount(example_ids, minlength=n_examples)
    if np.any(counts == 0):
        raise ValueError("Every example must contribute at least one observation")
    return sums / counts
