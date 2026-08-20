from __future__ import annotations

import numpy as np


def double_center(log_likelihoods: np.ndarray) -> np.ndarray:
    """Double-center a checkpoint x example log-likelihood matrix.

    Mirrors the geometry used by Kishino et al. (Findings ACL 2026):
    row-centering removes checkpoint-specific offsets and column-centering
    removes example difficulty shared across checkpoints.
    """
    x = np.asarray(log_likelihoods, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected [checkpoints, examples], got {x.shape}")
    return x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + x.mean()


def kl_proxy_bits_per_byte(
    q_i: np.ndarray,
    q_j: np.ndarray,
    mean_bytes: float,
) -> float:
    """Symmetric local KL proxy in bits/byte.

    The paper's local Euclidean approximation is
        2 KL(p_i, p_j) ~= ||q_i-q_j||^2 / N
    in nats. Dividing by mean bytes and ln(2) yields bits/byte.
    """
    if mean_bytes <= 0:
        raise ValueError("mean_bytes must be positive")
    qi = np.asarray(q_i, dtype=np.float64)
    qj = np.asarray(q_j, dtype=np.float64)
    if qi.shape != qj.shape or qi.ndim != 1:
        raise ValueError("q_i and q_j must be 1-D arrays of equal shape")
    n = qi.size
    return float(np.square(qi - qj).sum() / (2.0 * n * mean_bytes * np.log(2.0)))


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Linear centered kernel alignment without constructing n x n Gram matrices."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be 2-D with the same number of observations")
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
