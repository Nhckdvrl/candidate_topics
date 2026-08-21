from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class VarianceDecomposition:
    task_total: float
    null_total: float
    task_per_dim: float
    null_per_dim: float
    task_rank: int
    null_rank: int
    total_variance: float

    @property
    def task_fraction(self) -> float:
        denom = self.task_total + self.null_total
        return float(self.task_total / denom) if denom > 0 else 0.0


def task_null_projectors(jacobian: np.ndarray, rtol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return orthogonal projectors onto row(J) and null(J)."""
    j = np.asarray(jacobian, dtype=np.float64)
    if j.ndim != 2:
        raise ValueError(f"jacobian must be 2D, got {j.shape}")
    _, s, vt = np.linalg.svd(j, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        tol = rtol * max(j.shape) * s[0]
        rank = int(np.sum(s > tol))
    d = j.shape[1]
    v_task = vt[:rank].T if rank > 0 else np.zeros((d, 0), dtype=np.float64)
    v_null = vt[rank:].T if rank < d else np.zeros((d, 0), dtype=np.float64)
    p_task = v_task @ v_task.T if rank > 0 else np.zeros((d, d), dtype=np.float64)
    p_null = v_null @ v_null.T if rank < d else np.zeros((d, d), dtype=np.float64)
    return p_task, p_null, rank, d - rank


def sample_covariance(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must have shape [B,D], got {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("need at least two samples")
    centered = x - x.mean(axis=0, keepdims=True)
    return centered.T @ centered / (x.shape[0] - 1)


def decompose_variance(samples: np.ndarray, jacobian: np.ndarray, rtol: float = 1e-8) -> VarianceDecomposition:
    cov = sample_covariance(samples)
    p_task, p_null, task_rank, null_rank = task_null_projectors(jacobian, rtol=rtol)
    task_total = float(np.trace(p_task @ cov @ p_task))
    null_total = float(np.trace(p_null @ cov @ p_null))
    task_per_dim = task_total / task_rank if task_rank > 0 else 0.0
    null_per_dim = null_total / null_rank if null_rank > 0 else 0.0
    return VarianceDecomposition(
        task_total=task_total,
        null_total=null_total,
        task_per_dim=float(task_per_dim),
        null_per_dim=float(null_per_dim),
        task_rank=task_rank,
        null_rank=null_rank,
        total_variance=float(np.trace(cov)),
    )


def decompose_chunks(chunks: np.ndarray, jacobian: np.ndarray) -> dict:
    """Decompose each prediction step with a frozen local Jacobian."""
    x = np.asarray(chunks, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"chunks must have shape [B,H,D], got {x.shape}")
    per_step = [decompose_variance(x[:, h, :], jacobian) for h in range(x.shape[1])]
    task_sum = sum(v.task_total for v in per_step)
    null_sum = sum(v.null_total for v in per_step)
    return {
        "task_total_sum": float(task_sum),
        "null_total_sum": float(null_sum),
        "task_per_dim_sum": float(sum(v.task_per_dim for v in per_step)),
        "null_per_dim_sum": float(sum(v.null_per_dim for v in per_step)),
        "total_variance_sum": float(sum(v.total_variance for v in per_step)),
        "task_fraction": float(task_sum / max(task_sum + null_sum, 1e-12)),
        "task_rank": int(per_step[0].task_rank),
        "null_rank": int(per_step[0].null_rank),
        "step0": per_step[0],
    }


def fiper_calibration_ranges(calibration_chunks: np.ndarray, min_range: float = 1e-6) -> np.ndarray:
    """Per-dimension action ranges R_d used by FIPER ACE."""
    x = np.asarray(calibration_chunks, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"calibration_chunks must be [N,H,D], got {x.shape}")
    r = x.max(axis=(0, 1)) - x.min(axis=(0, 1))
    return np.maximum(r, min_range)


def _joint_hist_entropy(actions: np.ndarray, ranges: np.ndarray, alpha: float = 0.1) -> float:
    """FIPER-style joint-cell histogram entropy for one predicted timestep."""
    a = np.asarray(actions, dtype=np.float64)
    r = np.asarray(ranges, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"actions must be [B,D], got {a.shape}")
    if r.shape != (a.shape[1],):
        raise ValueError(f"ranges must be {(a.shape[1],)}, got {r.shape}")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    lo = a.min(axis=0)
    hi = a.max(axis=0)
    cell = alpha * np.maximum(r, 1e-12)
    n_bins = np.maximum(1, np.ceil((hi - lo) / cell).astype(np.int64))
    idx = np.floor((a - lo) / cell).astype(np.int64)
    idx = np.minimum(idx, n_bins - 1)
    _, counts = np.unique(idx, axis=0, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-(p * np.log2(p)).sum())


def fiper_ace(chunks: np.ndarray, calibration_ranges: np.ndarray, alpha: float = 0.1) -> float:
    """Action-Chunk Entropy (ACE), summing histogram entropy over horizon."""
    x = np.asarray(chunks, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"chunks must be [B,H,D], got {x.shape}")
    return float(sum(_joint_hist_entropy(x[:, h, :], calibration_ranges, alpha=alpha) for h in range(x.shape[1])))
