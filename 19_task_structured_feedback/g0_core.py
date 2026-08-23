from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

RIGHT_ARM_STATE_SLICE = slice(22, 29)   # SIMPLE G1 qpos ordering
RIGHT_ARM_ACTION_SLICE = slice(21, 28)  # Psi0 36-D action ordering


@dataclass(frozen=True)
class PerturbationPair:
    task: np.ndarray
    null: np.ndarray
    epsilon: float
    rank6: int
    smallest_nonzero_singular: float


@dataclass(frozen=True)
class ResponseMetrics:
    accommodation_task: float
    accommodation_null: float
    correction_task: float
    correction_null: float
    delta_correction: float


def _as_2d(x: np.ndarray, shape0: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected matrix, got shape {x.shape}")
    if shape0 is not None and x.shape[0] != shape0:
        raise ValueError(f"expected first dim={shape0}, got {x.shape}")
    return x


def construct_perturbations(
    jac_pos: np.ndarray,
    jac_rot: np.ndarray,
    epsilon: float = 0.08,
    rank_tol: float = 1e-8,
) -> PerturbationPair:
    """Construct equal-norm task-space and full-pose-null right-arm perturbations.

    jac_pos, jac_rot are 3x7 Jacobians restricted to the seven right-arm DoFs.
    The null direction is the final right singular vector of the full 6x7
    geometric Jacobian. The task direction is the top right singular vector of
    the 3x7 positional Jacobian, which maximizes first-order wrist translation
    for a fixed joint-space norm.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    jp = _as_2d(jac_pos, 3)
    jr = _as_2d(jac_rot, 3)
    if jp.shape[1] != 7 or jr.shape[1] != 7:
        raise ValueError("expected 7 right-arm DoFs")

    j6 = np.concatenate([jp, jr], axis=0)
    _, s6, vt6 = np.linalg.svd(j6, full_matrices=True)
    rank6 = int(np.sum(s6 > rank_tol))
    if rank6 < 6:
        raise ValueError(f"6D wrist Jacobian is rank-deficient (rank={rank6}); skip state")

    null_dir = vt6[-1]
    _, sp, vtp = np.linalg.svd(jp, full_matrices=True)
    if not np.isfinite(sp[0]) or sp[0] <= rank_tol:
        raise ValueError("positional wrist Jacobian has no usable task direction")
    task_dir = vtp[0]

    def canonical(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        idx = int(np.argmax(np.abs(v)))
        if v[idx] < 0:
            v = -v
        return v / np.linalg.norm(v)

    task_dir = canonical(task_dir)
    null_dir = canonical(null_dir)
    task = epsilon * task_dir
    null = epsilon * null_dir

    if not np.isclose(np.linalg.norm(task), np.linalg.norm(null), rtol=0, atol=1e-12):
        raise AssertionError("paired perturbations lost equal-norm matching")

    return PerturbationPair(
        task=task,
        null=null,
        epsilon=float(epsilon),
        rank6=rank6,
        smallest_nonzero_singular=float(s6[-1]),
    )


def accommodation(base_target: np.ndarray, perturbed_target: np.ndarray, delta: np.ndarray) -> float:
    """Fraction of a joint perturbation carried into Psi0's new absolute target.

    0: policy target does not move with the perturbation -> downstream WBC tends
       to restore the original posture.
    1: policy target shifts one-for-one with the perturbation -> downstream WBC
       tends to preserve/accommodate that deviation.
    """
    a0 = np.asarray(base_target, dtype=np.float64).reshape(-1)
    ad = np.asarray(perturbed_target, dtype=np.float64).reshape(-1)
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    if a0.shape != ad.shape or a0.shape != d.shape:
        raise ValueError(f"shape mismatch: {a0.shape}, {ad.shape}, {d.shape}")
    denom = float(d @ d)
    if denom <= 0:
        raise ValueError("zero perturbation")
    return float((ad - a0) @ d / denom)


def response_metrics(
    base_target: np.ndarray,
    task_target: np.ndarray,
    null_target: np.ndarray,
    delta_task: np.ndarray,
    delta_null: np.ndarray,
) -> ResponseMetrics:
    at = accommodation(base_target, task_target, delta_task)
    an = accommodation(base_target, null_target, delta_null)
    rt = 1.0 - at
    rn = 1.0 - an
    return ResponseMetrics(
        accommodation_task=at,
        accommodation_null=an,
        correction_task=rt,
        correction_null=rn,
        delta_correction=rt - rn,
    )


def finite_geometry_gate(
    task_translation_m: float,
    null_translation_m: float,
    null_rotation_rad: float,
    min_task_translation_m: float = 0.005,
    max_null_translation_m: float = 0.002,
    max_null_rotation_rad: float = np.deg2rad(1.0),
    min_translation_ratio: float = 5.0,
) -> tuple[bool, dict[str, float | bool]]:
    """Verify that finite perturbations instantiate the intended local contrast."""
    ratio = float(task_translation_m / max(null_translation_m, 1e-9))
    checks = {
        "task_translation_ok": bool(task_translation_m >= min_task_translation_m),
        "null_translation_ok": bool(null_translation_m <= max_null_translation_m),
        "null_rotation_ok": bool(null_rotation_rad <= max_null_rotation_rad),
        "translation_ratio_ok": bool(ratio >= min_translation_ratio),
    }
    return bool(all(checks.values())), {
        **checks,
        "task_translation_m": float(task_translation_m),
        "null_translation_m": float(null_translation_m),
        "null_rotation_rad": float(null_rotation_rad),
        "translation_ratio": ratio,
    }


def aggregate_by_episode(rows: Iterable[dict]) -> np.ndarray:
    """Return one mean delta per episode; inference seeds/states stay nested."""
    groups: dict[object, list[float]] = {}
    for row in rows:
        groups.setdefault(row["episode_id"], []).append(float(row["delta_correction"]))
    if not groups:
        raise ValueError("no rows")
    return np.asarray([np.mean(v) for _, v in sorted(groups.items(), key=lambda kv: str(kv[0]))], dtype=np.float64)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 20260823) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 2:
        raise ValueError("need at least two independent episodes")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def verdict(mean_delta: float, ci_low: float, ci_high: float) -> str:
    """Frozen G0 gate in dimensionless correction-fraction units."""
    if mean_delta >= 0.20 and ci_low > 0.0:
        return "PROCEED_TASK_STRUCTURED_FEEDBACK"
    if ci_high <= 0.10:
        return "KILL_NO_MEANINGFUL_TASK_SELECTIVITY"
    return "INCONCLUSIVE_DO_NOT_TUNE"
