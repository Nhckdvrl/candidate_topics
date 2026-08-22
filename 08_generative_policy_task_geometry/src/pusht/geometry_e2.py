"""E2 - task geometry of the sampled action distribution.

Only meaningful if E1 shows that a scalar diversity score does *not* determine outcome
dispersion. E2 asks the follow-up: when two states have nearly the same scalar entropy but
different outcome dispersion, is the difference explained by *where* the action
variability points relative to the task?

Unlike the planar-arm prototype, there is no analytic Jacobian here and we do not want
one -- inventing an analytic task map would re-introduce the circularity of AUDIT.md A1.
Instead the local action->outcome map is *estimated from the counterfactual rollouts we
already ran*:

    Y (outcome keypoints, B x 16)  ~  X (action chunk, B x 16) @ W

W is a purely empirical local sensitivity. Its left singular vectors span the
task-sensitive action directions; the rest of action space is goal-equivalent to first
order.

Two guards, because a linearisation that does not hold explains nothing:

  * `r2_cv` -- W is fit on half the samples and scored on the held-out half. If the
    cross-fitted R^2 is low, the local linear model is not trustworthy at this state and
    the state is excluded from geometric interpretation (and that exclusion is reported,
    not hidden).
  * the *predicted* outcome dispersion tr(W^T Sigma W) is compared against the
    *measured* dispersion from the simulator. The simulator stays the ground truth.
"""

from __future__ import annotations

import numpy as np


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    d = x.shape[1]
    return np.linalg.solve(x.T @ x + lam * np.eye(d), x.T @ y)


def local_sensitivity(
    chunks: np.ndarray,
    keypoints: np.ndarray,
    lam: float = 1e-6,
    seed: int = 0,
) -> dict:
    """Estimate the local action->outcome map at one probe state.

    `chunks` is [B, H, 2] and `keypoints` is [B, K, 2], both from the same B
    counterfactual executions of one restored simulator state.
    """
    b = chunks.shape[0]
    x = chunks.reshape(b, -1).astype(np.float64)
    y = keypoints.reshape(b, -1).astype(np.float64)
    xm, ym = x.mean(0, keepdims=True), y.mean(0, keepdims=True)
    xc, yc = x - xm, y - ym

    # scale-free ridge: lam relative to the action covariance trace
    lam_eff = lam * np.trace(xc.T @ xc) / max(xc.shape[1], 1)

    # --- cross-fitted R^2: is the local linearisation credible at all?
    rng = np.random.default_rng(seed)
    perm = rng.permutation(b)
    half = b // 2
    r2s = []
    for tr, te in ((perm[:half], perm[half:]), (perm[half:], perm[:half])):
        w = _fit_ridge(xc[tr], yc[tr], lam_eff)
        pred = xc[te] @ w
        ss_res = float(((yc[te] - pred) ** 2).sum())
        ss_tot = float(((yc[te] - yc[tr].mean(0, keepdims=True)) ** 2).sum())
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
    r2_cv = float(np.mean(r2s))

    w_full = _fit_ridge(xc, yc, lam_eff)
    sigma = xc.T @ xc / max(b - 1, 1)

    # Task-sensitive action directions = left singular vectors of W with large singular
    # values. Everything else is goal-equivalent to first order.
    u, s, _ = np.linalg.svd(w_full, full_matrices=True)
    energy = s**2
    cum = np.cumsum(energy) / max(energy.sum(), 1e-12)
    rank = int(np.searchsorted(cum, 0.95) + 1) if energy.sum() > 0 else 0
    rank = max(1, min(rank, len(s)))

    u_task, u_null = u[:, :rank], u[:, rank:]
    var_task = float(np.trace(u_task.T @ sigma @ u_task))
    var_null = float(np.trace(u_null.T @ sigma @ u_null)) if u_null.shape[1] else 0.0
    n_task, n_null = u_task.shape[1], u_null.shape[1]

    predicted_outcome_var = float(np.trace(w_full.T @ sigma @ w_full))

    return {
        "r2_cv": r2_cv,
        "sensitivity_rank95": rank,
        "task_var_total": var_task,
        "null_var_total": var_null,
        "task_var_per_dim": var_task / max(n_task, 1),
        "null_var_per_dim": var_null / max(n_null, 1) if n_null else 0.0,
        "task_fraction": var_task / max(var_task + var_null, 1e-12),
        "action_var_total": float(np.trace(sigma)),
        # sqrt to put it in the same pixel units as outcome_kp_dispersion_px
        "predicted_outcome_dispersion_px": float(np.sqrt(predicted_outcome_var / keypoints.shape[1])),
        "top_singular_value": float(s[0]) if len(s) else 0.0,
    }
