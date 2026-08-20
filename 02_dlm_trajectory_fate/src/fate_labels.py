from __future__ import annotations

import numpy as np


def _validate_inputs(
    correct: np.ndarray,
    saved_steps: np.ndarray,
    observed: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(correct, dtype=bool)
    steps = np.asarray(saved_steps, dtype=np.int64)
    if c.ndim != 2:
        raise ValueError("correct must have shape [N, T]")
    if steps.ndim != 1:
        raise ValueError("saved_steps must be 1-D")
    if len(steps) == 0:
        raise ValueError("saved_steps must not be empty")
    if np.any(np.diff(steps) <= 0):
        raise ValueError("saved_steps must be strictly increasing and unique")
    if np.any(steps < 0) or np.any(steps >= c.shape[1]):
        raise ValueError("saved_steps out of range")

    if observed is None:
        obs = np.ones_like(c, dtype=bool)
    else:
        obs = np.asarray(observed, dtype=bool)
        if obs.shape != c.shape:
            raise ValueError("observed must have the same shape as correct")
    return c, steps, obs


def _first_future_match(
    condition: np.ndarray,
    observed: np.ndarray,
    start: int,
) -> int:
    """Return positive lead to first observed future match, or -1."""
    if start + 1 >= condition.shape[0]:
        return -1
    idx = np.flatnonzero(condition[start + 1 :] & observed[start + 1 :])
    return int(idx[0] + 1) if idx.size else -1


def _commitment_lead(
    correct: np.ndarray,
    observed: np.ndarray,
    start: int,
    final_value: bool,
) -> int:
    """Lead to first future observed state after which all observed states match final_value."""
    future_obs = np.flatnonzero(observed[start + 1 :]) + start + 1
    if future_obs.size == 0:
        return -1
    for u in future_obs:
        suffix = observed[u:]
        if np.any(suffix) and np.all(correct[u:][suffix] == final_value):
            return int(u - start)
    return -1


def fate_from_correctness(
    correct: np.ndarray,
    saved_steps: np.ndarray,
    observed: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Construct conditional trajectory labels from a complete surface trace.

    `observed` distinguishes a parseable wrong answer from "no answer yet".
    Labels use -1 where a conditional task is undefined.
    """
    c, steps, obs = _validate_inputs(correct, saved_steps, observed)
    n, _ = c.shape

    current_observed = obs[:, steps]
    current_correct = c[:, steps].astype(np.int8)
    current_correct[~current_observed] = -1

    final_observed_1d = obs[:, -1]
    final_correct_1d = c[:, -1]
    final_observed = np.repeat(final_observed_1d[:, None], len(steps), axis=1)
    final_correct = np.repeat(final_correct_1d[:, None], len(steps), axis=1).astype(np.int8)
    final_correct[~final_observed] = -1

    shape = (n, len(steps))
    recoverable = np.full(shape, -1, dtype=np.int8)
    recovery_lead = np.full(shape, -1, dtype=np.int16)
    will_overwrite = np.full(shape, -1, dtype=np.int8)
    overwrite_lead = np.full(shape, -1, dtype=np.int16)
    transient_recovery = np.full(shape, -1, dtype=np.int8)
    transient_overwrite = np.full(shape, -1, dtype=np.int8)
    finish_correct_from_wrong = np.full(shape, -1, dtype=np.int8)
    finish_wrong_from_correct = np.full(shape, -1, dtype=np.int8)
    final_commitment_lead = np.full(shape, -1, dtype=np.int16)

    for si, step in enumerate(steps.tolist()):
        for i in range(n):
            if not current_observed[i, si]:
                continue

            cur = bool(c[i, step])
            future_obs = obs[i, step + 1 :]
            future_c = c[i, step + 1 :]

            if not cur:
                hit = bool(np.any(future_obs & future_c))
                recoverable[i, si] = int(hit)
                if hit:
                    recovery_lead[i, si] = _first_future_match(c[i], obs[i], step)

                if final_observed_1d[i]:
                    finish_correct_from_wrong[i, si] = int(final_correct_1d[i])
                    if not final_correct_1d[i]:
                        transient_recovery[i, si] = int(hit)
            else:
                hit = bool(np.any(future_obs & ~future_c))
                will_overwrite[i, si] = int(hit)
                if hit:
                    overwrite_lead[i, si] = _first_future_match(~c[i], obs[i], step)

                if final_observed_1d[i]:
                    finish_wrong_from_correct[i, si] = int(not final_correct_1d[i])
                    if final_correct_1d[i]:
                        transient_overwrite[i, si] = int(hit)

            if final_observed_1d[i]:
                final_commitment_lead[i, si] = _commitment_lead(
                    c[i], obs[i], step, bool(final_correct_1d[i])
                )

    return {
        "current_observed": current_observed.astype(np.int8),
        "current_correct": current_correct,
        "final_observed": final_observed.astype(np.int8),
        "final_correct": final_correct,
        "recoverable": recoverable,
        "recovery_lead": recovery_lead,
        "will_overwrite": will_overwrite,
        "overwrite_lead": overwrite_lead,
        "transient_recovery": transient_recovery,
        "transient_overwrite": transient_overwrite,
        "finish_correct_from_wrong": finish_correct_from_wrong,
        "finish_wrong_from_correct": finish_wrong_from_correct,
        "final_commitment_lead": final_commitment_lead,
    }
