from __future__ import annotations

import numpy as np


def fate_from_correctness(correct: np.ndarray, saved_steps: np.ndarray) -> dict[str, np.ndarray]:
    """Construct trajectory-fate labels from complete per-step surface correctness.

    correct: bool [N,T] from full x0 predictions before partial transfer into x.
    saved_steps: steps for which hidden states were captured.
    """
    c = np.asarray(correct, dtype=bool)
    steps = np.asarray(saved_steps, dtype=np.int64)
    if c.ndim != 2:
        raise ValueError("correct must have shape [N, T]")
    n, tmax = c.shape
    if np.any(steps < 0) or np.any(steps >= tmax):
        raise ValueError("saved_steps out of range")

    current = c[:, steps]
    recoverable = np.full((n, len(steps)), -1, dtype=np.int8)
    recovery_lead = np.full((n, len(steps)), -1, dtype=np.int16)
    overwrite = np.full((n, len(steps)), -1, dtype=np.int8)
    overwrite_lead = np.full((n, len(steps)), -1, dtype=np.int16)

    for si, step in enumerate(steps.tolist()):
        future = c[:, step + 1 :]
        for i in range(n):
            if not current[i, si]:
                if future.shape[1] == 0:
                    recoverable[i, si] = 0
                    continue
                hits = np.flatnonzero(future[i])
                recoverable[i, si] = int(hits.size > 0)
                if hits.size:
                    recovery_lead[i, si] = int(hits[0] + 1)
            else:
                if future.shape[1] == 0:
                    overwrite[i, si] = 0
                    continue
                hits = np.flatnonzero(~future[i])
                overwrite[i, si] = int(hits.size > 0)
                if hits.size:
                    overwrite_lead[i, si] = int(hits[0] + 1)

    return {
        "current_correct": current.astype(np.int8),
        "recoverable": recoverable,
        "recovery_lead": recovery_lead,
        "will_overwrite": overwrite,
        "overwrite_lead": overwrite_lead,
    }
