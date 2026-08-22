"""Deterministic identifiers for LIBERO physical states and policy-noise streams."""
from __future__ import annotations

import hashlib
import numpy as np


def hash_sim_state(flat_state: np.ndarray, *, decimals: int = 10) -> str:
    """Hash settled MuJoCo state, rounding only below numerical-noise scale."""
    x = np.asarray(flat_state, dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("sim state contains non-finite values")
    x = np.round(x, decimals=decimals)
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def deterministic_noise_seed(
    policy_seed: int,
    *,
    suite: str,
    task_id: int,
    init_idx: int,
    replan_idx: int,
) -> int:
    """Derive one reproducible inference-noise seed per decision.

    The same base policy_seed therefore defines the same Gaussian-noise stream for every
    checkpoint on the same physical state (common random numbers), without reusing one
    identical noise tensor at every replanning step.
    """
    payload = f"{int(policy_seed)}|{suite}|{int(task_id)}|{int(init_idx)}|{int(replan_idx)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"topic09").digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF
