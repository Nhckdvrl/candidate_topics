"""Action-Chunk Entropy (ACE), reproducing the released FIPER implementation.

Transcribed from `utiasDSL/fiper`:
  * `evaluation/method_eval_classes/entropy_eval.py` -> `ENTROPYEval`
  * `configs/eval/entropy.yaml`  -> cellsize_factor: 0.03, single_cellsize: False
  * `configs/task/push_t.yaml`   -> position action mapping, horizon 16, batch 256

Behaviour we deliberately keep identical to the release, because ACE is the baseline we
are trying *not* to strawman:

  * cell width per dimension = cellsize_factor * (calibration range of that dimension),
    where the calibration range is max-min over a pooled calibration set of predicted
    action chunks;
  * zero-range dimensions are replaced by the max range (so a padded constant dimension
    contributes a single cell and therefore zero entropy);
  * the histogram grid limits are recomputed *per state* from the sampled batch, padded
    by 1% of the batch spread, and indices come from `np.digitize`;
  * entropy is Shannon entropy in bits over the cell counts (empty cells contribute 0);
  * the chunk score is the **mean** over prediction-horizon steps.

The one thing we do not copy is FIPER's hard-coded 3-D `_entropy_endpoints`. PushT
actions are 2-D. `test_ace.py` checks that scoring native 2-D actions equals scoring the
same actions zero-padded to 3-D and run through the released code path, so this is a
refactor, not a change.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as _shannon

FIPER_CELLSIZE_FACTOR = 0.03


def calibration_cell_size(
    calibration_chunks: np.ndarray,
    cellsize_factor: float = FIPER_CELLSIZE_FACTOR,
    single_cellsize: bool = False,
) -> np.ndarray:
    """Per-dimension cell width from pooled calibration action predictions.

    `calibration_chunks` is [N, H, D] or [N, D]; FIPER flattens everything but the last
    axis before taking the range.
    """
    x = np.asarray(calibration_chunks, dtype=np.float64).reshape(-1, np.shape(calibration_chunks)[-1])
    ranges = x.max(axis=0) - x.min(axis=0)
    max_range = ranges.max()
    if single_cellsize:
        ranges = np.full_like(ranges, max_range)
    else:
        ranges = np.where(ranges == 0, max_range, ranges)
    return ranges * cellsize_factor


def _step_entropy(points: np.ndarray, cell_size: np.ndarray) -> float:
    """FIPER `_entropy_endpoints`, generalised from 3 dims to D dims."""
    p = np.asarray(points, dtype=np.float64)
    d = p.shape[1]
    idx = np.empty_like(p, dtype=np.int64)
    n_cells = np.empty(d, dtype=np.int64)
    for k in range(d):
        lo, hi = p[:, k].min(), p[:, k].max()
        buf = 0.01 * (hi - lo)
        lo, hi = lo - buf, hi + buf
        grid = np.arange(lo, hi + cell_size[k], cell_size[k])
        idx[:, k] = np.digitize(p[:, k], grid) - 1
        n_cells[k] = max(len(grid) - 1, 1)
    # ravel to a flat cell id; counting via unique is equivalent to FIPER's dense
    # count array but does not allocate prod(n_cells) entries.
    idx = np.clip(idx, 0, np.maximum(n_cells - 1, 0))
    flat = np.ravel_multi_index([idx[:, k] for k in range(d)], tuple(np.maximum(n_cells, 1)))
    counts = np.bincount(flat)
    return float(_shannon(counts, base=2))


def ace(chunks: np.ndarray, cell_size: np.ndarray) -> float:
    """ACE for one state: mean over prediction steps of the joint-cell entropy.

    `chunks` is [B, H, D]: B action chunks sampled from the policy at one observation.
    """
    x = np.asarray(chunks, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"chunks must be [B, H, D], got {x.shape}")
    return float(np.mean([_step_entropy(x[:, h, :], cell_size) for h in range(x.shape[1])]))


def action_dispersion(chunks: np.ndarray) -> dict:
    """Estimator-free scalar diversity summaries, so no conclusion rests on ACE alone."""
    x = np.asarray(chunks, dtype=np.float64)
    b, h, d = x.shape
    per_step_trace = np.array([np.trace(np.cov(x[:, t, :], rowvar=False)) for t in range(h)])
    flat = x.reshape(b, h * d)
    centred = flat - flat.mean(axis=0, keepdims=True)
    return {
        "act_rms_dispersion": float(np.sqrt((centred**2).sum(axis=1).mean() / h)),
        "act_trace_cov_mean": float(per_step_trace.mean()),
        "act_std_mean": float(np.sqrt(np.maximum(per_step_trace, 0) / d).mean()),
    }
