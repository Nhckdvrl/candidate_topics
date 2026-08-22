"""ACE must match the *released* FIPER estimator, not a paraphrase of the paper."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import entropy as shannon

from src.pusht.ace import FIPER_CELLSIZE_FACTOR, ace, action_dispersion, calibration_cell_size


def _fiper_reference(chunks3: np.ndarray, cell: np.ndarray) -> float:
    """Literal transcription of `ENTROPYEval._entropy_endpoints` + its horizon average."""

    def one_step(endpoints):
        grids = []
        for k in range(3):
            mn, mx = endpoints[:, k].min(), endpoints[:, k].max()
            buf = 0.01 * (mx - mn)
            grids.append(np.arange(mn - buf, mx + buf + cell[k], cell[k]))
        idx = [np.digitize(endpoints[:, k], grids[k]) - 1 for k in range(3)]
        n = [max(len(g) - 1, 1) for g in grids]
        counts = np.zeros(n, dtype=int)
        for i in range(len(endpoints)):
            counts[idx[0][i], idx[1][i], idx[2][i]] += 1
        return float(shannon(counts.flatten(), base=2))

    return float(np.mean([one_step(chunks3[:, h, :]) for h in range(chunks3.shape[1])]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ace_equals_released_fiper_on_padded_actions(seed):
    rng = np.random.default_rng(seed)
    calib = 256.0 + rng.normal(scale=40.0, size=(300, 16, 2))
    chunks = 256.0 + rng.normal(scale=15.0, size=(64, 16, 2))

    ours = ace(chunks, calibration_cell_size(calib))

    pad = lambda a: np.concatenate([a, np.zeros((*a.shape[:2], 1))], axis=-1)
    pos = pad(calib).reshape(-1, 3)
    ranges = pos.max(0) - pos.min(0)
    ranges = np.where(ranges == 0, ranges.max(), ranges)
    ref = _fiper_reference(pad(chunks), ranges * FIPER_CELLSIZE_FACTOR)

    assert ours == pytest.approx(ref, abs=1e-9)


def test_cell_size_uses_released_constant_and_zero_range_rule():
    x = np.stack([np.linspace(0, 10, 20), np.full(20, 3.0)], axis=-1)[None].repeat(4, 0)
    cell = calibration_cell_size(x)
    assert cell[0] == pytest.approx(10.0 * FIPER_CELLSIZE_FACTOR)
    # a constant dimension inherits the max range, so it forms a single cell
    assert cell[1] == pytest.approx(10.0 * FIPER_CELLSIZE_FACTOR)


def test_ace_is_zero_for_identical_samples():
    chunks = np.tile(np.linspace(0, 1, 16)[None, :, None], (32, 1, 2))
    cell = np.array([0.1, 0.1])
    assert ace(chunks, cell) == pytest.approx(0.0)


def test_ace_increases_with_spread():
    rng = np.random.default_rng(0)
    calib = 256.0 + rng.normal(scale=40.0, size=(300, 16, 2))
    cell = calibration_cell_size(calib)
    narrow = 256.0 + rng.normal(scale=1.0, size=(128, 16, 2))
    wide = 256.0 + rng.normal(scale=40.0, size=(128, 16, 2))
    assert ace(wide, cell) > ace(narrow, cell)


def test_action_dispersion_scales_linearly():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(64, 8, 2))
    a = action_dispersion(base)
    b = action_dispersion(base * 3.0)
    assert b["act_rms_dispersion"] == pytest.approx(3.0 * a["act_rms_dispersion"], rel=1e-9)
