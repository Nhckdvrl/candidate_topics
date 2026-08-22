import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topic12.stats import (
    circular_shift_pvalue,
    gate_label,
    quadratic_residual,
    relation_stats,
    safe_spearman,
)


def test_spearman_perfect():
    x = np.arange(28, dtype=float)
    assert safe_spearman(x, x) > 0.999


def test_depth_residual_removes_quadratic_shape():
    x = np.linspace(-1, 1, 28)
    y = 3 + 2*x - 4*x*x
    assert np.max(np.abs(quadratic_residual(y))) < 1e-10


def test_relation_stats_topk():
    x = np.arange(28, dtype=float)
    stats = relation_stats(x, x, topk=5)
    assert stats.topk_overlap == 5
    assert stats.spearman_rho > 0.999


def test_gate_labels_are_locked():
    assert gate_label(0.7, 0.3, 0.9, 0.4) == "STRONG_LAYER_LEVEL_ALIGNMENT"
    assert gate_label(0.7, 0.3, 0.9, 0.1) == "BROAD_DEPTH_ALIGNMENT_ONLY"
    assert gate_label(-0.7, -0.9, -0.3, -0.4) == "STRONG_NEGATIVE_RELATION"
    assert gate_label(0.05, -0.2, 0.2, 0.0) == "CREDIBLE_DISSOCIATION"
