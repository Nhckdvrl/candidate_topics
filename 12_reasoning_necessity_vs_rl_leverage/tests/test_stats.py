import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topic12.stats import (
    competence_loss_curve,
    gate_label,
    intervention_label,
    quadratic_residual,
    relation_stats,
    safe_spearman,
)


def test_spearman_perfect():
    x = np.arange(28, dtype=float)
    assert safe_spearman(x, x) > 0.999


def test_conditional_necessity_does_not_cancel_with_lucky_gains():
    base = np.array([1, 1, 0, 0], dtype=float)
    layer = np.array([[0, 1, 1, 0]], dtype=float)
    curve = competence_loss_curve(base, layer)
    assert np.isclose(curve[0], 0.5)


def test_quadratic_depth_residual_removes_shared_broad_shape():
    d = np.linspace(-1, 1, 28)
    x = 2.0 - 3.0 * d * d
    y = -1.0 - 7.0 * d * d
    assert safe_spearman(x, y) > 0.99
    assert np.max(np.abs(quadratic_residual(x))) < 1e-10
    assert np.max(np.abs(quadratic_residual(y))) < 1e-10


def test_relation_stats_topk():
    x = np.arange(28, dtype=float)
    stats = relation_stats(x, x, topk=5)
    assert stats.topk_overlap == 5
    assert stats.spearman_rho > 0.999


def test_intervention_gate_prevents_destructive_deletion_from_being_scientific_null():
    necessity = np.array([0.95] * 8 + [0.2] * 20)
    label = intervention_label(necessity)
    assert label == "TOO_DESTRUCTIVE_USE_MILD_SWEEP"
    assert gate_label(0.0, -0.1, 0.1, 0.0, label) == "INCONCLUSIVE_INTERVENTION"


def test_gate_labels_are_locked():
    assert gate_label(0.7, 0.3, 0.9, 0.4) == "STRONG_LAYER_LEVEL_ALIGNMENT"
    assert gate_label(0.7, 0.3, 0.9, 0.1) == "BROAD_DEPTH_ALIGNMENT_ONLY"
    assert gate_label(-0.7, -0.9, -0.3, -0.4) == "STRONG_NEGATIVE_RELATION"
    assert gate_label(0.05, -0.2, 0.2, 0.0) == "DISSOCIATION_CANDIDATE"
