import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from train_pairwise_probe import disagreement_hidden_win, pair_metrics


def test_pair_metrics_rewards_matched_target_flip():
    y = np.array([0, 1, 0, 1] * 20)
    p_orig = np.where(y == 1, 0.9, 0.1)
    p_flip = 1.0 - p_orig
    m_orig = np.where(y == 1, 2.0, -2.0)
    m_flip = -m_orig
    out = pair_metrics(y, p_orig, p_flip, m_orig, m_flip, n_boot=200, seed=0)
    assert out["pair_hidden_auc"] == 1.0
    assert out["pair_hidden_accuracy"] == 1.0
    assert out["target_flip_direction_accuracy"] == 1.0


def test_disagreement_subset_is_label_free_and_hidden_can_win():
    y = np.array([0, 1, 0, 1] * 25)
    # Hidden predictions are correct in both conditions.
    p_orig = np.where(y == 1, 0.9, 0.1)
    p_flip = 1.0 - p_orig
    # Native output is confidently wrong in original and target-flip conditions.
    m_orig = np.where(y == 1, -3.0, 3.0)
    m_flip = -m_orig
    out = disagreement_hidden_win(
        y, p_orig, p_flip, m_orig, m_flip,
        commit_margin=2.0, n_boot=200, seed=0,
    )
    assert out["committed_disagreement_events"] == 200
    assert out["hidden_win_rate_on_committed_disagreement"] == 1.0
    assert out["hidden_win_ci_lo"] == 1.0


def test_no_disagreement_does_not_fake_a_rescue_signal():
    y = np.array([0, 1] * 20)
    p_orig = np.where(y == 1, 0.9, 0.1)
    p_flip = 1.0 - p_orig
    m_orig = np.where(y == 1, 3.0, -3.0)
    m_flip = -m_orig
    out = disagreement_hidden_win(
        y, p_orig, p_flip, m_orig, m_flip,
        commit_margin=2.0, n_boot=50, seed=0,
    )
    assert out["committed_disagreement_events"] == 0
    assert np.isnan(out["hidden_win_rate_on_committed_disagreement"])
