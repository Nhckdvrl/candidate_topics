import numpy as np
import pandas as pd

from src.panel import choose_identifiable_pair, pair_stats
from src.relative_probe import SharedLinearProbe, paired_relative_metrics


def _panel(a, b, c=None):
    rows = []
    vals = {"2k": a, "9k": b}
    if c is not None:
        vals["3k"] = c
    for checkpoint, ys in vals.items():
        for seed, y in enumerate(ys):
            rows.append({"task": "t", "seed": seed, "checkpoint": checkpoint, "success": y})
    return pd.DataFrame(rows)


def test_pair_stats_counts_bidirectional_crossover():
    df = _panel([1, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 1])
    s = pair_stats(df, "2k", "9k")
    assert s.n_a_wins == 2
    assert s.n_b_wins == 2
    assert s.n_disagree == 4
    assert s.bidirectional_support == 2


def test_pair_selection_prefers_bidirectional_support():
    # 2k vs 9k has more total disagreement but nearly all one-way.
    # 2k vs 3k has fewer disagreements but much cleaner two-way support.
    a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    c = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    chosen = choose_identifiable_pair(_panel(a, b, c))
    assert chosen is not None
    assert {chosen.checkpoint_a, chosen.checkpoint_b} == {"2k", "3k"}
    assert chosen.bidirectional_support >= 2


def test_shared_relative_probe_detects_checkpoint_specific_signal():
    rng = np.random.default_rng(1)
    n = 300
    # Half the states favor A and half favor B. Generic state difficulty is shared,
    # while a checkpoint-specific component changes sign with the winner.
    winner_a = rng.integers(0, 2, size=n)
    generic = rng.normal(size=(n, 2))
    x, y, sid, cp = [], [], [], []
    for i in range(n):
        direction = 1.0 if winner_a[i] else -1.0
        xa = np.r_[generic[i], direction + 0.15 * rng.normal()]
        xb = np.r_[generic[i], -direction + 0.15 * rng.normal()]
        x.extend([xa, xb])
        y.extend([int(winner_a[i]), int(not winner_a[i])])
        sid.extend([i, i])
        cp.extend(["A", "B"])
    x = np.asarray(x)
    y = np.asarray(y)
    probe = SharedLinearProbe().fit(x, y)
    scores = probe.score(x)
    m = paired_relative_metrics(np.asarray(sid), np.asarray(cp), y, scores, "A", "B")
    assert m["relative_auc"] > 0.95
    assert m["zero_threshold_balanced_accuracy"] > 0.9


def test_generic_state_only_features_cannot_predict_relative_winner():
    # Identical features for both checkpoints in each physical state imply q_A-q_B=0.
    n = 40
    winner_a = np.asarray([0, 1] * (n // 2))
    state = np.arange(n)
    state_features = np.stack([np.sin(state), np.cos(state)], axis=1)
    x, y, sid, cp = [], [], [], []
    for i in range(n):
        x.extend([state_features[i], state_features[i]])
        y.extend([int(winner_a[i]), int(not winner_a[i])])
        sid.extend([i, i])
        cp.extend(["A", "B"])
    probe = SharedLinearProbe().fit(np.asarray(x), np.asarray(y))
    scores = probe.score(np.asarray(x))
    m = paired_relative_metrics(np.asarray(sid), np.asarray(cp), np.asarray(y), scores, "A", "B")
    assert m["relative_auc"] == 0.5
    assert m["zero_threshold_balanced_accuracy"] == 0.5
