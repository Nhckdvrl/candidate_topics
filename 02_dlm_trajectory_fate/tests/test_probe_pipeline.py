import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from fate_labels import fate_from_correctness
from train_probes import fixed_step_results, pretransition_results


def test_final_controlled_transient_probe_pipeline():
    rng = np.random.default_rng(7)
    n_each = 40
    c = np.vstack(
        [
            np.tile(np.array([0, 0, 1, 0], dtype=bool), (n_each, 1)),
            np.tile(np.array([0, 0, 0, 0], dtype=bool), (n_each, 1)),
        ]
    )
    observed = np.ones_like(c, dtype=bool)
    capture = np.array([0, 1], dtype=np.int16)
    labels = fate_from_correctness(c, capture, observed)

    n = len(c)
    hidden = rng.normal(size=(n, 2, 1, 1, 8)).astype(np.float32)
    y = labels["transient_recovery"][:, 1].astype(float)
    hidden[:, 1, 0, 0, 0] = y * 6.0 + rng.normal(scale=0.2, size=n)

    data = {
        "problem_id": np.arange(n),
        "capture_steps": capture,
        "hidden_indices": np.array([3]),
        "hidden": hidden,
        "entropy": rng.normal(size=(n, 2)).astype(np.float32),
        "selected_prob": rng.uniform(size=(n, 2)).astype(np.float32),
        "clean_maxprob": rng.uniform(size=(n, 2)).astype(np.float32),
        "frac_unmasked": np.tile(np.array([0.0, 0.25]), (n, 1)).astype(np.float32),
        "prompt_tokens": rng.integers(20, 60, size=n),
    }

    rows, _ = fixed_step_results(
        data,
        labels,
        min_class_count=10,
        folds=4,
        n_bootstrap=50,
    )
    target_rows = [
        r
        for r in rows
        if r["task"] == "transient_recovery" and r["step"] == 1
    ]
    assert len(target_rows) == 1
    row = target_rows[0]
    assert row["auc"] > 0.95
    assert row["delta_vs_initial"] > 0.2

    pre = pretransition_results(
        data,
        labels,
        thresholds=[1],
        min_class_count=10,
        folds=4,
        n_bootstrap=20,
    )
    assert any(
        r["task"] == "transient_recovery"
        and r["step"] == 1
        and r["min_lead"] == 1
        for r in pre
    )
