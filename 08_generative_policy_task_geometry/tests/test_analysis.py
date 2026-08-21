import numpy as np
import pandas as pd

from src.analyze_g0 import matched_entropy_effect


def test_matched_entropy_detects_geometry_risk_gap():
    rng = np.random.default_rng(0)
    n = 200
    ace = rng.normal(size=n)
    tf = np.linspace(0, 1, n)
    risk = 0.1 + 0.6 * tf + rng.normal(scale=0.02, size=n)
    perm = rng.permutation(n)
    tf = tf[perm]
    risk = risk[perm]
    df = pd.DataFrame({"ace": ace, "task_fraction": tf, "risk": risk})
    out = matched_entropy_effect(df, max_z=0.2)
    assert out["n_pairs"] >= 20
    assert out["risk_diff_high_minus_low"] > 0.25
    assert out["task_fraction_diff"] > 0.4
