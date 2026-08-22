from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "qwen3_1p7b_table13_math.csv"


def test_table13_transcription_shape_and_extremes():
    df = pd.read_csv(TABLE)
    layers = df[df["setting"].str.startswith("Layer")].copy()
    layers["layer"] = layers["layer"].astype(int)
    assert sorted(layers["layer"].tolist()) == list(range(28))

    by_layer = layers.set_index("layer")
    assert np.isclose(by_layer.loc[10, "c_math"], 1.14)
    assert np.isclose(by_layer.loc[24, "c_math"], 0.28)


def test_reported_c_is_consistent_with_published_rounded_scores():
    # Table cells are rounded, so recomputing C from rounded math_avg cannot be
    # bit-identical to the paper's C column. It should nevertheless agree within
    # the rounding envelope; a larger mismatch catches transcription errors.
    df = pd.read_csv(TABLE)
    base = float(df.loc[df["setting"] == "Base", "math_avg"].iloc[0])
    full = float(df.loc[df["setting"] == "Full", "math_avg"].iloc[0])
    layers = df[df["setting"].str.startswith("Layer")].copy()
    recalculated = (layers["math_avg"].to_numpy(float) - base) / (full - base)
    reported = layers["c_math"].to_numpy(float)
    assert np.max(np.abs(recalculated - reported)) < 0.03
