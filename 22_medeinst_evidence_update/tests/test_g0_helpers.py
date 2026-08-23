from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g0_bias_trap_screen import (
    extract_final_diagnosis,
    fixed_random_sample,
    normalize,
    resolve_diagnosis,
    wilson_interval,
)
from g0_pair_locality import pair_metrics


def test_extract_and_resolve_final_diagnosis_only():
    labels = ["Pneumonia", "Pulmonary embolism", "Stable angina"]
    text = "Reasoning mentions Pneumonia, but evidence favors PE.\nFINAL_DIAGNOSIS: Pulmonary embolism"
    final = extract_final_diagnosis(text, strict_marker=True)
    assert final == "Pulmonary embolism"
    assert resolve_diagnosis(final, labels) == "Pulmonary embolism"
    assert extract_final_diagnosis("Pneumonia", strict_marker=True) is None


def test_pair_metrics_detect_local_flip_and_store_diff():
    c = {
        "case_id": "x",
        "narrative": "Sex: F, Age: 50. Fever present. Cough present.",
        "ground_truth": "Pneumonia",
        "age": 50,
        "sex": "F",
    }
    t = {
        "case_id": "x",
        "narrative": "Sex: F, Age: 50. Fever absent. Cough present.",
        "ground_truth": "Pulmonary embolism",
        "age": 50,
        "sex": "F",
    }
    m = pair_metrics(c, t)
    assert m["gt_flips"] and m["same_age"] and m["same_sex"]
    assert 0 < m["changed_token_fraction"] < 0.5
    assert len(m["changes"]) >= 1


def test_normalize():
    assert normalize(" Possible NSTEMI / STEMI. ") == "possible nstemi / stemi"


def test_fixed_random_sample_is_reproducible():
    pairs = [({"case_id": str(i)}, {"case_id": str(i)}) for i in range(20)]
    a = fixed_random_sample(pairs, 7, 123)
    b = fixed_random_sample(pairs, 7, 123)
    assert [x[0]["case_id"] for x in a] == [x[0]["case_id"] for x in b]


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
