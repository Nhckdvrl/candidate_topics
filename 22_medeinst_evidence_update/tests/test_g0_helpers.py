from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g0_bias_trap_screen import (
    extract_prediction,
    fixed_random_sample,
    normalize,
    resolve_diagnosis,
    stable_pair_seed,
    wilson_interval,
)
from g0_pair_locality import pair_metrics


def test_extract_explicit_marker():
    labels = ["Pneumonia", "Pulmonary embolism", "Stable angina"]
    pred, candidate, method = extract_prediction(
        "The final answer is below.\nFINAL_DIAGNOSIS: Pulmonary embolism",
        labels,
    )
    assert pred == "Pulmonary embolism"
    assert candidate == "Pulmonary embolism"
    assert method == "explicit_marker"


def test_extract_without_custom_marker():
    labels = ["Pneumonia", "Pulmonary embolism", "Stable angina"]
    pred, candidate, method = extract_prediction(
        "Given the evidence, the most likely diagnosis is Pulmonary embolism.",
        labels,
    )
    assert pred == "Pulmonary embolism"
    assert method in {"diagnosis_is", "resolved_final_line", "last_canonical_mention"}


def test_last_final_line_beats_earlier_final_content_mention():
    labels = ["Pneumonia", "Pulmonary embolism", "Stable angina"]
    pred, _, _ = extract_prediction(
        "Pneumonia is less likely after reviewing the new evidence.\nPulmonary embolism",
        labels,
    )
    assert pred == "Pulmonary embolism"


def test_unresolved_final_is_not_guessed():
    labels = ["Pneumonia", "Pulmonary embolism", "Stable angina"]
    pred, candidate, method = extract_prediction("The presentation is concerning.", labels)
    assert pred is None
    assert candidate is None
    assert method == "unresolved_final"


def test_resolve_diagnosis_prefers_longest_canonical_hit():
    labels = ["Angina", "Stable angina"]
    assert resolve_diagnosis("Final: Stable angina", labels) == "Stable angina"


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


def test_pair_seed_is_stable_and_case_specific():
    assert stable_pair_seed(123, "case_1") == stable_pair_seed(123, "case_1")
    assert stable_pair_seed(123, "case_1") != stable_pair_seed(123, "case_2")


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
