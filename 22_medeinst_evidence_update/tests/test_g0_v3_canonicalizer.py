from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g0_recanonicalize_v3 import (
    build_mapper_user_text,
    mapper_label_orders,
    parse_mapper_choice,
)


def test_mapper_orders_are_deterministic_distinct_permutations():
    labels = ["A", "B", "C", "D", "E"]
    a1, b1 = mapper_label_orders(labels)
    a2, b2 = mapper_label_orders(labels)
    assert a1 == a2
    assert b1 == b2
    assert sorted(a1) == sorted(labels)
    assert sorted(b1) == sorted(labels)
    assert a1 != b1


def test_parse_mapper_choice_accepts_only_numeric_closed_label_id():
    labels = ["Pneumonia", "Pulmonary embolism", "Stable angina"]
    assert parse_mapper_choice("2", labels) == "Pulmonary embolism"
    assert parse_mapper_choice("2.", labels) == "Pulmonary embolism"
    assert parse_mapper_choice("0", labels) is None
    assert parse_mapper_choice("4", labels) is None
    assert parse_mapper_choice("Label 2", labels) is None
    assert parse_mapper_choice("2 because it matches", labels) is None


def test_mapper_request_contains_only_diagnosis_text_and_closed_labels():
    labels = ["Pneumonia", "Pulmonary embolism"]
    text = build_mapper_user_text("Likely pulmonary embolus.", labels)
    assert "Likely pulmonary embolus." in text
    assert "Pneumonia" in text
    assert "Pulmonary embolism" in text
    assert "Clinical narrative" not in text
    assert "ground_truth" not in text
    assert "case_type" not in text
    assert "control" not in text.lower()
    assert "trap" not in text.lower()


def test_mapper_request_has_explicit_abstention_rule():
    labels = ["Pneumonia", "Pulmonary embolism"]
    text = build_mapper_user_text("Pneumonia versus pulmonary embolism", labels)
    assert "return 0" in text.lower()
    assert "multiple alternative diagnoses" in text.lower()
