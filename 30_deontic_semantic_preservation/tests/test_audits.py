import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_simplification_pairs import audit_pairs


def test_condition_without_modality_is_not_eligible(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"original": "If it rains, the office closes.", "simplified": "The office closes in rain."},
        {"original": "Employees must report if absent.", "simplified": "Employees must report."},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = audit_pairs(path)
    assert result["n_deontic_eligible"] == 1
    assert result["structural_flag_counts"]["condition_lost"] == 1


def test_examples_keep_text_and_structured_parses(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"original": "Employees must report.", "simplified": "Employees may report."}),
        encoding="utf-8",
    )
    example = audit_pairs(path)["examples"][0]
    assert example["original_text"] == "Employees must report."
    assert example["original"]["modality"] == "OBLIGATION"
