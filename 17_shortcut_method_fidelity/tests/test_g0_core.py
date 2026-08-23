from pathlib import Path
import importlib.util
import json
import sys
import pytest


MODULE_PATH = Path(__file__).parents[1] / "g0_core.py"
SPEC = importlib.util.spec_from_file_location("topic17_g0", MODULE_PATH)
g0 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = g0
SPEC.loader.exec_module(g0)


def rows(failed: int):
    result = []
    for lineage in range(20):
        family = f"family-{lineage % 2}"
        for unit in range(5):
            is_failure = lineage < failed and unit == 0
            result.append(g0.UnitRow(
                lineage_id=f"L{lineage}", paper_id=f"P{lineage}", hop=1,
                method_family=family, unit_name=f"u{unit}", critical=True,
                status="silent_divergence" if is_failure else "same",
                inheritance_kind="procedural", evidence_current="current quote",
                evidence_cited="cited quote",
                failure_cause="undeclared_conflict" if is_failure else "not_applicable",
            ))
    return result


def good_audit():
    return {
        "n_units_double_annotated": 100,
        "n_audited_failure_units": 20,
        "n_annotation_comparisons": 200,
        "n_pairwise_comparisons": 100,
        "status_pairwise_agreement": .9,
        "inheritance_pairwise_agreement": .95,
        "status_agreement_with_adjudication": .9,
        "inheritance_agreement_with_adjudication": .95,
        "critical_agreement_with_adjudication": .95,
    }


def test_survival_and_kill_regions_are_executable():
    strong = g0.summarize(rows(6), good_audit())
    assert g0.decide(strong)["verdict"] == "SURVIVE"

    null = g0.summarize(rows(0), good_audit())
    assert g0.decide(null)["verdict"] == "KILL"


def test_missing_audit_is_invalid_not_a_scientific_null():
    summary = g0.summarize(rows(6), None)
    assert g0.decide(summary)["verdict"] == "INVALID"


def test_document_conflict_does_not_claim_physical_implementation():
    summary = g0.summarize(rows(6), good_audit())
    assert "does not prove" in summary["semantic_scope"]
    assert summary["papers"][0]["has_undeclared_documentary_conflict"]


def test_failure_status_requires_a_matching_cause():
    obj = {
        "lineage_id": "L", "paper_id": "P", "hop": 1,
        "method_family": "f", "unit_name": "u", "critical": True,
        "status": "silent_divergence", "inheritance_kind": "procedural",
        "failure_cause": "not_applicable",
    }
    with pytest.raises(ValueError, match="undeclared_conflict"):
        g0.parse_unit(obj, "test")


def test_formal_audit_rejects_machine_annotators(tmp_path):
    gold_rows = rows(6)
    gold = {row.key: row for row in gold_rows}
    row = gold_rows[0]
    obj = {
        **row.__dict__,
        "annotator_id": "model-a",
        "annotator_type": "model",
    }
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="human"):
        g0.audit_metrics(audit, gold)
