import json
from pathlib import Path

import pytest

from g0_core import (
    Edge,
    bootstrap_claim_balanced_ci,
    classify_rows,
    claim_means,
    load_edges,
    summarize_group,
)


def edge(**kwargs):
    base = dict(
        edge_id="e1",
        claim_id="c1",
        source_paper_id="p1",
        citing_paper_id="p2",
        source_claim="X may cause Y",
        citing_claim="X causes Y",
        same_core_proposition=True,
        directly_supported_by_source=True,
        evidence_audit_complete=True,
        evidence_status="NONE",
        certainty_shift="UP",
    )
    base.update(kwargs)
    return Edge(**base)


def test_primary_filter_requires_no_new_support_and_complete_audit():
    rows = [
        edge(edge_id="a"),
        edge(edge_id="b", evidence_status="OWN_PRIMARY"),
        edge(edge_id="c", evidence_audit_complete=False, evidence_status="UNKNOWN"),
        edge(edge_id="d", same_core_proposition=False),
        edge(edge_id="e", directly_supported_by_source=False),
    ]
    groups = classify_rows(rows)
    assert [e.edge_id for e in groups["primary_no_new_support"]] == ["a"]
    assert [e.edge_id for e in groups["secondary_with_new_support"]] == ["b"]


def test_claim_balanced_statistic_does_not_let_large_claim_dominate():
    rows = [
        edge(edge_id=f"big-{i}", claim_id="big", certainty_shift="UP")
        for i in range(9)
    ] + [edge(edge_id="small", claim_id="small", certainty_shift="DOWN")]
    summary = summarize_group(rows, n_boot=200, seed=7)
    assert summary["edge_net_upward"] == pytest.approx(0.8)
    assert summary["claim_balanced_net_upward"] == pytest.approx(0.0)
    assert summary["n_claims"] == 2


def test_claim_means_encode_up_same_down():
    rows = [
        edge(edge_id="a", claim_id="c", certainty_shift="UP"),
        edge(edge_id="b", claim_id="c", certainty_shift="SAME"),
        edge(edge_id="c", claim_id="c", certainty_shift="DOWN"),
    ]
    assert claim_means(rows)["c"] == pytest.approx(0.0)


def test_bootstrap_is_deterministic():
    rows = [
        edge(edge_id="a", claim_id="c1", certainty_shift="UP"),
        edge(edge_id="b", claim_id="c2", certainty_shift="DOWN"),
    ]
    assert bootstrap_claim_balanced_ci(rows, 100, 123) == bootstrap_claim_balanced_ci(
        rows, 100, 123
    )


def test_load_rejects_duplicate_edge_id(tmp_path: Path):
    obj = {
        "edge_id": "dup",
        "claim_id": "c",
        "source_paper_id": "p1",
        "citing_paper_id": "p2",
        "source_claim": "X may cause Y",
        "citing_claim": "X causes Y",
        "same_core_proposition": True,
        "directly_supported_by_source": True,
        "evidence_audit_complete": True,
        "evidence_status": "NONE",
        "certainty_shift": "UP",
    }
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps(obj) + "\n" + json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="duplicate edge_id"):
        load_edges(path)


def test_none_requires_complete_evidence_audit(tmp_path: Path):
    obj = {
        "edge_id": "e",
        "claim_id": "c",
        "source_paper_id": "p1",
        "citing_paper_id": "p2",
        "source_claim": "X may cause Y",
        "citing_claim": "X causes Y",
        "same_core_proposition": True,
        "directly_supported_by_source": True,
        "evidence_audit_complete": False,
        "evidence_status": "NONE",
        "certainty_shift": "UP",
    }
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="requires evidence_audit_complete=true"):
        load_edges(path)


def test_unknown_certainty_is_reported_but_not_scored():
    rows = [edge(edge_id="a"), edge(edge_id="b", certainty_shift="UNKNOWN")]
    summary = summarize_group(rows, n_boot=50, seed=1)
    assert summary["n_edges"] == 2
    assert summary["n_unknown_certainty"] == 1
    assert summary["edge_net_upward"] == pytest.approx(1.0)
