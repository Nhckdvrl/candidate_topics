from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare = load("topic17_prepare", "prepare_osf_candidates.py")
preflight = load("topic17_preflight", "build_machine_preflight.py")
resolve = load("topic17_resolve", "resolve_pmc_shortcuts.py")


def test_candidate_selection_is_family_specific_and_deterministic():
    rows = [{
        "field": "biology", "pmc_id": "123", "pubmed_id": "9",
        "title": "paper", "Cit1_SC": "probable",
        "MethCit1": "Immunostaining was performed as described previously.",
        "Cit2_SC": "possible", "MethCit2": "Western blot followed Smith et al.",
    }]
    found = prepare.candidates(rows)
    assert [row["candidate_id"] for row in found] == ["PMC123-cit1", "PMC123-cit2"]
    assert [row["method_family"] for row in found] == ["immunostaining", "western_blot"]


def test_non_pmc_target_is_excluded_from_scientific_denominator():
    row = {
        "candidate_id": "L1", "parent_pmcid": "PMC1",
        "method_family": "immunostaining", "shortcut_sentence": "as described",
        "resolution_status": "target_not_open_pmc",
    }
    assert preflight.build([row]) == []


def test_current_document_detail_is_not_mislabeled_as_lost():
    row = {
        "candidate_id": "L1", "parent_pmcid": "PMC1",
        "method_family": "immunostaining",
        "shortcut_sentence": "as described", "current_evidence": "Fixed in 4% PFA.",
        "resolution_status": "target_method_not_found", "target_reference": {},
    }
    units = {unit["unit_name"]: unit for unit in preflight.build([row])}
    assert units["fixation"]["status"] == "present_in_current"
    assert units["sectioning"]["status"] == "lost_or_unrecoverable"


def test_paragraph_match_recovers_embedded_shortcut():
    root = resolve.ET.fromstring(
        "<article><body><p>Cells were immunostained as described previously [1].</p>"
        "<p>Unrelated material.</p></body></article>"
    )
    node, score = resolve.find_best_paragraph(root, "immunostained as described previously")
    assert "immunostained" in resolve.element_text(node)
    assert score > 0.7


def test_method_retrieval_does_not_require_a_methods_section_title():
    root = resolve.ET.fromstring(
        "<article><body><sec><title>Experimental procedures</title>"
        "<p>Cells were fixed and immunostained with primary antibody.</p>"
        "</sec></body></article>"
    )
    assert "immunostained" in resolve.relevant_methods(root, "immunostaining")
