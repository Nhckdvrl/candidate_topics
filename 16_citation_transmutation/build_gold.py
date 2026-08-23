#!/usr/bin/env python3
"""Build the frozen Topic 16 measurement-audit set.

The set is intentionally diagnostic rather than prevalence-representative.  It
contains controlled adversarial citation vignettes spanning all locked labels,
plus a documented Greenberg-style historical chain.  Gold labels are embedded
under ``gold_*`` keys and are never shown to the annotation model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAINS = [
    ("exercise-il6", "Moderate exercise", "may lower", "circulating IL-6", "in sedentary adults"),
    ("drug-bp", "Drug A", "may reduce", "systolic blood pressure", "in adults with hypertension"),
    ("protein-pathway", "Protein K", "may activate", "pathway M", "in human hepatocytes"),
    ("diet-glucose", "A high-fibre diet", "may improve", "glycaemic control", "in adults with type 2 diabetes"),
    ("exposure-asthma", "Early-life smoke exposure", "may increase", "asthma risk", "in children"),
    ("gene-survival", "High GENE1 expression", "may predict", "shorter survival", "in ovarian cancer"),
    ("sleep-memory", "Sleep after training", "may improve", "memory consolidation", "in healthy adults"),
    ("microbe-colitis", "Bacterium B", "may aggravate", "colitis severity", "in mice"),
]


def statement(subject: str, relation: str, obj: str, scope: str) -> str:
    return f"{subject} {relation} {obj} {scope}."


def add_case(rows: list[dict], *, domain: tuple[str, str, str, str, str], variant: str,
             source_claim: str, citing_claim: str, source_context: str,
             citing_context: str, bundle: list[dict], audit_complete: bool,
             same: bool, support: bool, evidence: str, shift: str) -> None:
    key, *_ = domain
    rows.append({
        "edge_id": f"gold-{key}-{variant}",
        "claim_id": f"gold-claim-{key}",
        "source_paper_id": f"CONTROLLED-SOURCE-{key}",
        "citing_paper_id": f"CONTROLLED-CITING-{key}-{variant}",
        "source_claim": source_claim,
        "citing_claim": citing_claim,
        "source_context": source_context,
        "citing_context": citing_context,
        "evidence_audit_complete": audit_complete,
        "evidence_bundle": bundle,
        "gold_same_core_proposition": same,
        "gold_directly_supported_by_source": support,
        "gold_evidence_audit_complete": audit_complete,
        "gold_evidence_status": evidence,
        "gold_certainty_shift": shift,
        "gold_origin": "controlled_adversarial_vignette",
    })


def controlled_rows() -> list[dict]:
    rows: list[dict] = []
    for i, domain in enumerate(DOMAINS):
        _, subject, hedged_relation, obj, scope = domain
        base_relation = hedged_relation.replace("may ", "")
        plain_relation = {
            "lower": "lowers", "reduce": "reduces", "activate": "activates",
            "improve": "improves", "increase": "increases", "predict": "predicts",
            "aggravate": "aggravates",
        }[base_relation]
        src = statement(subject, hedged_relation, obj, scope)
        fact = statement(subject, plain_relation, obj, scope)
        source_context = f"Exploratory evidence was compatible with the possibility that {src[0].lower() + src[1:]} Confirmation is required."

        add_case(
            rows, domain=domain, variant="none-up", source_claim=src,
            citing_claim=fact, source_context=source_context,
            citing_context=f"Background: {fact} [single source]. This proposition is not an outcome of the present study.",
            bundle=[{"audit_scope": "full citing paper, tables, supplement, and reference-linked passages",
                     "finding": "No own result, additional primary study, review, or meta-analysis supports this proposition; only the named source is used."}],
            audit_complete=True, same=True, support=True, evidence="NONE", shift="UP")

        add_case(
            rows, domain=domain, variant="own-primary", source_claim=src,
            citing_claim=fact, source_context=source_context,
            citing_context=f"The present preregistered experiment tested the proposition. {fact}",
            bundle=[{"kind": "new_result_in_citing_article", "finding": "A newly reported controlled experiment in this article directly supports the proposition."}],
            audit_complete=True, same=True, support=True, evidence="OWN_PRIMARY", shift="UP")

        changed_scope = scope.replace("adults", "adolescents").replace("children", "adults").replace("mice", "humans").replace("human hepatocytes", "mouse hepatocytes")
        if changed_scope == scope:
            changed_scope = scope + " receiving intensive treatment"
        add_case(
            rows, domain=domain, variant="population-nearmiss", source_claim=src,
            citing_claim=statement(subject, hedged_relation, obj, changed_scope),
            source_context=source_context,
            citing_context="The cited article is invoked for this population-specific statement.",
            bundle=[{"audit_scope": "complete", "finding": "No evidence beyond the cited source was found."}],
            audit_complete=True, same=False, support=False, evidence="NONE", shift="UNKNOWN")

        inverse = plain_relation
        inverse_obj = obj
        replacements = {"lowers": "raises", "reduces": "increases", "activates": "inhibits", "improves": "worsens",
                        "increases": "decreases", "aggravates": "alleviates"}
        for old, new in replacements.items():
            if old in inverse:
                inverse = inverse.replace(old, new, 1)
                break
        if domain[0] == "gene-survival":
            inverse = plain_relation
            inverse_obj = "longer survival"
        add_case(
            rows, domain=domain, variant="direction-nearmiss", source_claim=src,
            citing_claim=statement(subject, inverse, inverse_obj, scope), source_context=source_context,
            citing_context="The citing statement reverses the reported direction.",
            bundle=[{"audit_scope": "complete", "finding": "No other new support."}],
            audit_complete=True, same=False, support=False, evidence="NONE", shift="UNKNOWN")

        add_case(
            rows, domain=domain, variant="external-primary", source_claim=src,
            citing_claim=statement(subject, "probably " + plain_relation, obj, scope),
            source_context=source_context,
            citing_context="The claim is supported by the cited source and two later replications.",
            bundle=[{"kind": "other_study_records", "finding": "Two independent later cohort/experimental studies directly support the same proposition."}],
            audit_complete=True, same=True, support=True, evidence="EXTERNAL_PRIMARY", shift="UP")

        add_case(
            rows, domain=domain, variant="synthesis-same", source_claim=fact,
            citing_claim=fact, source_context=f"The source concludes: {fact}",
            citing_context="A systematic review restates the proposition after quantitative synthesis.",
            bundle=[{"kind": "pooled_literature_result", "finding": "A new systematic review/meta-analysis pools the source with other studies."}],
            audit_complete=True, same=True, support=True, evidence="SYNTHESIS", shift="SAME")

        add_case(
            rows, domain=domain, variant="incomplete", source_claim=src,
            citing_claim=fact, source_context=source_context,
            citing_context="Only the isolated citation sentence was retrieved.",
            bundle=[], audit_complete=False, same=True, support=True,
            evidence="UNKNOWN", shift="UP")

        down = statement(subject, "might " + base_relation, obj, scope)
        add_case(
            rows, domain=domain, variant="none-down", source_claim=fact,
            citing_claim=down, source_context=f"The source states without qualification: {fact}",
            citing_context=f"The later paper says only that {down[0].lower() + down[1:]}",
            bundle=[{"audit_scope": "complete", "finding": "No new supporting evidence beyond the source."}],
            audit_complete=True, same=True, support=True, evidence="NONE", shift="DOWN")

    return rows


def historical_rows() -> list[dict]:
    base = {
        "claim_id": "greenberg-app-precedes-ibm",
        "evidence_audit_complete": True,
        "gold_same_core_proposition": True,
        "gold_directly_supported_by_source": True,
        "gold_evidence_audit_complete": True,
        "gold_evidence_status": "NONE",
        "gold_origin": "Greenberg_2009_documented_chain",
        "gold_source_note": "Greenberg 2009 supplementary material and Clark et al. 2014 Fig. 20",
    }
    pairs = [
        ("greenberg-S37-S96", "GREENBERG2009:S37", "GREENBERG2009:S96",
         "One possibility is that beta-amyloid precursor protein is accumulated first in inclusion-body myositis.",
         "Increased beta-amyloid precursor protein mRNA and epitopes appear to precede other abnormalities in inclusion-body myositis muscle fibers.", "UP"),
        ("greenberg-S96-S129", "GREENBERG2009:S96", "GREENBERG2009:S129",
         "Increased beta-amyloid precursor protein mRNA and epitopes appear to precede other abnormalities in inclusion-body myositis muscle fibers.",
         "We have previously demonstrated that accumulation of beta-amyloid precursor protein epitopes precedes other abnormalities in inclusion-body myositis muscle fibers.", "UP"),
        ("greenberg-S80-S97", "GREENBERG2009:S80", "GREENBERG2009:S97",
         "Those muscle fibers may represent early changes of inclusion-body myositis.",
         "Beta-amyloid precursor protein accumulation is thought to precede other changes in inclusion-body myositis.", "UP"),
        ("greenberg-S129-S2", "GREENBERG2009:S129", "GREENBERG2009:S2",
         "We have previously demonstrated that accumulation of beta-amyloid precursor protein epitopes precedes other abnormalities in inclusion-body myositis muscle fibers.",
         "The accumulation of amyloid precursor protein and its fragments is often stated to precede other abnormalities in inclusion-body myositis muscle fibers.", "DOWN"),
    ]
    out = []
    for edge_id, src_id, cite_id, src, cite, shift in pairs:
        row = dict(base)
        row.update({
            "edge_id": edge_id, "source_paper_id": src_id, "citing_paper_id": cite_id,
            "source_claim": src, "citing_claim": cite,
            "source_context": src,
            "citing_context": cite + " The citation lineage is documented in Greenberg's claim-specific audit.",
            "evidence_bundle": [{"audit": "Greenberg 2009 complete claim-specific citation-network audit",
                                 "finding": "No new evidence supporting the temporal precedence proposition entered this citation step."}],
            "gold_certainty_shift": shift,
        })
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("raw_gold_edges.jsonl"))
    args = p.parse_args()
    rows = controlled_rows() + historical_rows()
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "n_rows": len(rows),
                      "controlled": len(controlled_rows()), "historical": len(historical_rows())}))


if __name__ == "__main__":
    main()
