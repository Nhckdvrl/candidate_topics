#!/usr/bin/env python3
"""Convert resolved PMC lineages into a machine-only unit audit preflight.

This does not create adjudicated scientific labels.  Its output is deliberately
run without ``--audit``, so the formal G0 gate must remain INVALID until humans
independently verify the evidence and statuses.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ONTOLOGY = {
    "immunostaining": {
        "fixation": r"fix|formalin|paraformaldehyde|PFA",
        "sectioning": r"section|cryostat|paraffin|microtome",
        "permeabilization_or_blocking": r"permeabili|block|serum|triton",
        "primary_antibody_incubation": r"primary antibod|incubat.*antibod|antibod.*incubat",
        "secondary_detection_and_imaging": r"secondary antibod|fluorescen|confocal|microscop|visuali",
    },
    "western_blot": {
        "protein_extraction": r"extract|lysate|lysis|homogen",
        "loading_and_electrophoresis": r"SDS.PAGE|electrophores|protein.*[µu]?g|loaded",
        "transfer_and_blocking": r"transfer|membrane|PVDF|nitrocellulose|block",
        "primary_antibody_incubation": r"primary antibod|incubat.*antibod|antibod.*incubat",
        "secondary_detection": r"secondary antibod|chemilum|ECL|detect|visuali",
    },
}


def evidence_for(row: dict) -> str:
    target = row.get("target_reference", {})
    citation = target.get("citation", "")
    return citation or "No cited resource could be resolved from the parent paragraph"


def matching_line(text: str, pattern: str) -> str:
    return next(
        (line for line in text.splitlines() if re.search(pattern, line, re.I)),
        text[:1000],
    )


def build(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        # Absence from PMC is a retrieval-platform limitation, not evidence that
        # the cited resource is inaccessible or methodologically inadequate.
        # Exclude it from the scientific denominator until publisher/DOI search.
        if row["resolution_status"] in {"parent_sentence_not_found", "target_not_open_pmc"}:
            continue
        family = row["method_family"]
        cited = row.get("cited_method_evidence", "")
        current = row.get("current_evidence", row.get("shortcut_sentence", ""))
        for unit, pattern in ONTOLOGY[family].items():
            current_match = re.search(pattern, current, re.I)
            cited_match = re.search(pattern, cited, re.I)
            if current_match:
                # Reconstruction asks whether the current document plus its
                # lineage supplies the unit.  A unit explicitly present in the
                # current paper cannot be called lost merely because the target
                # extraction did not find it.
                status, cause = "present_in_current", "not_applicable"
                cited_evidence = (
                    matching_line(cited, pattern) if cited_match else
                    "No matching unit recovered from cited methods; current paper supplies it: "
                    + evidence_for(row)
                )
            elif cited_match:
                status, cause = "omitted_but_recoverable", "not_applicable"
                cited_evidence = matching_line(cited, pattern)
            elif row["resolution_status"] == "target_method_not_found":
                status, cause = "lost_or_unrecoverable", "content_missing"
                cited_evidence = (
                    "Open target searched; no family-relevant method paragraph recovered: "
                    + evidence_for(row)
                )
            elif row["resolution_status"] == "resolved":
                status, cause = "lost_or_unrecoverable", "content_missing"
                cited_evidence = (
                    f"Unit pattern absent from current and recovered cited methods: {evidence_for(row)}"
                )
            else:
                continue
            output.append({
                "lineage_id": row["candidate_id"],
                "paper_id": row["parent_pmcid"],
                "hop": 1,
                "method_family": family,
                "unit_name": unit,
                "critical": True,
                "status": status,
                "inheritance_kind": "procedural",
                "evidence_current": current,
                "evidence_cited": cited_evidence,
                "failure_cause": cause,
                "annotation_status": "machine_preflight_not_adjudicated",
                "source_resolution_status": row["resolution_status"],
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.lineages.open(encoding="utf-8") if line.strip()]
    units = build(rows)
    with args.output.open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "units": len(units),
        "lineages": len({row["lineage_id"] for row in units}),
        "warning": "machine preflight only; formal gate must remain INVALID without human audit",
    }, indent=2))


if __name__ == "__main__":
    main()
