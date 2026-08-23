#!/usr/bin/env python3
"""Merge frozen retrieval candidates with explicit pre-measurement audit decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STATUSES = {"NONE", "OWN_PRIMARY", "EXTERNAL_PRIMARY", "SYNTHESIS", "UNKNOWN"}


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    candidates = rows(args.candidates)
    decisions = {str(x["edge_id"]): x for x in rows(args.decisions)}
    if set(decisions) != {str(x["edge_id"]) for x in candidates}:
        raise ValueError("decisions must contain exactly one row for every candidate")

    with args.output.open("w", encoding="utf-8") as out:
        for row in candidates:
            decision = decisions[row["edge_id"]]
            complete = decision.get("evidence_audit_complete")
            status = str(decision.get("human_evidence_status", "UNKNOWN")).upper()
            notes = str(decision.get("audit_notes", "")).strip()
            if not isinstance(complete, bool) or status not in STATUSES or not notes:
                raise ValueError(f"invalid audit decision for {row['edge_id']}")
            if not complete and status != "UNKNOWN":
                raise ValueError(f"incomplete audit must be UNKNOWN: {row['edge_id']}")
            row["evidence_audit_complete"] = complete
            # Kept for post-measurement validation, but llm_annotate.py never places
            # extra human_* fields in the model prompt.
            row["human_evidence_status"] = status
            row["human_audit_notes"] = notes
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
