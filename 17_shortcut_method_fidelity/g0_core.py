#!/usr/bin/env python3
"""Minimal G0 scorer for Topic 17: shortcut-citation method fidelity.

Input is one JSON object per critical method unit. Extraction is intentionally
outside this script so that the scientific measurement remains auditable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ALLOWED = {
    "same",
    "explicitly_modified",
    "omitted_but_recoverable",
    "lost_or_unrecoverable",
    "silent_divergence",
}


@dataclass(frozen=True)
class UnitRow:
    lineage_id: str
    paper_id: str
    hop: int
    method_family: str
    unit_name: str
    critical: bool
    status: str


def load_rows(path: Path) -> list[UnitRow]:
    rows: list[UnitRow] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            required = [
                "lineage_id",
                "paper_id",
                "hop",
                "method_family",
                "unit_name",
                "critical",
                "status",
            ]
            missing = [k for k in required if k not in obj]
            if missing:
                raise ValueError(f"line {lineno}: missing {missing}")
            if not isinstance(obj["critical"], bool):
                raise ValueError(f"line {lineno}: critical must be Boolean")
            status = str(obj["status"])
            if status not in ALLOWED:
                raise ValueError(
                    f"line {lineno}: invalid status {status!r}; allowed={sorted(ALLOWED)}"
                )
            hop = int(obj["hop"])
            if hop < 0:
                raise ValueError(f"line {lineno}: hop must be >= 0")
            rows.append(
                UnitRow(
                    lineage_id=str(obj["lineage_id"]),
                    paper_id=str(obj["paper_id"]),
                    hop=hop,
                    method_family=str(obj["method_family"]),
                    unit_name=str(obj["unit_name"]),
                    critical=obj["critical"],
                    status=status,
                )
            )
    if not rows:
        raise ValueError("input contains no rows")
    return rows


def classify_paper(rows: list[UnitRow]) -> dict:
    critical = [r for r in rows if r.critical]
    if not critical:
        return {
            "n_critical_units": 0,
            "reconstructible": None,
            "has_silent_divergence": None,
            "has_unrecoverable_unit": None,
        }

    statuses = {r.status for r in critical}
    has_silent = "silent_divergence" in statuses
    has_unrecoverable = "lost_or_unrecoverable" in statuses
    reconstructible = not (has_silent or has_unrecoverable)
    return {
        "n_critical_units": len(critical),
        "reconstructible": reconstructible,
        "has_silent_divergence": has_silent,
        "has_unrecoverable_unit": has_unrecoverable,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()

    rows = load_rows(args.input)
    grouped: dict[tuple[str, str], list[UnitRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.lineage_id, row.paper_id)].append(row)

    papers = []
    for (lineage_id, paper_id), group in sorted(grouped.items()):
        meta = group[0]
        outcome = classify_paper(group)
        papers.append(
            {
                "lineage_id": lineage_id,
                "paper_id": paper_id,
                "hop": meta.hop,
                "method_family": meta.method_family,
                **outcome,
            }
        )

    valid = [p for p in papers if p["reconstructible"] is not None]
    n = len(valid)
    summary = {
        "unit_rows": len(rows),
        "paper_nodes": len(papers),
        "paper_nodes_with_critical_units": n,
        "reconstructible_fraction": (
            sum(bool(p["reconstructible"]) for p in valid) / n if n else None
        ),
        "silent_divergence_fraction": (
            sum(bool(p["has_silent_divergence"]) for p in valid) / n if n else None
        ),
        "unrecoverable_fraction": (
            sum(bool(p["has_unrecoverable_unit"]) for p in valid) / n if n else None
        ),
        "unit_status_counts": dict(Counter(r.status for r in rows if r.critical)),
        "papers": papers,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
