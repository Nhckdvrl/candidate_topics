#!/usr/bin/env python3
"""Auditable scorer and frozen decision gate for Topic 17.

The code deliberately calls ``silent_divergence`` an *undeclared documentary
conflict*.  Papers alone cannot establish what was physically done; claiming
actual implementation drift would require lab records or replication.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ALLOWED = {
    "same", "present_in_current", "explicitly_modified", "omitted_but_recoverable",
    "lost_or_unrecoverable", "silent_divergence",
}
INHERITANCE_KINDS = {"procedural", "credit_or_history", "uncertain"}
FAILURE_CAUSES = {
    "content_missing", "source_inaccessible", "ambiguous_pointer",
    "undeclared_conflict", "not_applicable",
}

# Frozen G0 bar.  Edit only before annotating/outcome inspection.
MIN_LINEAGES = 20
MIN_METHOD_FAMILIES = 2
MIN_AUDITED_UNITS = 100
MIN_AUDITED_FAILURE_UNITS = 20
MIN_EVIDENCE_COVERAGE = 0.95
MIN_STATUS_AGREEMENT = 0.80
MIN_INHERITANCE_AGREEMENT = 0.90
SUBSTANTIAL_LINEAGE_RATE = 0.25
SURVIVE_WILSON_LOWER = 0.10
MIN_NONACCESS_FAILURE_LINEAGES = 2


@dataclass(frozen=True)
class UnitRow:
    lineage_id: str
    paper_id: str
    hop: int
    method_family: str
    unit_name: str
    critical: bool
    status: str
    inheritance_kind: str = "uncertain"
    evidence_current: str = ""
    evidence_cited: str = ""
    failure_cause: str = "not_applicable"

    @property
    def key(self) -> tuple[str, str, str]:
        return self.lineage_id, self.paper_id, self.unit_name


def load_objects(path: Path) -> list[tuple[int, dict]]:
    objects = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: row must be an object")
            objects.append((lineno, obj))
    if not objects:
        raise ValueError(f"{path}: input contains no rows")
    return objects


def parse_unit(obj: dict, where: str) -> UnitRow:
    required = {
        "lineage_id", "paper_id", "hop", "method_family", "unit_name",
        "critical", "status",
    }
    missing = required - obj.keys()
    if missing:
        raise ValueError(f"{where}: missing {sorted(missing)}")
    if not isinstance(obj["critical"], bool):
        raise ValueError(f"{where}: critical must be Boolean")
    if obj["status"] not in ALLOWED:
        raise ValueError(f"{where}: invalid status {obj['status']!r}")
    raw_hop = obj["hop"]
    try:
        hop = int(raw_hop)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: hop must be an integer") from exc
    if (hop < 0 or isinstance(raw_hop, bool)
            or (isinstance(raw_hop, float) and not raw_hop.is_integer())):
        raise ValueError(f"{where}: hop must be a non-negative integer")
    inheritance = str(obj.get("inheritance_kind", "uncertain"))
    if inheritance not in INHERITANCE_KINDS:
        raise ValueError(f"{where}: invalid inheritance_kind {inheritance!r}")
    cause = str(obj.get("failure_cause", "not_applicable"))
    if cause not in FAILURE_CAUSES:
        raise ValueError(f"{where}: invalid failure_cause {cause!r}")
    if obj["status"] == "silent_divergence" and cause != "undeclared_conflict":
        raise ValueError(f"{where}: silent_divergence requires undeclared_conflict cause")
    if obj["status"] == "lost_or_unrecoverable" and cause not in {
        "content_missing", "source_inaccessible", "ambiguous_pointer"
    }:
        raise ValueError(f"{where}: unrecoverable unit requires a loss cause")
    if obj["status"] not in {"silent_divergence", "lost_or_unrecoverable"} \
            and cause != "not_applicable":
        raise ValueError(f"{where}: non-failure status requires not_applicable cause")
    for field in ("lineage_id", "paper_id", "method_family", "unit_name"):
        if not str(obj[field]).strip():
            raise ValueError(f"{where}: {field} must be non-empty")
    return UnitRow(
        lineage_id=str(obj["lineage_id"]), paper_id=str(obj["paper_id"]), hop=hop,
        method_family=str(obj["method_family"]), unit_name=str(obj["unit_name"]),
        critical=obj["critical"], status=str(obj["status"]),
        inheritance_kind=inheritance,
        evidence_current=str(obj.get("evidence_current", "")).strip(),
        evidence_cited=str(obj.get("evidence_cited", "")).strip(),
        failure_cause=cause,
    )


def load_rows(path: Path) -> list[UnitRow]:
    rows = [parse_unit(obj, f"{path}:{lineno}") for lineno, obj in load_objects(path)]
    seen = set()
    paper_meta: dict[tuple[str, str], tuple[int, str]] = {}
    for row in rows:
        if row.key in seen:
            raise ValueError(f"duplicate adjudicated unit: {row.key}")
        seen.add(row.key)
        node = (row.lineage_id, row.paper_id)
        meta = (row.hop, row.method_family)
        if node in paper_meta and paper_meta[node] != meta:
            raise ValueError(f"inconsistent hop/method family within paper node: {node}")
        paper_meta[node] = meta
    return rows


def classify_paper(rows: list[UnitRow]) -> dict:
    critical = [r for r in rows if r.critical and r.inheritance_kind == "procedural"]
    if not critical:
        return {"n_critical_units": 0, "reconstructible": None,
                "has_undeclared_documentary_conflict": None,
                "has_unrecoverable_unit": None}
    statuses = {r.status for r in critical}
    conflict = "silent_divergence" in statuses
    lost = "lost_or_unrecoverable" in statuses
    return {
        "n_critical_units": len(critical),
        "reconstructible": not (conflict or lost),
        "has_undeclared_documentary_conflict": conflict,
        # Backward-compatible name in machine output, with corrected semantics above.
        "has_silent_divergence": conflict,
        "has_unrecoverable_unit": lost,
    }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def audit_metrics(path: Path, gold: dict[tuple[str, str, str], UnitRow]) -> dict:
    by_key: dict[tuple[str, str, str], list[tuple[str, UnitRow]]] = defaultdict(list)
    for lineno, obj in load_objects(path):
        if "annotator_id" not in obj or not str(obj["annotator_id"]).strip():
            raise ValueError(f"{path}:{lineno}: annotator_id is required")
        if obj.get("annotator_type") != "human":
            raise ValueError(
                f"{path}:{lineno}: annotator_type='human' is required; "
                "model labels are preflight evidence, not the reliability audit"
            )
        row = parse_unit(obj, f"{path}:{lineno}")
        if row.key not in gold:
            raise ValueError(f"{path}:{lineno}: audit unit absent from adjudicated input")
        annotator = str(obj["annotator_id"])
        if any(a == annotator for a, _ in by_key[row.key]):
            raise ValueError(f"duplicate audit annotation: {row.key}, {annotator}")
        by_key[row.key].append((annotator, row))

    complete = {key: vals for key, vals in by_key.items() if len(vals) >= 2}
    comparisons = sum(len(vals) for vals in complete.values())
    if comparisons == 0:
        return {"n_units_double_annotated": 0, "n_audited_failure_units": 0,
                "status_pairwise_agreement": None,
                "inheritance_pairwise_agreement": None,
                "status_agreement_with_adjudication": None,
                "inheritance_agreement_with_adjudication": None,
                "critical_agreement_with_adjudication": None}

    status_ok = inheritance_ok = critical_ok = 0
    pairwise_n = pairwise_status_ok = pairwise_inheritance_ok = 0
    for key, vals in complete.items():
        target = gold[key]
        for _, row in vals:
            status_ok += row.status == target.status
            inheritance_ok += row.inheritance_kind == target.inheritance_kind
            critical_ok += row.critical == target.critical
        for left_index in range(len(vals)):
            for right_index in range(left_index + 1, len(vals)):
                pairwise_n += 1
                left, right = vals[left_index][1], vals[right_index][1]
                pairwise_status_ok += left.status == right.status
                pairwise_inheritance_ok += left.inheritance_kind == right.inheritance_kind
    return {
        "n_units_double_annotated": len(complete),
        "n_audited_failure_units": sum(
            gold[key].status in {"lost_or_unrecoverable", "silent_divergence"}
            for key in complete
        ),
        "n_annotation_comparisons": comparisons,
        "n_pairwise_comparisons": pairwise_n,
        "status_pairwise_agreement": pairwise_status_ok / pairwise_n,
        "inheritance_pairwise_agreement": pairwise_inheritance_ok / pairwise_n,
        "status_agreement_with_adjudication": status_ok / comparisons,
        "inheritance_agreement_with_adjudication": inheritance_ok / comparisons,
        "critical_agreement_with_adjudication": critical_ok / comparisons,
    }


def summarize(rows: list[UnitRow], audit: dict | None = None) -> dict:
    grouped: dict[tuple[str, str], list[UnitRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.lineage_id, row.paper_id)].append(row)
    papers = []
    for (lineage_id, paper_id), group in sorted(grouped.items()):
        papers.append({
            "lineage_id": lineage_id, "paper_id": paper_id,
            "hop": group[0].hop, "method_family": group[0].method_family,
            **classify_paper(group),
        })

    valid_papers = [p for p in papers if p["reconstructible"] is not None]
    by_lineage: dict[str, list[dict]] = defaultdict(list)
    for paper in valid_papers:
        by_lineage[paper["lineage_id"]].append(paper)
    lineages = []
    for lineage_id, nodes in sorted(by_lineage.items()):
        failed = any(not p["reconstructible"] for p in nodes)
        lineages.append({
            "lineage_id": lineage_id,
            "method_families": sorted({p["method_family"] for p in nodes}),
            "reconstructible": not failed,
            "has_undeclared_documentary_conflict": any(
                p["has_undeclared_documentary_conflict"] for p in nodes
            ),
            "has_unrecoverable_unit": any(p["has_unrecoverable_unit"] for p in nodes),
        })
    n_lineages = len(lineages)
    n_failed = sum(not x["reconstructible"] for x in lineages)
    critical = [r for r in rows if r.critical and r.inheritance_kind == "procedural"]
    evidence_complete = [r for r in critical if r.evidence_current and r.evidence_cited]
    nonaccess_ids = {
        r.lineage_id for r in critical
        if r.status in {"silent_divergence", "lost_or_unrecoverable"}
        and r.failure_cause != "source_inaccessible"
    }
    return {
        "schema_version": 2,
        "semantic_scope": (
            "silent_divergence means an undeclared conflict between documents; "
            "it does not prove the procedure physically performed"
        ),
        "unit_rows": len(rows),
        "paper_nodes": len(papers),
        "lineages": lineages,
        "n_lineages": n_lineages,
        "n_failed_lineages": n_failed,
        "failed_lineage_fraction": n_failed / n_lineages if n_lineages else None,
        "failed_lineage_wilson_95ci": wilson_interval(n_failed, n_lineages),
        "method_families": sorted({r.method_family for r in critical}),
        "evidence_coverage": len(evidence_complete) / len(critical) if critical else 0.0,
        "n_nonaccess_failure_lineages": len(nonaccess_ids),
        "unit_status_counts": dict(Counter(r.status for r in critical)),
        "inheritance_kind_counts": dict(Counter(r.inheritance_kind for r in rows)),
        "audit": audit,
        "papers": papers,
    }


def decide(summary: dict) -> dict:
    failures = []
    if summary["n_lineages"] < MIN_LINEAGES:
        failures.append(f"lineages={summary['n_lineages']} < {MIN_LINEAGES}")
    if len(summary["method_families"]) < MIN_METHOD_FAMILIES:
        failures.append(f"method_families={len(summary['method_families'])} < {MIN_METHOD_FAMILIES}")
    if summary["evidence_coverage"] < MIN_EVIDENCE_COVERAGE:
        failures.append(f"evidence_coverage={summary['evidence_coverage']:.3f} < {MIN_EVIDENCE_COVERAGE}")
    audit = summary["audit"]
    if audit is None:
        failures.append("independent double-annotation audit is missing")
    else:
        if audit["n_units_double_annotated"] < MIN_AUDITED_UNITS:
            failures.append(f"audited_units={audit['n_units_double_annotated']} < {MIN_AUDITED_UNITS}")
        if audit.get("n_audited_failure_units", 0) < MIN_AUDITED_FAILURE_UNITS:
            failures.append(f"audited failure units < {MIN_AUDITED_FAILURE_UNITS}")
        if (audit.get("status_pairwise_agreement") or 0) < MIN_STATUS_AGREEMENT:
            failures.append("pairwise status audit agreement below 0.80")
        if (audit.get("inheritance_pairwise_agreement") or 0) < MIN_INHERITANCE_AGREEMENT:
            failures.append("pairwise inheritance audit agreement below 0.90")
    if failures:
        return {"verdict": "INVALID", "reasons": failures}

    rate = summary["failed_lineage_fraction"]
    lower, upper = summary["failed_lineage_wilson_95ci"]
    failed_families = {
        family for lineage in summary["lineages"] if not lineage["reconstructible"]
        for family in lineage["method_families"]
    }
    if (rate >= SUBSTANTIAL_LINEAGE_RATE and lower > SURVIVE_WILSON_LOWER
            and len(failed_families) >= 2
            and summary["n_nonaccess_failure_lineages"] >= MIN_NONACCESS_FAILURE_LINEAGES):
        return {"verdict": "SURVIVE", "reason": (
            "a substantial, cross-family failure rate remains after reliability audit "
            "and is not reducible to inaccessible sources"
        )}
    if upper < SUBSTANTIAL_LINEAGE_RATE:
        return {"verdict": "KILL", "reason": (
            "the sample rules out a 25% lineage-level documentary failure rate"
        )}
    return {"verdict": "INCONCLUSIVE", "reason": (
        "the confidence interval overlaps the frozen operational bar; do not relabel this as support"
    )}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="adjudicated unit JSONL")
    p.add_argument("--audit", type=Path, help="blind annotations; >=2 annotators per audited unit")
    args = p.parse_args()
    rows = load_rows(args.input)
    gold = {row.key: row for row in rows}
    audit = audit_metrics(args.audit, gold) if args.audit else None
    result = summarize(rows, audit)
    result["decision"] = decide(result)
    result["frozen_thresholds"] = {
        "min_lineages": MIN_LINEAGES, "min_method_families": MIN_METHOD_FAMILIES,
        "min_audited_units": MIN_AUDITED_UNITS,
        "min_audited_failure_units": MIN_AUDITED_FAILURE_UNITS,
        "substantial_failed_lineage_rate": SUBSTANTIAL_LINEAGE_RATE,
        "survive_wilson_lower": SURVIVE_WILSON_LOWER,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
