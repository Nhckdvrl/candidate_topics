#!/usr/bin/env python3
"""Locked G0 scorer for Topic 16: citation transmutation.

The scorer consumes *measured* citation edges. It never infers claim identity,
evidence provenance, or certainty itself. Those measurements may come from a
human-validated LLM pipeline, but the statistical estimand is frozen here.

Primary scientific object:
  same core proposition + source really supports it + complete evidence audit
  + no new supporting evidence + determinate certainty comparison.

Primary statistic:
  claim-balanced net upward shift = mean_claim(P(UP) - P(DOWN)).

Claims, not edges, are the independent resampling unit.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EVIDENCE_STATUSES = {
    "NONE",
    "OWN_PRIMARY",
    "EXTERNAL_PRIMARY",
    "SYNTHESIS",
    "UNKNOWN",
}
CERTAINTY_SHIFTS = {"UP", "SAME", "DOWN", "UNKNOWN"}
SHIFT_SCORE = {"UP": 1.0, "SAME": 0.0, "DOWN": -1.0}


@dataclass(frozen=True)
class Edge:
    edge_id: str
    claim_id: str
    source_paper_id: str
    citing_paper_id: str
    source_claim: str
    citing_claim: str
    same_core_proposition: bool
    directly_supported_by_source: bool
    evidence_audit_complete: bool
    evidence_status: str
    certainty_shift: str

    @property
    def is_primary_eligible(self) -> bool:
        return (
            self.same_core_proposition
            and self.directly_supported_by_source
            and self.evidence_audit_complete
            and self.evidence_status == "NONE"
            and self.certainty_shift != "UNKNOWN"
        )


def _as_bool(value, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a JSON Boolean, got {value!r}")


def _as_nonempty_str(value, field: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field} must be non-empty")
    return value


def _as_choice(value, field: str, choices: set[str]) -> str:
    value = _as_nonempty_str(value, field).upper()
    if value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}, got {value!r}")
    return value


def load_edges(path: Path) -> list[Edge]:
    rows: list[Edge] = []
    seen_edge_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON: {exc}") from exc
            try:
                edge = Edge(
                    edge_id=_as_nonempty_str(obj["edge_id"], "edge_id"),
                    claim_id=_as_nonempty_str(obj["claim_id"], "claim_id"),
                    source_paper_id=_as_nonempty_str(
                        obj["source_paper_id"], "source_paper_id"
                    ),
                    citing_paper_id=_as_nonempty_str(
                        obj["citing_paper_id"], "citing_paper_id"
                    ),
                    source_claim=_as_nonempty_str(obj["source_claim"], "source_claim"),
                    citing_claim=_as_nonempty_str(obj["citing_claim"], "citing_claim"),
                    same_core_proposition=_as_bool(
                        obj["same_core_proposition"], "same_core_proposition"
                    ),
                    directly_supported_by_source=_as_bool(
                        obj["directly_supported_by_source"],
                        "directly_supported_by_source",
                    ),
                    evidence_audit_complete=_as_bool(
                        obj["evidence_audit_complete"], "evidence_audit_complete"
                    ),
                    evidence_status=_as_choice(
                        obj["evidence_status"], "evidence_status", EVIDENCE_STATUSES
                    ),
                    certainty_shift=_as_choice(
                        obj["certainty_shift"], "certainty_shift", CERTAINTY_SHIFTS
                    ),
                )
            except KeyError as exc:
                raise ValueError(f"line {lineno}: missing field {exc}") from exc

            if edge.edge_id in seen_edge_ids:
                raise ValueError(f"line {lineno}: duplicate edge_id {edge.edge_id!r}")
            seen_edge_ids.add(edge.edge_id)

            if edge.evidence_status == "NONE" and not edge.evidence_audit_complete:
                raise ValueError(
                    f"line {lineno}: evidence_status=NONE requires "
                    "evidence_audit_complete=true"
                )
            rows.append(edge)

    if not rows:
        raise ValueError("input contains no rows")
    return rows


def claim_means(edges: Iterable[Edge]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for edge in edges:
        if edge.certainty_shift == "UNKNOWN":
            continue
        grouped[edge.claim_id].append(SHIFT_SCORE[edge.certainty_shift])
    return {claim_id: statistics.fmean(scores) for claim_id, scores in grouped.items()}


def bootstrap_claim_balanced_ci(
    edges: Iterable[Edge], n_boot: int, seed: int
) -> tuple[float, float]:
    if n_boot <= 0:
        raise ValueError("n_boot must be > 0")
    per_claim = claim_means(edges)
    if not per_claim:
        return float("nan"), float("nan")
    values = list(per_claim.values())
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return lo, hi


def summarize_group(edges: Iterable[Edge], n_boot: int, seed: int) -> dict:
    edges = list(edges)
    determinate = [e for e in edges if e.certainty_shift != "UNKNOWN"]
    if not determinate:
        return {
            "n_edges": len(edges),
            "n_claims": 0,
            "n_unknown_certainty": sum(e.certainty_shift == "UNKNOWN" for e in edges),
            "counts": {"UP": 0, "SAME": 0, "DOWN": 0},
            "edge_net_upward": None,
            "claim_balanced_net_upward": None,
            "cluster_bootstrap_95ci": None,
        }

    counts = {
        shift: sum(e.certainty_shift == shift for e in determinate)
        for shift in ("UP", "SAME", "DOWN")
    }
    edge_scores = [SHIFT_SCORE[e.certainty_shift] for e in determinate]
    per_claim = claim_means(determinate)
    lo, hi = bootstrap_claim_balanced_ci(determinate, n_boot=n_boot, seed=seed)
    return {
        "n_edges": len(edges),
        "n_claims": len(per_claim),
        "n_unknown_certainty": sum(e.certainty_shift == "UNKNOWN" for e in edges),
        "counts": counts,
        "upward_fraction": counts["UP"] / len(determinate),
        "same_fraction": counts["SAME"] / len(determinate),
        "downward_fraction": counts["DOWN"] / len(determinate),
        "edge_net_upward": statistics.fmean(edge_scores),
        "claim_balanced_net_upward": statistics.fmean(per_claim.values()),
        "cluster_bootstrap_95ci": [lo, hi],
    }


def classify_rows(edges: list[Edge]) -> dict[str, list[Edge]]:
    same_supported = [
        e for e in edges if e.same_core_proposition and e.directly_supported_by_source
    ]
    complete = [e for e in same_supported if e.evidence_audit_complete]
    primary = [
        e
        for e in complete
        if e.evidence_status == "NONE" and e.certainty_shift != "UNKNOWN"
    ]
    with_new_support = [
        e
        for e in complete
        if e.evidence_status in {"OWN_PRIMARY", "EXTERNAL_PRIMARY", "SYNTHESIS"}
    ]
    return {
        "same_core_supported": same_supported,
        "complete_evidence_audit": complete,
        "primary_no_new_support": primary,
        "secondary_with_new_support": with_new_support,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args()
    if args.bootstrap <= 0:
        p.error("--bootstrap must be > 0")

    edges = load_edges(args.input)
    groups = classify_rows(edges)

    result = {
        "all_rows": len(edges),
        "same_core_supported_rows": len(groups["same_core_supported"]),
        "complete_evidence_audit_rows": len(groups["complete_evidence_audit"]),
        "excluded_incomplete_or_unknown_evidence": sum(
            (not e.evidence_audit_complete) or e.evidence_status == "UNKNOWN"
            for e in groups["same_core_supported"]
        ),
        "excluded_unknown_certainty_no_new_support": sum(
            e.evidence_audit_complete
            and e.evidence_status == "NONE"
            and e.certainty_shift == "UNKNOWN"
            for e in groups["same_core_supported"]
        ),
        "primary_no_new_support": summarize_group(
            groups["primary_no_new_support"], n_boot=args.bootstrap, seed=args.seed
        ),
        "secondary_with_new_support": summarize_group(
            groups["secondary_with_new_support"],
            n_boot=args.bootstrap,
            seed=args.seed + 1,
        ),
        "estimand": "claim_balanced_mean(P(UP)-P(DOWN)) on no-new-support edges",
        "resampling_unit": "claim_id",
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
