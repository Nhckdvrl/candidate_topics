#!/usr/bin/env python3
"""Evaluate locked LLM measurements against the pre-annotated gold audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def precision(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    return {"tp": tp, "fp": fp, "precision": tp / (tp + fp) if tp + fp else None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--measured", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    gold_rows = {r["edge_id"]: r for r in read_jsonl(args.gold)}
    measured_rows = {r["edge_id"]: r for r in read_jsonl(args.measured)}
    if set(gold_rows) != set(measured_rows):
        raise ValueError("gold/measured edge_id sets differ")
    ids = list(gold_rows)
    gold = [gold_rows[x] for x in ids]
    measured = [measured_rows[x] for x in ids]

    same = precision([r["same_core_proposition"] for r in measured],
                     [r["gold_same_core_proposition"] for r in gold])
    direct = precision([r["directly_supported_by_source"] for r in measured],
                       [r["gold_directly_supported_by_source"] for r in gold])
    none = precision([r["evidence_status"] == "NONE" for r in measured],
                     [r["gold_evidence_status"] == "NONE" for r in gold])
    pred_primary = [m["same_core_proposition"] and m["directly_supported_by_source"]
                    and m["evidence_status"] == "NONE" and m["certainty_shift"] != "UNKNOWN"
                    for m in measured]
    gold_primary = [g["gold_same_core_proposition"] and g["gold_directly_supported_by_source"]
                    and g["gold_evidence_status"] == "NONE"
                    and g["gold_certainty_shift"] != "UNKNOWN" for g in gold]
    primary = precision(pred_primary, gold_primary)

    determinate = [(g, m) for g, m in zip(gold, measured)
                   if g["gold_certainty_shift"] != "UNKNOWN"]
    exact = sum(g["gold_certainty_shift"] == m["certainty_shift"] for g, m in determinate)
    non_abstained = [(g, m) for g, m in determinate if m["certainty_shift"] != "UNKNOWN"]
    exact_non_abstained = sum(g["gold_certainty_shift"] == m["certainty_shift"]
                              for g, m in non_abstained)

    confusion: dict[str, Counter] = defaultdict(Counter)
    for g, m in determinate:
        confusion[g["gold_certainty_shift"]][m["certainty_shift"]] += 1

    historical = [(g, m) for g, m in zip(gold, measured)
                  if g.get("gold_origin") == "Greenberg_2009_documented_chain"]
    historical_exact = sum(g["gold_certainty_shift"] == m["certainty_shift"]
                           for g, m in historical)
    historical_up_recovered = sum(g["gold_certainty_shift"] == "UP" and
                                  m["certainty_shift"] == "UP" for g, m in historical)
    historical_gold_up = sum(g["gold_certainty_shift"] == "UP" for g, _ in historical)

    result = {
        "n_gold": len(gold),
        "same_core_proposition": same,
        "directly_supported_by_source": direct,
        "evidence_status_NONE": none,
        "primary_eligibility_precision": primary,
        "certainty_shift": {
            "n_gold_determinate": len(determinate),
            "exact_agreement_including_abstentions": exact / len(determinate),
            "n_model_determinate": len(non_abstained),
            "coverage": len(non_abstained) / len(determinate),
            "agreement_conditional_on_model_determinate": (
                exact_non_abstained / len(non_abstained) if non_abstained else None),
            "confusion_gold_to_measured": {k: dict(v) for k, v in confusion.items()},
        },
        "known_greenberg_case": {
            "n_edges": len(historical),
            "exact": historical_exact,
            "gold_up_edges": historical_gold_up,
            "up_edges_recovered": historical_up_recovered,
            "recovered": historical_up_recovered == historical_gold_up,
        },
        "frozen_gates": {
            "same_core_precision_min": 0.90,
            "none_precision_min": 0.90,
            "certainty_agreement_min": 0.85,
        },
    }
    result["gate_pass"] = bool(
        same["precision"] is not None and same["precision"] >= 0.90
        and none["precision"] is not None and none["precision"] >= 0.90
        and result["certainty_shift"]["exact_agreement_including_abstentions"] >= 0.85
        and result["known_greenberg_case"]["recovered"]
    )
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
