#!/usr/bin/env python3
"""Post-G0 descriptive audit for Topic 20.

This is NOT a new scientific gate and must not be used to tune G0. It audits two
properties noticed only after the frozen G0 had already passed:

1. exact duplicate numerical/format pairs in the released seed-0 test set;
2. whether generation errors disproportionately select the scientific-notation
   operand.

The second observation is exploratory on seed-0 test and therefore requires a
fresh-seed confirmation before it can support a G1 mechanism claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

SCI_RE = re.compile(r"[×x*]\s*10\^")


def parse_value(s: str) -> float:
    return float(eval(str(s).replace("×", "*").replace("x", "*").replace("^", "**").replace(",", "")))


def is_sci(s: str) -> bool:
    return SCI_RE.search(str(s)) is not None


def first_answer_value(completion: str):
    number_re = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[×x*]\s*10\^?-?\d+)?")
    m = number_re.search(str(completion))
    if not m:
        return None
    try:
        return parse_value(m.group(0))
    except Exception:
        return None


def same_num(x, y) -> bool:
    return x is not None and y is not None and math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, default=Path("20_numeracy_representation_access/artifacts/g0/test_records.jsonl"))
    p.add_argument("--out", type=Path, default=Path("20_numeracy_representation_access/artifacts/g0/post_g0_audit.json"))
    args = p.parse_args()

    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]

    # Exact displayed-pair duplicates preserve both operand order and notation.
    pair_counts = Counter((r["a"], r["b"]) for r in rows)
    hard_pair_counts = Counter((r["a"], r["b"]) for r in rows if r["hard"])
    crit_pair_counts = Counter((r["a"], r["b"]) for r in rows if r["hard"] and r["critical"])

    hard = [r for r in rows if r["hard"]]
    crit = [r for r in hard if r["critical"]]
    gen_errors = [r for r in hard if not r["generation_correct"]]

    def classify_choice(r):
        pred = first_answer_value(r.get("completion", ""))
        a = parse_value(r["a"])
        b = parse_value(r["b"])
        if same_num(pred, a) and not same_num(pred, b):
            side = "a"
        elif same_num(pred, b) and not same_num(pred, a):
            side = "b"
        else:
            return "neither_or_ambiguous", None
        chosen_is_sci = is_sci(r[side])
        return side, chosen_is_sci

    def subset_stats(subset):
        choices = [classify_choice(r) for r in subset]
        operand_choices = [x for x in choices if x[0] in {"a", "b"}]
        sci_choices = sum(bool(x[1]) for x in operand_choices)
        return {
            "n": len(subset),
            "n_exact_operand_choice": len(operand_choices),
            "n_scientific_operand_choice": sci_choices,
            "scientific_choice_rate_among_exact_operand_choices": (
                sci_choices / len(operand_choices) if operand_choices else None
            ),
            "digit_counts": dict(sorted(Counter(int(r["digit"]) for r in subset).items())),
            "gold_position_counts": dict(sorted(Counter(r["gold_position"] for r in subset).items())),
        }

    duplicate_crit_instances = sum(c - 1 for c in crit_pair_counts.values() if c > 1)
    unique_crit = len(crit_pair_counts)

    summary = {
        "note": "Descriptive seed-0 audit only; notation pattern is exploratory and must be confirmed on a fresh seed.",
        "n_test": len(rows),
        "n_unique_displayed_pairs": len(pair_counts),
        "n_duplicate_test_instances": sum(c - 1 for c in pair_counts.values() if c > 1),
        "n_hard": len(hard),
        "n_unique_hard_displayed_pairs": len(hard_pair_counts),
        "n_duplicate_hard_instances": sum(c - 1 for c in hard_pair_counts.values() if c > 1),
        "n_hard_critical": len(crit),
        "n_unique_hard_critical_displayed_pairs": unique_crit,
        "n_duplicate_hard_critical_instances": duplicate_crit_instances,
        "hard_critical": subset_stats(crit),
        "hard_generation_errors": subset_stats(gen_errors),
        "duplicate_hard_critical_pairs": [
            {"a": a, "b": b, "count": c}
            for (a, b), c in sorted(crit_pair_counts.items()) if c > 1
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
