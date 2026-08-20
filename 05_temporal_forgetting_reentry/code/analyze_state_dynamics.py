#!/usr/bin/env python3
"""Exploratory state-dynamics audit that cannot rescue a failed primary topic.

If robust forgotten items are rare, this script shows *why*: stable states,
late acquisition, or repeated C/W oscillation. These are descriptive outputs to
surface a potentially different natural phenomenon, not post-hoc evidence for
re-entry.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--examples-per-pattern", type=int, default=5)
    args = ap.parse_args()

    rows = read_jsonl(args.groups)
    group_counts = Counter(r["group"] for r in rows)
    seq_counts = Counter(r.get("robust_state_sequence", "") for r in rows)
    cw_counts = Counter(r.get("cw_compressed", "") for r in rows)
    flip_counts = Counter(int(r.get("n_robust_flips", 0)) for r in rows)

    examples = defaultdict(list)
    for r in rows:
        key = r.get("cw_compressed", "") or "NONE"
        if len(examples[key]) < args.examples_per_pattern:
            examples[key].append(str(r["problem_id"]))

    summary = {
        "n_problems": len(rows),
        "group_counts": dict(group_counts),
        "state_sequence_counts": dict(seq_counts.most_common()),
        "cw_compressed_counts": dict(cw_counts.most_common()),
        "flip_counts": {str(k): v for k, v in sorted(flip_counts.items())},
        "example_problem_ids": dict(examples),
        "interpretation_guardrail": (
            "Exploratory only. A new phenomenon found here must be re-registered "
            "with its own validation gate; it cannot rescue failed re-entry hypotheses."
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
