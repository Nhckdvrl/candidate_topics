#!/usr/bin/env python3
"""Fast integrity audit before expensive inference/analysis."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from common import read_jsonl, as_bool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    rows = read_jsonl(args.input)
    required = {"problem_id", "checkpoint", "checkpoint_order", "response", "correct"}
    missing = Counter()
    by = defaultdict(int)
    orders = set()
    for r in rows:
        for k in required:
            if k not in r:
                missing[k] += 1
        if "correct" in r:
            as_bool(r["correct"])
        if "problem_id" in r and "checkpoint_order" in r:
            by[(str(r["problem_id"]), int(r["checkpoint_order"]))] += 1
            orders.add(int(r["checkpoint_order"]))
    counts = Counter(by.values())
    print(f"rows={len(rows)} problems={len({k[0] for k in by})} checkpoint_orders={sorted(orders)}")
    print(f"samples_per_problem_checkpoint={dict(sorted(counts.items()))}")
    print(f"missing_fields={dict(missing)}")
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
