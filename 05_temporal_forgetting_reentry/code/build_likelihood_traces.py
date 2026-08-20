#!/usr/bin/env python3
"""Build frozen trace rows for G0-B likelihood analysis.

Uses the same traces selected before G0:
- F/S: old-self correct trace;
- N: canonical verified correct trace.
No outcome-dependent trace selection occurs here.
"""
from __future__ import annotations

import argparse
from common import read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.groups)
    pairs = read_jsonl(args.pairs)
    pair_by_f = {str(p["forgotten_problem_id"]): p for p in pairs}
    pair_by_n = {str(p["never_problem_id"]): p for p in pairs}
    out = []

    for r in rows:
        pid = str(r["problem_id"])
        g = r["group"]
        trace = None
        source = None
        pair_id = None
        split = None
        if g == "forgotten" and pid in pair_by_f and r.get("old_correct_trace"):
            trace = r["old_correct_trace"]
            source = "oldself"
            pair_id = pair_by_f[pid]["pair_id"]
            split = pair_by_f[pid]["split"]
        elif g == "never_correct" and pid in pair_by_n and r.get("verified_correct_trace"):
            trace = r["verified_correct_trace"]
            source = "verified_correct"
            pair_id = pair_by_n[pid]["pair_id"]
            split = pair_by_n[pid]["split"]
        elif g == "stable_correct" and r.get("old_correct_trace"):
            trace = r["old_correct_trace"]
            source = "oldself"
            split = "control"
        if trace is None or not r.get("prompt"):
            continue
        out.append(
            {
                "problem_id": pid,
                "pair_id": pair_id,
                "split": split,
                "group": g,
                "source": source,
                "prompt": r["prompt"],
                "gold_answer": r.get("gold_answer"),
                "trace": trace,
            }
        )

    write_jsonl(args.output, out)
    print(f"wrote {len(out)} likelihood traces")


if __name__ == "__main__":
    main()
