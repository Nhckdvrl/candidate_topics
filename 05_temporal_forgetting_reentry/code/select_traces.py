#!/usr/bin/env python3
"""Freeze old-self, final-wrong, and external-correct traces before G0."""
from __future__ import annotations

import argparse
from collections import defaultdict

from common import read_jsonl, write_jsonl, as_bool, explicit_answer_leak_reasons, split_reasoning_steps


def valid_reasoning_trace(text: str, gold: str | None, allow_final_answer: bool = True) -> bool:
    text = str(text).strip()
    if len(text) < 20 or len(split_reasoning_steps(text)) < 2:
        return False
    if not allow_final_answer and explicit_answer_leak_reasons(text, gold):
        return False
    return True


def choose_shortest(rows: list[dict], correctness: bool, gold: str | None) -> str | None:
    cand = []
    for r in rows:
        try:
            c = as_bool(r["correct"])
        except Exception:
            continue
        if c != correctness:
            continue
        txt = str(r.get("response", "")).strip()
        if valid_reasoning_trace(txt, gold, allow_final_answer=True):
            cand.append(txt)
    if not cand:
        return None
    return min(cand, key=lambda x: (len(x.split()), len(x), x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    samples = read_jsonl(args.samples)
    groups = read_jsonl(args.groups)
    by_poc = defaultdict(list)
    for r in samples:
        by_poc[(str(r["problem_id"]), int(r["checkpoint_order"]))].append(r)

    out = []
    stats = defaultdict(int)
    for g in groups:
        if g["group"] not in {"forgotten", "never_correct", "stable_correct"}:
            continue
        pid = str(g["problem_id"])
        gold = g.get("gold_answer")
        final_order = int(g["final"]["checkpoint_order"])
        row = dict(g)

        if g["group"] in {"forgotten", "stable_correct"}:
            old = g.get("old_checkpoint")
            if not old:
                stats["missing_old_checkpoint"] += 1
                continue
            old_order = int(old["checkpoint_order"])
            old_trace = choose_shortest(by_poc[(pid, old_order)], True, gold)
            if not old_trace:
                stats["missing_old_correct_trace"] += 1
                continue
            row["old_correct_trace"] = old_trace

        if g["group"] == "forgotten":
            wrong_trace = choose_shortest(by_poc[(pid, final_order)], False, gold)
            if wrong_trace:
                row["final_wrong_trace"] = wrong_trace
            else:
                stats["missing_final_wrong_trace"] += 1

        gold_solution = str(g.get("gold_solution", "")).strip()
        if gold_solution:
            if g["group"] == "never_correct":
                row["verified_correct_trace"] = gold_solution
            elif g["group"] == "forgotten":
                row["other_correct_trace"] = gold_solution

        out.append(row)
        stats[f"kept_{g['group']}"] += 1

    write_jsonl(args.output, out)
    print(dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
