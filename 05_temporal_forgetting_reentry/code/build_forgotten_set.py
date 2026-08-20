#!/usr/bin/env python3
"""Classify robust temporal states from repeated checkpoint samples.

Input: one JSONL row per sampled completion with at least
  problem_id, checkpoint, checkpoint_order, correct
and preferably
  prompt, gold_answer, subject, level, response.

Primary labels are defined from empirical pass-rate thresholds, while Wilson
intervals are reported as diagnostics. A problem is not classified unless all
expected checkpoints have >= min_samples observations; this prevents missing
checkpoint data from masquerading as `never_correct`.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from common import read_jsonl, write_jsonl, as_bool, wilson_interval


def state(rate: float, correct_thr: float, wrong_thr: float) -> str:
    if rate >= correct_thr:
        return "C"
    if rate <= wrong_thr:
        return "W"
    return "U"


def compress_cw(states: list[str]) -> list[str]:
    out: list[str] = []
    for s in states:
        if s not in {"C", "W"}:
            continue
        if not out or out[-1] != s:
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--correct-threshold", type=float, default=0.75)
    ap.add_argument("--wrong-threshold", type=float, default=0.125)
    ap.add_argument("--min-samples", type=int, default=16)
    ap.add_argument(
        "--allow-incomplete-checkpoints",
        action="store_true",
        help="Exploratory only. Primary G-1 should leave this off.",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit("empty input")

    expected_orders = sorted({int(r["checkpoint_order"]) for r in rows})
    final_order = max(expected_orders)
    by_pc: dict[tuple[str, int, str], list[bool]] = defaultdict(list)
    meta: dict[str, dict] = {}

    for r in rows:
        pid = str(r["problem_id"])
        order = int(r["checkpoint_order"])
        ckpt = str(r["checkpoint"])
        by_pc[(pid, order, ckpt)].append(as_bool(r["correct"]))
        if pid not in meta:
            keep = ["problem_id", "prompt", "problem", "gold_answer", "gold_solution", "subject", "level"]
            meta[pid] = {k: r[k] for k in keep if k in r}

    rates: dict[str, list[dict]] = defaultdict(list)
    for (pid, order, ckpt), vals in by_pc.items():
        if len(vals) < args.min_samples:
            continue
        k = sum(vals)
        lo, hi = wilson_interval(k, len(vals))
        rate = k / len(vals)
        rates[pid].append(
            {
                "checkpoint": ckpt,
                "checkpoint_order": order,
                "n": len(vals),
                "n_correct": k,
                "pass_rate": rate,
                "wilson95": [lo, hi],
                "state": state(rate, args.correct_threshold, args.wrong_threshold),
            }
        )

    out: list[dict] = []
    excluded_incomplete = 0
    for pid, ckpts in rates.items():
        ckpts = sorted(ckpts, key=lambda x: x["checkpoint_order"])
        got_orders = [x["checkpoint_order"] for x in ckpts]
        if not args.allow_incomplete_checkpoints and got_orders != expected_orders:
            excluded_incomplete += 1
            continue
        if final_order not in got_orders:
            continue

        final = next(x for x in ckpts if x["checkpoint_order"] == final_order)
        earlier = [x for x in ckpts if x["checkpoint_order"] < final_order]
        robust_old_correct = [x for x in earlier if x["state"] == "C"]
        cw = compress_cw([x["state"] for x in ckpts])
        n_flips = max(0, len(cw) - 1)

        group = "other"
        old = None
        if final["state"] == "W" and robust_old_correct:
            group = "forgotten"
            # Latest robust-correct old checkpoint minimizes parameter-time gap.
            old = max(robust_old_correct, key=lambda x: x["checkpoint_order"])
        elif final["state"] == "W" and earlier and all(x["state"] == "W" for x in earlier):
            group = "never_correct"
        elif final["state"] == "C" and robust_old_correct:
            group = "stable_correct"
            old = max(robust_old_correct, key=lambda x: x["checkpoint_order"])
        elif final["state"] == "C" and any(x["state"] == "W" for x in earlier):
            group = "late_acquired_or_recovered"

        out.append(
            {
                **meta.get(pid, {"problem_id": pid}),
                "group": group,
                "final": final,
                "old_checkpoint": old,
                "trajectory_rates": ckpts,
                "robust_state_sequence": "".join(x["state"] for x in ckpts),
                "cw_compressed": "".join(cw),
                "n_robust_flips": n_flips,
            }
        )

    write_jsonl(args.output, out)
    counts = defaultdict(int)
    for r in out:
        counts[r["group"]] += 1
    print(f"expected_orders={expected_orders}")
    print(f"final_checkpoint_order={final_order}")
    print(f"excluded_incomplete={excluded_incomplete}")
    print(f"groups={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
