"""Render the Topic 23 G0 panel tables from raw records.

Usage:  python summarize.py records/*.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from g0_core import POLICY_CONDITIONS, Condition, GateConfig, analyze_records

ORDER = [c.value for c in POLICY_CONDITIONS] + [Condition.ORACLE_RIGHT_DISABLED.value]


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def table(rows: list[dict]) -> str:
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["dr_level"]][r["condition"]].append(bool(r["success"]))
    present = [c for c in ORDER if any(c in by[L] for L in by)]
    out = ["| level | " + " | ".join(present) + " |",
           "| --- | " + " | ".join("---:" for _ in present) + " |"]
    for L in sorted(by):
        cells = []
        for c in present:
            v = by[L][c]
            cells.append(f"{sum(v)}/{len(v)}" if v else "—")
        out.append(f"| {L} | " + " | ".join(cells) + " |")
    cells = []
    for c in present:
        v = [s for L in by for s in by[L][c]]
        cells.append(f"**{sum(v)}/{len(v)}**" if v else "—")
    out.append("| **all** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def kinematics(rows: list[dict]) -> str:
    by = collections.defaultdict(list)
    stale = 0
    for r in rows:
        if r.get("right_arm_excursion_rad") is None:
            continue
        if r.get("route_attribution") is None:
            stale += 1
        by[r["condition"]].append(r)
    if stale:
        print(f"> WARNING: {stale} rows predate peak-motion route attribution "
              f"and contribute 'none' to the attribution column.\n")
    out = ["| condition | right-arm excursion (rad) | clamp leak (rad) | base path (m) | route attribution |",
           "| --- | ---: | ---: | ---: | --- |"]
    for c in ORDER:
        rs = by.get(c)
        if not rs:
            continue
        ex = np.mean([r["right_arm_excursion_rad"] for r in rs])
        # The leak column only means something where a clamp is active. On
        # `canonical` the same number is just distance from the neutral pose.
        lk = (
            np.mean([r.get("right_arm_clamp_leak_rad") or 0.0 for r in rs])
            if c not in ("canonical",) else None
        )
        bp = [r.get("base_path_m") for r in rs if r.get("base_path_m") is not None]
        att = collections.Counter()
        for r in rs:
            att[tuple(r.get("route_attribution") or [])] += 1
        top = ", ".join(
            f"{'+'.join(k) or 'none'} ×{v}" for k, v in att.most_common(3)
        )
        lk_s = "n/a" if lk is None else f"{lk:.3f}"
        bp_s = f"{np.mean(bp):.2f}" if bp else "—"
        out.append(f"| {c} | {ex:.3f} | {lk_s} | {bp_s} | {top} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    rows = load(args.records)
    if args.title:
        print(f"## {args.title}\n")
    print(f"{len(rows)} rows\n")
    print("### Success by condition\n")
    print(table(rows))
    print("\n### Kinematics and route\n")
    print(kinematics(rows))
    print("\n### Frozen gate verdict\n```json")
    rep = analyze_records(rows, GateConfig())
    keep = ["n_matched_configs", "success_rate", "oracle_success_rate",
            "arm_program_cost", "canonical_right_route_rate", "max_clamp_leak_rad",
            "substitution_events", "paired_right_disabled_minus_full_hold", "verdict"]
    print(json.dumps({k: rep[k] for k in keep if k in rep}, indent=2))
    print("```")


if __name__ == "__main__":
    main()
