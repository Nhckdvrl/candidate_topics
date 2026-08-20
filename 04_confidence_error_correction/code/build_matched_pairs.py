#!/usr/bin/env python3
"""Build accessibility-matched high/low semantic-commitment pairs for G-1v2.

G-1v2 intentionally does NOT gate items on top-wrong argmax stability.
Argmax stability is mechanically related to the treatment: diffuse wrong
beliefs are expected to have unstable top-1 identities. Position susceptibility
is measured separately and audited before G0.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def quantile(xs: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def valid_item(r: dict, require_k: int | None) -> bool:
    if "p_correct" not in r or "wrong_concentration" not in r:
        return False
    if require_k is not None and int(r.get("choice_count", len(r["choices"]))) != require_k:
        return False
    probs = [float(x) for x in r["semantic_probs"]]
    if not all(np.isfinite(probs)):
        return False
    answer = int(r["answer"])
    return max(range(len(probs)), key=lambda i: probs[i]) != answer


def pair_cost(
    h: dict,
    l: dict,
    p_caliper: float,
    q_ratio: float,
    a_ratio: float,
    susceptibility_caliper: float | None,
) -> float | None:
    if h.get("category") != l.get("category"):
        return None
    dp = abs(float(h["p_correct"]) - float(l["p_correct"]))
    if dp > p_caliper:
        return None

    hq = max(int(h.get("question_token_count", 1)), 1)
    lq = max(int(l.get("question_token_count", 1)), 1)
    if max(hq, lq) / min(hq, lq) > q_ratio:
        return None

    ha = max(int(h.get("correct_answer_token_count", 1)), 1)
    la = max(int(l.get("correct_answer_token_count", 1)), 1)
    if max(ha, la) / min(ha, la) > a_ratio:
        return None

    ds = 0.0
    if susceptibility_caliper is not None:
        hs = float(h.get("position_susceptibility_js", 0.0))
        ls = float(l.get("position_susceptibility_js", 0.0))
        ds = abs(hs - ls)
        if ds > susceptibility_caliper:
            return None

    length_penalty = abs(math.log(hq / lq)) + 0.5 * abs(math.log(ha / la))
    return dp / max(p_caliper, 1e-9) + 0.15 * length_penalty + 0.25 * ds


def optimal_match_category(
    high: list[dict],
    low: list[dict],
    p_caliper: float,
    q_ratio: float,
    a_ratio: float,
    susceptibility_caliper: float | None,
) -> list[tuple[dict, dict, float]]:
    if not high or not low:
        return []
    BIG = 1e6
    cost = np.full((len(high), len(low)), BIG, dtype=float)
    for i, h in enumerate(high):
        for j, l in enumerate(low):
            c = pair_cost(
                h, l, p_caliper, q_ratio, a_ratio, susceptibility_caliper
            )
            if c is not None:
                cost[i, j] = c
    rows, cols = linear_sum_assignment(cost)
    out = []
    for i, j in zip(rows, cols):
        if cost[i, j] < BIG:
            out.append((high[i], low[j], float(cost[i, j])))
    return out


def stratified_split(pairs: list[dict], discovery_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_cat[str(p["category"])].append(p)
    for cat_pairs in by_cat.values():
        rng.shuffle(cat_pairs)
        n_disc = int(round(len(cat_pairs) * discovery_fraction))
        if len(cat_pairs) >= 2:
            n_disc = min(max(n_disc, 1), len(cat_pairs) - 1)
        for i, p in enumerate(cat_pairs):
            p["split"] = "discovery" if i < n_disc else "confirmation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pairs-output", required=True)
    ap.add_argument("--eligible-output", required=True)
    ap.add_argument("--report-output", required=True)
    ap.add_argument("--require-k", type=int, default=10)
    ap.add_argument("--p-caliper", type=float, default=0.02)
    ap.add_argument("--question-length-ratio", type=float, default=1.35)
    ap.add_argument("--answer-length-ratio", type=float, default=1.50)
    ap.add_argument("--susceptibility-caliper", type=float, default=None)
    ap.add_argument("--high-quantile", type=float, default=0.70)
    ap.add_argument("--low-quantile", type=float, default=0.30)
    ap.add_argument("--discovery-fraction", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    all_rows = read_jsonl(args.input)
    rows = [r for r in all_rows if valid_item(r, args.require_k)]
    if not rows:
        raise SystemExit("No eligible initially-wrong items")

    vals = [float(r["wrong_concentration"]) for r in rows]
    lo_cut = quantile(vals, args.low_quantile)
    hi_cut = quantile(vals, args.high_quantile)
    low = [r for r in rows if float(r["wrong_concentration"]) <= lo_cut]
    high = [r for r in rows if float(r["wrong_concentration"]) >= hi_cut]

    by_cat_high = defaultdict(list)
    by_cat_low = defaultdict(list)
    for r in high:
        by_cat_high[str(r.get("category", "unknown"))].append(r)
    for r in low:
        by_cat_low[str(r.get("category", "unknown"))].append(r)

    matched = []
    for cat in sorted(set(by_cat_high) | set(by_cat_low)):
        matched.extend(
            optimal_match_category(
                by_cat_high[cat],
                by_cat_low[cat],
                args.p_caliper,
                args.question_length_ratio,
                args.answer_length_ratio,
                args.susceptibility_caliper,
            )
        )

    pairs = []
    for idx, (h, l, cost) in enumerate(matched):
        pairs.append(
            {
                "pair_id": f"pair_{idx:05d}",
                "category": str(h.get("category", "unknown")),
                "high": h,
                "low": l,
                "match_cost": cost,
                "p_correct_abs_diff": abs(float(h["p_correct"]) - float(l["p_correct"])),
                "commitment_diff": float(h["wrong_concentration"]) - float(l["wrong_concentration"]),
                "susceptibility_abs_diff": abs(
                    float(h.get("position_susceptibility_js", 0.0))
                    - float(l.get("position_susceptibility_js", 0.0))
                ),
            }
        )
    stratified_split(pairs, args.discovery_fraction, args.seed)

    write_jsonl(args.eligible_output, rows)
    write_jsonl(args.pairs_output, pairs)

    diffs = [p["p_correct_abs_diff"] for p in pairs]
    cdiff = [p["commitment_diff"] for p in pairs]
    sdiff = [p["susceptibility_abs_diff"] for p in pairs]
    split_counts = Counter(p["split"] for p in pairs)
    cats = Counter(p["category"] for p in pairs)

    stabilities = [float(r.get("top_wrong_stability", float("nan"))) for r in rows]
    suscept = [float(r.get("position_susceptibility_js", float("nan"))) for r in rows]
    report = {
        "measurement_version": "g1v2_logmean",
        "n_scored_input": len(all_rows),
        "n_eligible_wrong": len(rows),
        "wrong_concentration_low_cut": lo_cut,
        "wrong_concentration_high_cut": hi_cut,
        "n_low_pool": len(low),
        "n_high_pool": len(high),
        "n_pairs": len(pairs),
        "mean_abs_p_correct_diff": float(np.mean(diffs)) if diffs else None,
        "median_abs_p_correct_diff": float(np.median(diffs)) if diffs else None,
        "mean_commitment_separation": float(np.mean(cdiff)) if cdiff else None,
        "median_commitment_separation": float(np.median(cdiff)) if cdiff else None,
        "mean_susceptibility_abs_diff": float(np.mean(sdiff)) if sdiff else None,
        "eligible_median_top_wrong_stability_diagnostic_only": (
            float(np.nanmedian(stabilities)) if stabilities else None
        ),
        "eligible_median_position_susceptibility_js": (
            float(np.nanmedian(suscept)) if suscept else None
        ),
        "split_counts": dict(split_counts),
        "category_pair_counts": dict(cats),
        "same_category_fraction": 1.0 if pairs else None,
        "settings": vars(args),
    }
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
