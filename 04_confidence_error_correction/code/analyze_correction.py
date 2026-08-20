#!/usr/bin/env python3
"""Locked analysis for Topic 04 correction dynamics."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(xs: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def bootstrap_mean(xs: list[float], n_boot: int, seed: int, ci: float = 0.95) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(len(xs))] for _ in xs]
        boots.append(sum(sample) / len(sample))
    alpha = (1 - ci) / 2
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "lo": percentile(boots, alpha),
        "hi": percentile(boots, 1 - alpha),
    }


def summarize_item_curves(rows: list[dict]) -> dict[tuple, dict]:
    by_item = defaultdict(list)
    for r in rows:
        key = (str(r["seed"]), r["pair_id"], r["id"], r["group"])
        by_item[key].append(r)

    out = {}
    for key, rs in by_item.items():
        rs = sorted(rs, key=lambda x: int(x["cycle"]))
        by_cycle = {int(r["cycle"]): r for r in rs}
        if 0 not in by_cycle:
            continue
        base = by_cycle[0]
        post_cycles = sorted(c for c in by_cycle if c > 0)
        if not post_cycles:
            continue

        ps = [float(by_cycle[c]["p_correct"]) for c in post_cycles]
        gains = [p - float(base["p_correct"]) for p in ps]

        t_top1 = None
        for c in post_cycles:
            if not int(by_cycle[c]["top1_correct"]):
                continue
            nxt = c + 1
            if nxt not in by_cycle or int(by_cycle[nxt]["top1_correct"]):
                t_top1 = c
                break

        early_cycles = [c for c in post_cycles if c <= 2]
        max_cycle = max(post_cycles)
        late_cycles = [c for c in post_cycles if c >= max_cycle - 2]
        last = by_cycle[max_cycle]

        out[key] = {
            "seed": str(key[0]),
            "pair_id": key[1],
            "id": key[2],
            "group": key[3],
            "category": base.get("category", "unknown"),
            "base_p_correct": float(base["base_p_correct"]),
            "wrong_concentration": float(base["wrong_concentration"]),
            "question_token_count": int(base.get("question_token_count", 0)),
            "correct_answer_token_count": int(base.get("correct_answer_token_count", 0)),
            "auc_correct": float(np.mean(ps)),
            "auc_gain": float(np.mean(gains)),
            "immediate_gain": float(by_cycle[post_cycles[0]]["p_correct"]) - float(base["p_correct"]),
            "early_gain": float(np.mean([by_cycle[c]["p_correct"] for c in early_cycles])) - float(base["p_correct"]),
            "late_correct": float(np.mean([by_cycle[c]["p_correct"] for c in late_cycles])),
            "t_top1": t_top1,
            "old_error_suppression": float(base["p_old_wrong"]) - float(last["p_old_wrong"]),
        }
    return out


def pair_differences(items: dict, metric: str) -> list[dict]:
    cells = defaultdict(dict)
    for v in items.values():
        cells[(v["seed"], v["pair_id"])][v["group"]] = v
    out = []
    for (seed, pair_id), g in cells.items():
        if "high" not in g or "low" not in g:
            continue
        hv, lv = g["high"].get(metric), g["low"].get(metric)
        if hv is None or lv is None:
            continue
        out.append(
            {
                "seed": seed,
                "pair_id": pair_id,
                "diff": float(hv) - float(lv),
                "category": g["high"]["category"],
            }
        )
    return out


def aggregate_pair_diffs_over_seeds(diffs: list[dict]) -> list[float]:
    by_pair = defaultdict(list)
    for d in diffs:
        by_pair[d["pair_id"]].append(float(d["diff"]))
    return [sum(v) / len(v) for v in by_pair.values()]


def per_seed_reports(diffs: list[dict], n_boot: int) -> dict:
    by_seed = defaultdict(list)
    for d in diffs:
        by_seed[d["seed"]].append(float(d["diff"]))
    return {
        seed: bootstrap_mean(xs, n_boot, seed=1000 + i)
        for i, (seed, xs) in enumerate(sorted(by_seed.items()))
    }


def continuous_regression(items: dict) -> dict:
    """OLS on item curves averaged over seeds; pair-cluster robust SE."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    by_id = defaultdict(list)
    for v in items.values():
        by_id[(v["pair_id"], v["id"], v["group"])].append(v)
    rows = []
    for (pair_id, item_id, group), vals in by_id.items():
        row = dict(vals[0])
        row["auc_gain"] = float(np.mean([x["auc_gain"] for x in vals]))
        row["pair_id"] = pair_id
        row["id"] = item_id
        row["group"] = group
        rows.append(row)
    if len(rows) < 20:
        return {"available": False, "error": "too few rows"}

    df = pd.DataFrame(rows)
    formula = (
        "auc_gain ~ wrong_concentration + base_p_correct "
        "+ wrong_concentration:base_p_correct "
        "+ question_token_count + correct_answer_token_count + C(category)"
    )
    fit = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["pair_id"]}
    )
    keys = [
        "wrong_concentration",
        "base_p_correct",
        "wrong_concentration:base_p_correct",
    ]
    return {
        "available": True,
        "n": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "coefficients": {
            k: {
                "coef": float(fit.params[k]),
                "se": float(fit.bse[k]),
                "p": float(fit.pvalues[k]),
                "ci95": [float(x) for x in fit.conf_int().loc[k].tolist()],
            }
            for k in keys
            if k in fit.params
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--split", choices=["discovery", "confirmation"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--equivalence-margin", type=float, default=0.02)
    ap.add_argument("--screen-effect", type=float, default=0.02)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.input) if r["split"] == args.split]
    items = summarize_item_curves(rows)

    metrics = [
        "auc_correct",
        "auc_gain",
        "immediate_gain",
        "early_gain",
        "late_correct",
        "old_error_suppression",
        "t_top1",
    ]
    report = {
        "split": args.split,
        "n_evaluation_rows": len(rows),
        "n_item_seed_curves": len(items),
        "metrics": {},
    }
    for metric in metrics:
        diffs = pair_differences(items, metric)
        pooled_pairs = aggregate_pair_diffs_over_seeds(diffs)
        report["metrics"][metric] = {
            "high_minus_low_pooled_over_seeds": bootstrap_mean(
                pooled_pairs, args.bootstrap, seed=20260821
            ),
            "per_seed": per_seed_reports(diffs, args.bootstrap),
        }

    primary = report["metrics"]["auc_gain"]["high_minus_low_pooled_over_seeds"]
    if primary["n"]:
        diffs = aggregate_pair_diffs_over_seeds(pair_differences(items, "auc_gain"))
        eq = bootstrap_mean(diffs, args.bootstrap, seed=20260822, ci=0.90)
        m = args.equivalence_margin
        eq["margin"] = m
        eq["inside_margin"] = bool(
            eq["lo"] is not None and eq["lo"] > -m and eq["hi"] < m
        )
        report["equivalence_screen_auc_gain"] = eq

        seed_means = [
            x["mean"]
            for x in report["metrics"]["auc_gain"]["per_seed"].values()
            if x["mean"] is not None
        ]
        sign = np.sign(primary["mean"])
        same_sign = sum(np.sign(x) == sign for x in seed_means)
        report["directional_screen"] = {
            "effect_threshold": args.screen_effect,
            "seed_same_sign_count": int(same_sign),
            "seed_count": len(seed_means),
            "ci_excludes_zero": bool(primary["lo"] > 0 or primary["hi"] < 0),
            "abs_mean_meets_threshold": bool(abs(primary["mean"]) >= args.screen_effect),
            "passes_discovery_screen": bool(
                args.split == "discovery"
                and len(seed_means) > 0
                and same_sign >= min(2, len(seed_means))
                and (primary["lo"] > 0 or primary["hi"] < 0)
                and abs(primary["mean"]) >= args.screen_effect
            ),
        }

    cycle_cells = defaultdict(list)
    cycle_top1 = defaultdict(list)
    for r in rows:
        cycle_cells[int(r["cycle"])].append(float(r["p_correct"]))
        cycle_top1[int(r["cycle"])].append(int(r["top1_correct"]))
    report["aggregate_mean_p_correct_by_cycle"] = {
        str(c): float(np.mean(v)) for c, v in sorted(cycle_cells.items())
    }
    report["aggregate_top1_rate_by_cycle"] = {
        str(c): float(np.mean(v)) for c, v in sorted(cycle_top1.items())
    }
    report["continuous_regression"] = continuous_regression(items)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
