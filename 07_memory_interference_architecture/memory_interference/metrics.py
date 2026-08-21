from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def accuracy(rows: Sequence[Mapping]) -> float:
    valid = [r for r in rows if not r.get("skipped", False)]
    if not valid:
        return float("nan")
    return float(np.mean([bool(r["correct"]) for r in valid]))


def interference_asymmetry(ri_accuracy: float, pi_accuracy: float) -> float:
    """I = Error_PI - Error_RI = Accuracy_RI - Accuracy_PI.

    Positive means stronger proactive interference / primacy protection.
    Negative means stronger retroactive interference / recency overwrite.
    """
    return ri_accuracy - pi_accuracy


def log_auc(level_to_accuracy: Mapping[int, float]) -> float:
    """Trapezoidal AUC over log10(U+1), where U is later-update count.

    This is source-inspired but is not numerically the paper's RIES/PIES because
    that paper defines its interference-level symbol differently.
    """
    pairs = sorted((int(k), float(v)) for k, v in level_to_accuracy.items() if int(k) >= 1)
    if len(pairs) < 2:
        return float("nan")
    xs = np.asarray([math.log10(n + 1) for n, _ in pairs], dtype=float)
    ys = np.asarray([a for _, a in pairs], dtype=float)
    return float(np.trapezoid(ys, xs))


def summarize(rows: Sequence[Mapping]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int], List[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["condition"], int(row["num_updates"]))].append(row)

    per_condition = []
    for (model, condition, n), group in sorted(grouped.items()):
        per_condition.append(
            {
                "model": model,
                "condition": condition,
                "num_updates": n,
                "n": sum(not r.get("skipped", False) for r in group),
                "accuracy": accuracy(group),
                "mean_target_rank": float(
                    np.mean([r["target_rank"] for r in group if not r.get("skipped", False)])
                )
                if any(not r.get("skipped", False) for r in group)
                else float("nan"),
            }
        )

    lookup = {(r["model"], r["condition"], r["num_updates"]): r for r in per_condition}
    models = sorted({r["model"] for r in per_condition})
    levels = sorted({r["num_updates"] for r in per_condition})
    output: List[Dict] = []
    for model in models:
        ri_curve, pi_curve = {}, {}
        for n in levels:
            ri = lookup.get((model, "RI", n))
            pi = lookup.get((model, "PI", n))
            if ri and pi:
                ri_curve[n] = ri["accuracy"]
                pi_curve[n] = pi["accuracy"]
                output.append(
                    {
                        "model": model,
                        "num_updates": n,
                        "ri_accuracy": ri["accuracy"],
                        "pi_accuracy": pi["accuracy"],
                        "I": interference_asymmetry(ri["accuracy"], pi["accuracy"]),
                        "n_ri": ri["n"],
                        "n_pi": pi["n"],
                    }
                )
        if len(ri_curve) >= 2 and len(pi_curve) >= 2:
            output.append(
                {
                    "model": model,
                    "num_updates": "AUC",
                    "ri_accuracy": log_auc(ri_curve),
                    "pi_accuracy": log_auc(pi_curve),
                    "I": log_auc(ri_curve) - log_auc(pi_curve),
                    "n_ri": sum(lookup[(model, "RI", n)]["n"] for n in ri_curve),
                    "n_pi": sum(lookup[(model, "PI", n)]["n"] for n in pi_curve),
                }
            )
    return output


def bootstrap_model_gap(
    rows: Sequence[Mapping], model_a: str, model_b: str, *, n_boot: int = 2000, seed: int = 0
) -> Dict[str, float]:
    """Paired, level-stratified bootstrap of the mean-level asymmetry gap.

    The primary estimand gives each frozen interference level equal weight. Bootstrap
    resampling therefore occurs within level, preserving that weighting instead of
    letting random resamples over/under-represent a level.
    """

    def keyed(model: str):
        out = defaultdict(dict)
        for r in rows:
            if r["model"] != model or r.get("skipped", False):
                continue
            key = (r["episode_id"], r["query_key"], int(r["num_updates"]))
            out[key][r["condition"]] = int(bool(r["correct"]))
        return {k: v for k, v in out.items() if "RI" in v and "PI" in v}

    a, b = keyed(model_a), keyed(model_b)
    common = sorted(set(a) & set(b))
    if not common:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    by_level = defaultdict(list)
    for key in common:
        by_level[key[2]].append(key)

    def estimate(level_samples):
        level_gaps = []
        for _, keys in sorted(level_samples.items()):
            a_i = np.mean([a[k]["RI"] - a[k]["PI"] for k in keys])
            b_i = np.mean([b[k]["RI"] - b[k]["PI"] for k in keys])
            level_gaps.append(a_i - b_i)
        return float(np.mean(level_gaps))

    point = estimate(by_level)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sampled = {
            level: [keys[rng.randrange(len(keys))] for _ in keys]
            for level, keys in by_level.items()
        }
        boots.append(estimate(sampled))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"estimate": point, "lo": float(lo), "hi": float(hi), "n": len(common)}
