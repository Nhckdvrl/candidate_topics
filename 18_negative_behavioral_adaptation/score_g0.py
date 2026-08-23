#!/usr/bin/env python3
"""Validate, score, and make the frozen Topic 18 G0 decision."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# Frozen operational bar.  Changing these after outcome inspection invalidates G0.
MIN_BASE = 64
MIN_MODELS = 3
SURVIVE_DELTA = 0.20
SURVIVE_CI_LOWER = 0.10
MODEL_MIN_DELTA = 0.10
ROBUSTNESS_MIN_DELTA = 0.10
KILL_CI_UPPER = 0.10
MAX_INVALID_FRACTION = 0.02
MAX_BASELINE_MARKED_BIAS = 0.10
MAX_BASELINE_SURFACE_BIAS = 0.25


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: row must be an object")
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def normalized_choice(text: str, allowed: set[str]) -> str | None:
    hits = [t for t in re.findall(r"[A-Za-z]+", text.upper()) if t in allowed]
    return hits[0] if len(hits) == 1 else None


def percentile_ci(values: list[float], n_boot: int, seed: int) -> list[float]:
    if n_boot < 100:
        raise ValueError("--bootstrap must be >= 100")
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return [means[int(.025 * (n_boot - 1))], means[int(.975 * (n_boot - 1))]]


def exact_mcnemar_p(positive_only: int, negative_only: int) -> float:
    """Exact two-sided paired sign test, with no scipy dependency."""
    n = positive_only + negative_only
    if n == 0:
        return 1.0
    k = min(positive_only, negative_only)
    lower = sum(math.comb(n, j) for j in range(k + 1)) / (2**n)
    return min(1.0, 2 * lower)


def validate_design(rows: list[dict]) -> dict[str, dict]:
    required = {
        "item_id", "pair_id", "condition", "action_a", "action_b",
        "marked_action", "neutral_action", "observation_order", "choice_order",
        "correct_action", "prompt",
    }
    design: dict[str, dict] = {}
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for i, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"design row {i}: missing {sorted(missing)}")
        item_id = str(row["item_id"])
        if item_id in design:
            raise ValueError(f"duplicate design item_id: {item_id}")
        if row["condition"] not in {"positive", "negative", "baseline"}:
            raise ValueError(f"{item_id}: invalid condition")
        if row.get("schema_version") != 2:
            raise ValueError(f"{item_id}: scorer requires frozen schema_version 2")
        if row["observation_order"] not in {"ab", "ba"} or row["choice_order"] not in {"ab", "ba"}:
            raise ValueError(f"{item_id}: invalid order code")
        if row["action_a"] == row["action_b"]:
            raise ValueError(f"{item_id}: actions must differ")
        if {row["marked_action"], row["neutral_action"]} != {
            row["action_a"], row["action_b"]
        }:
            raise ValueError(f"{item_id}: marked/neutral actions inconsistent")
        expected = row["marked_action"] if row["condition"] == "positive" else (
            row["neutral_action"] if row["condition"] == "negative" else None
        )
        if row["correct_action"] != expected:
            raise ValueError(f"{item_id}: incorrect answer key")
        design[item_id] = row
        by_pair[str(row["pair_id"])].append(row)

    for pair_id, group in by_pair.items():
        if Counter(r["condition"] for r in group) != Counter(
            {"positive": 1, "negative": 1, "baseline": 1}
        ):
            raise ValueError(f"{pair_id}: must contain one row per condition")
        frozen = ("action_a", "action_b", "marked_action", "neutral_action",
                  "observation_order", "choice_order")
        if any(len({r[k] for r in group}) != 1 for k in frozen):
            raise ValueError(f"{pair_id}: nuisance factors differ across conditions")
        # The literal prompt must be identical after masking numeric outcomes.
        # This catches accidental condition-specific instructions or wording.
        masked = {
            re.sub(r"produced score [+-]?\d+", "produced score <OUTCOME>", r["prompt"])
            for r in group
        }
        if len(masked) != 1:
            raise ValueError(f"{pair_id}: prompt text differs beyond feedback values")
        score_patterns = {
            r["condition"]: sorted(re.findall(r"produced score ([+-]?\d+)", r["prompt"]))
            for r in group
        }
        if score_patterns != {
            "positive": ["+1", "0"], "negative": ["-1", "0"],
            "baseline": ["0", "0"],
        }:
            raise ValueError(f"{pair_id}: prompt outcomes do not match frozen design")
    return design


def score_model(model_id: str, design: dict[str, dict], outputs: dict[str, str],
                n_boot: int, seed: int, model_family: str | None = None) -> tuple[dict, list[dict]]:
    scored = []
    for item_id, row in design.items():
        allowed = {str(row["action_a"]).upper(), str(row["action_b"]).upper()}
        choice = normalized_choice(outputs[item_id], allowed)
        correct = None if row["condition"] == "baseline" else (
            choice == str(row["correct_action"]).upper()
        )
        scored.append({**row, "choice": choice, "correct": correct})

    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in scored:
        by_pair[str(row["pair_id"])][str(row["condition"])] = row
    diffs = [float(g["positive"]["correct"]) - float(g["negative"]["correct"])
             for g in by_pair.values()]
    pos = [float(g["positive"]["correct"]) for g in by_pair.values()]
    neg = [float(g["negative"]["correct"]) for g in by_pair.values()]
    baseline_valid = [g["baseline"] for g in by_pair.values()
                      if g["baseline"]["choice"] is not None]

    def stratum_delta(field: str, value: str) -> float:
        vals = [float(g["positive"]["correct"]) - float(g["negative"]["correct"])
                for g in by_pair.values() if str(g["positive"][field]) == value]
        return statistics.fmean(vals)

    nuisance = {}
    for field in ("marked_position", "observation_order", "choice_order"):
        if field == "marked_position":
            levels = ("a", "b")
            for g in by_pair.values():
                for r in g.values():
                    r[field] = "a" if r["marked_action"] == r["action_a"] else "b"
        else:
            levels = ("ab", "ba")
        nuisance[field] = {level: stratum_delta(field, level) for level in levels}

    marked_rate = statistics.fmean(
        r["choice"] == str(r["marked_action"]).upper() for r in baseline_valid
    ) if baseline_valid else None
    surface_bias = {}
    for pair in sorted({(r["action_a"], r["action_b"]) for r in baseline_valid}):
        rs = [r for r in baseline_valid if (r["action_a"], r["action_b"]) == pair]
        surface_bias["/".join(pair)] = abs(statistics.fmean(
            r["choice"] == str(r["action_a"]).upper() for r in rs
        ) - .5)

    positive_only = sum(x == 1 for x in diffs)
    negative_only = sum(x == -1 for x in diffs)
    invalid = sum(r["choice"] is None for r in scored)
    result = {
        "model_id": model_id,
        "model_family": model_family or model_id,
        "n_pairs": len(by_pair),
        "invalid_output_fraction": invalid / len(scored),
        "accuracy_positive": statistics.fmean(pos),
        "accuracy_negative": statistics.fmean(neg),
        "delta_inhibition": statistics.fmean(diffs),
        "paired_delta_bootstrap_95ci": percentile_ci(diffs, n_boot, seed),
        "paired_exact_p": exact_mcnemar_p(positive_only, negative_only),
        "pair_outcomes": {
            "positive_only_correct": positive_only,
            "negative_only_correct": negative_only,
            "same_outcome": sum(x == 0 for x in diffs),
        },
        # Kept by pair so panel inference can cluster on the shared stimulus
        # rather than pretending model x stimulus observations are independent.
        "paired_differences": {
            pair_id: float(group["positive"]["correct"])
            - float(group["negative"]["correct"])
            for pair_id, group in sorted(by_pair.items())
        },
        "nuisance_stratum_deltas": nuisance,
        "baseline_marked_selection_rate": marked_rate,
        "baseline_surface_action_a_deviation": surface_bias,
    }
    return result, scored


def decide(models: list[dict], n_base: int, complete_factorial: bool) -> dict:
    validity_failures = []
    warnings = []
    if n_base < MIN_BASE:
        validity_failures.append(f"n_base={n_base} < {MIN_BASE}")
    if not complete_factorial:
        validity_failures.append("design is not one or more complete 64-cell factorial blocks")
    if len(models) < MIN_MODELS:
        validity_failures.append(f"models={len(models)} < {MIN_MODELS}")
    if len({m["model_family"] for m in models}) < MIN_MODELS:
        validity_failures.append(f"distinct model families < {MIN_MODELS}")
    for m in models:
        if m["invalid_output_fraction"] > MAX_INVALID_FRACTION:
            validity_failures.append(f"{m['model_id']}: invalid outputs exceed {MAX_INVALID_FRACTION}")
        marked = m["baseline_marked_selection_rate"]
        if marked is None or abs(marked - .5) > MAX_BASELINE_MARKED_BIAS:
            validity_failures.append(f"{m['model_id']}: baseline marked-action bias too large")
        if m["baseline_surface_action_a_deviation"] and max(
            m["baseline_surface_action_a_deviation"].values()
        ) > MAX_BASELINE_SURFACE_BIAS:
            # Complete marked-identity crossing makes this preference orthogonal
            # to condition.  Report it, but only marked-action imbalance can
            # confound the aggregate positive-vs-negative contrast.
            warnings.append(f"{m['model_id']}: strong preference within at least one symbol pair")
    if validity_failures:
        return {"verdict": "INVALID", "reasons": validity_failures, "warnings": warnings}

    pair_ids = set(models[0]["paired_differences"])
    if any(set(m["paired_differences"]) != pair_ids for m in models):
        return {"verdict": "INVALID", "reasons": ["models were not scored on identical pairs"]}
    # One value per shared base stimulus: average across the frozen model panel,
    # then bootstrap stimuli.  This avoids pseudo-replicating the same prompt.
    pooled_diffs = [statistics.fmean(
        m["paired_differences"][pair_id] for m in models
    ) for pair_id in sorted(pair_ids)]
    pooled_delta = statistics.fmean(pooled_diffs)
    pooled_ci = percentile_ci(pooled_diffs, 10000, 20260823)
    all_model_positive = all(m["delta_inhibition"] >= MODEL_MIN_DELTA for m in models)
    n_models_large = sum(m["delta_inhibition"] >= MODEL_MIN_DELTA for m in models)
    all_strata = all(
        value >= ROBUSTNESS_MIN_DELTA
        for m in models for field in m["nuisance_stratum_deltas"].values()
        for value in field.values()
    )
    if (pooled_delta >= SURVIVE_DELTA and pooled_ci[0] >= SURVIVE_CI_LOWER
            and all_model_positive and all_strata):
        verdict = "SURVIVE"
        reason = "large paired deficit survives every frozen model and counterbalance stratum"
    elif pooled_ci[1] < KILL_CI_UPPER:
        verdict = "KILL"
        reason = "matched design rules out an inhibition deficit of operational interest"
    elif n_models_large <= 1:
        verdict = "KILL"
        reason = "at most one frozen model has even a 0.10 gap; the cross-model claim fails"
    else:
        verdict = "INCONCLUSIVE"
        reason = "effect lies between the frozen survival and kill regions; no rescue sweep permitted"
    return {
        "verdict": verdict,
        "reason": reason,
        "pooled_delta": pooled_delta,
        "pooled_bootstrap_95ci": pooled_ci,
        "all_models_at_least_0.10": all_model_positive,
        "n_models_at_least_0.10": n_models_large,
        "all_counterbalance_strata_at_least_0.10": all_strata,
        "warnings": warnings,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True,
                   help="JSONL rows: model_family, model_id, item_id, output")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args()

    design = validate_design(load_jsonl(args.design))
    pred_rows = load_jsonl(args.predictions)
    by_model: dict[str, dict[str, str]] = defaultdict(dict)
    model_families: dict[str, str] = {}
    for i, row in enumerate(pred_rows, 1):
        needed = {"model_family", "model_id", "item_id", "output"}
        missing = needed - row.keys()
        if missing:
            raise ValueError(f"prediction row {i}: missing {sorted(missing)}")
        model_id = str(row["model_id"])
        family = str(row["model_family"])
        if not model_id or not family:
            raise ValueError(f"prediction row {i}: model identity must be non-empty")
        if model_id in model_families and model_families[model_id] != family:
            raise ValueError(f"model {model_id}: inconsistent model_family")
        model_families[model_id] = family
        item_id = str(row["item_id"])
        if item_id in by_model[model_id]:
            raise ValueError(f"duplicate prediction: model={model_id}, item={item_id}")
        by_model[model_id][item_id] = str(row["output"])

    for model_id, outputs in by_model.items():
        missing = sorted(set(design) - set(outputs))
        extra = sorted(set(outputs) - set(design))
        if missing or extra:
            raise ValueError(
                f"{model_id}: prediction coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
            )

    model_results = []
    for offset, (model_id, outputs) in enumerate(sorted(by_model.items())):
        result, _ = score_model(
            model_id, design, outputs, args.bootstrap, args.seed + offset,
            model_family=model_families[model_id],
        )
        model_results.append(result)

    n_base = len(design) // 3
    result = {
        "schema_version": 2,
        "n_models": len(model_results),
        "n_base": n_base,
        "models": model_results,
        "decision": decide(model_results, n_base, n_base % 64 == 0),
        "frozen_thresholds": {
            "min_base": MIN_BASE, "min_models": MIN_MODELS,
            "survive_delta": SURVIVE_DELTA, "survive_ci_lower": SURVIVE_CI_LOWER,
            "kill_ci_upper": KILL_CI_UPPER,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
