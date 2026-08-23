#!/usr/bin/env python3
"""Score matched positive/negative adaptation predictions for Topic 18."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
    return rows


def normalized_choice(text: str, allowed: set[str]) -> str | None:
    tokens = re.findall(r"[A-Za-z]+", text.upper())
    hits = [t for t in tokens if t in allowed]
    if len(hits) == 1:
        return hits[0]
    return None


def bootstrap_ci(values: list[float], n_boot: int, seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    return [
        means[int(0.025 * (n_boot - 1))],
        means[int(0.975 * (n_boot - 1))],
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args()

    design = {row["item_id"]: row for row in load_jsonl(args.design)}
    preds = load_jsonl(args.predictions)
    pred_map = {str(row["item_id"]): str(row["output"]) for row in preds}

    missing = sorted(set(design) - set(pred_map))
    extra = sorted(set(pred_map) - set(design))
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} items; first={missing[:5]}")
    if extra:
        raise ValueError(f"predictions contain {len(extra)} unknown items; first={extra[:5]}")

    scored = []
    invalid = 0
    for item_id, row in design.items():
        allowed = {row["action_a"].upper(), row["action_b"].upper()}
        choice = normalized_choice(pred_map[item_id], allowed)
        if choice is None:
            invalid += 1
        correct = choice == row["correct_action"].upper()
        scored.append({**row, "choice": choice, "correct": bool(correct)})

    by_condition = defaultdict(list)
    by_pair = defaultdict(dict)
    for row in scored:
        by_condition[row["condition"]].append(float(row["correct"]))
        by_pair[row["pair_id"]][row["condition"]] = float(row["correct"])

    if set(by_condition) != {"positive", "negative"}:
        raise ValueError(f"unexpected conditions: {sorted(by_condition)}")

    incomplete_pairs = [pid for pid, d in by_pair.items() if set(d) != {"positive", "negative"}]
    if incomplete_pairs:
        raise ValueError(f"incomplete matched pairs: {incomplete_pairs[:5]}")

    pair_diffs = [d["positive"] - d["negative"] for d in by_pair.values()]
    pos_acc = statistics.fmean(by_condition["positive"])
    neg_acc = statistics.fmean(by_condition["negative"])

    result = {
        "n_items": len(scored),
        "n_pairs": len(by_pair),
        "invalid_output_fraction": invalid / len(scored),
        "accuracy_positive": pos_acc,
        "accuracy_negative": neg_acc,
        "delta_inhibition": pos_acc - neg_acc,
        "paired_delta_mean": statistics.fmean(pair_diffs),
        "paired_delta_bootstrap_95ci": bootstrap_ci(
            pair_diffs, n_boot=args.bootstrap, seed=args.seed
        ),
        "pair_outcomes": {
            "positive_only_correct": sum(x == 1.0 for x in pair_diffs),
            "negative_only_correct": sum(x == -1.0 for x in pair_diffs),
            "same_outcome": sum(x == 0.0 for x in pair_diffs),
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
