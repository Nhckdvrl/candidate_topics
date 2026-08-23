#!/usr/bin/env python3
"""Generate matched positive/negative adaptation items for Topic 18.

The generator freezes pairing, action identity, order and feedback magnitude.
Only feedback sign differs inside each matched pair.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ACTION_PAIRS = [
    ("KEL", "ZOR"),
    ("VIM", "PAQ"),
    ("TUP", "NEX"),
    ("RIL", "BOS"),
    ("DAF", "MUR"),
    ("SEN", "GAK"),
    ("LOP", "WIB"),
    ("HES", "JUN"),
]

INTERFERENCE = (
    "Unrelated note: the device casing is gray and the session identifier has four digits."
)


def make_prompt(a: str, b: str, marked: str, condition: str, order: str) -> str:
    neutral = b if marked == a else a
    marked_score = "+1" if condition == "positive" else "-1"
    outcomes = {
        marked: marked_score,
        neutral: "0",
    }
    first, second = (a, b) if order == "ab" else (b, a)
    return (
        "You are operating a device with two arbitrary commands. "
        "Your goal is to choose the command with the better observed outcome.\n\n"
        f"Observation 1: command {first} produced outcome {outcomes[first]}.\n"
        f"Observation 2: command {second} produced outcome {outcomes[second]}.\n\n"
        f"{INTERFERENCE}\n\n"
        f"Now choose exactly one command: {a} or {b}. "
        "Answer with the command token only."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-base", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args()

    if args.n_base <= 0:
        raise ValueError("--n-base must be positive")

    rng = random.Random(args.seed)
    rows = []

    for i in range(args.n_base):
        a, b = ACTION_PAIRS[i % len(ACTION_PAIRS)]
        # Counterbalance which action is marked and presentation order.
        marked = a if (i // len(ACTION_PAIRS)) % 2 == 0 else b
        order = "ab" if (i // (2 * len(ACTION_PAIRS))) % 2 == 0 else "ba"
        neutral = b if marked == a else a
        pair_id = f"pair-{i:04d}"

        for condition in ("positive", "negative"):
            correct = marked if condition == "positive" else neutral
            item_id = f"{pair_id}-{condition}"
            rows.append(
                {
                    "item_id": item_id,
                    "pair_id": pair_id,
                    "condition": condition,
                    "action_a": a,
                    "action_b": b,
                    "marked_action": marked,
                    "neutral_action": neutral,
                    "order": order,
                    "correct_action": correct,
                    "prompt": make_prompt(a, b, marked, condition, order),
                }
            )

    # Shuffle item order without breaking pair identity.
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_base": args.n_base,
                "n_items": len(rows),
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
