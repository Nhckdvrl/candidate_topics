#!/usr/bin/env python3
"""Generate the frozen, fully crossed Topic 18 G0 design.

The positive, negative, and no-preference-control episodes differ only in the
two observed outcomes.  Everything that can create a position/identity bias is
crossed explicitly: marked action, observation order, and answer-option order.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path


ACTION_PAIRS = [
    ("KEL", "ZOR"), ("VIM", "PAQ"), ("TUP", "NEX"), ("RIL", "BOS"),
    ("DAF", "MUR"), ("SEN", "GAK"), ("LOP", "WIB"), ("HES", "JUN"),
]

INTERFERENCE = (
    "Unrelated note: the device casing is gray and the session identifier has four digits."
)
CONDITIONS = ("positive", "negative", "baseline")


def make_prompt(
    a: str,
    b: str,
    marked: str,
    condition: str,
    observation_order: str,
    choice_order: str,
) -> str:
    """Render an experience followed by a first subsequent action.

    Crucially, the test never says "choose the better observed outcome".  That
    wording would turn the experiment into explicit comparison rather than
    behavioral adaptation from experience.
    """
    neutral = b if marked == a else a
    if condition == "positive":
        outcomes = {marked: "+1", neutral: "0"}
    elif condition == "negative":
        outcomes = {marked: "-1", neutral: "0"}
    elif condition == "baseline":
        outcomes = {marked: "0", neutral: "0"}
    else:
        raise ValueError(f"unknown condition: {condition}")

    first, second = (a, b) if observation_order == "ab" else (b, a)
    option_first, option_second = (a, b) if choice_order == "ab" else (b, a)
    return (
        "You operate a device by choosing one of two arbitrary commands. "
        "Each use produces a score, and your standing goal is to maximize total score.\n\n"
        f"Previous use 1: command {first} produced score {outcomes[first]}.\n"
        f"Previous use 2: command {second} produced score {outcomes[second]}.\n\n"
        f"{INTERFERENCE}\n\n"
        "A new use begins now. Choose one command. "
        f"Options: {option_first} or {option_second}. "
        "Answer with the command token only."
    )


def factorial_cells() -> list[tuple[str, str, str, str, str]]:
    return [
        (a, b, marked_position, observation_order, choice_order)
        for (a, b), marked_position, observation_order, choice_order in itertools.product(
            ACTION_PAIRS, ("a", "b"), ("ab", "ba"), ("ab", "ba")
        )
    ]


def build_rows(n_base: int, seed: int) -> list[dict]:
    if n_base <= 0:
        raise ValueError("--n-base must be positive")

    cells = factorial_cells()
    rng = random.Random(seed)
    # A complete block is exactly balanced.  Partial blocks are deterministic
    # random subsets, so --n-base 64 is the smallest confirmatory design.
    selected: list[tuple[str, str, str, str, str]] = []
    while len(selected) < n_base:
        block = cells.copy()
        rng.shuffle(block)
        selected.extend(block[: min(len(block), n_base - len(selected))])

    rows: list[dict] = []
    for i, (a, b, marked_position, observation_order, choice_order) in enumerate(selected):
        marked = a if marked_position == "a" else b
        neutral = b if marked == a else a
        pair_id = f"pair-{i:04d}"
        for condition in CONDITIONS:
            correct = marked if condition == "positive" else (
                neutral if condition == "negative" else None
            )
            rows.append(
                {
                    "schema_version": 2,
                    "item_id": f"{pair_id}-{condition}",
                    "pair_id": pair_id,
                    "condition": condition,
                    "action_a": a,
                    "action_b": b,
                    "marked_action": marked,
                    "neutral_action": neutral,
                    "observation_order": observation_order,
                    "choice_order": choice_order,
                    "correct_action": correct,
                    "prompt": make_prompt(
                        a, b, marked, condition, observation_order, choice_order
                    ),
                }
            )
    rng.shuffle(rows)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-base", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args()

    rows = build_rows(args.n_base, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output": str(args.output),
        "n_base": args.n_base,
        "n_items": len(rows),
        "complete_factorial": args.n_base % len(factorial_cells()) == 0,
        "factorial_block_size": len(factorial_cells()),
        "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
