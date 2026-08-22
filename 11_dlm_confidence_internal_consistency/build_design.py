#!/usr/bin/env python3
"""Build the locked 2x2 factorial dataset for Topic 11.

The design separates two factors with minimal interventions:
  * internal consistency: whether the announced initial state matches the state
    actually used by the downstream arithmetic trajectory;
  * external correctness: whether that downstream state matches the initial
    state specified by the problem prompt.

For every anchor pair (x, y), we build two mirrored orientations so number-token
preference cancels at the pair level.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Operation:
    symbol: str
    value: int

    def apply(self, x: int) -> int:
        if self.symbol == "+":
            return x + self.value
        if self.symbol == "-":
            return x - self.value
        if self.symbol == "*":
            return x * self.value
        raise ValueError(f"Unsupported operation: {self.symbol}")

    def instruction(self) -> str:
        if self.symbol == "+":
            return f"add {self.value}"
        if self.symbol == "-":
            return f"subtract {self.value}"
        return f"multiply by {self.value}"


@dataclass(frozen=True)
class Sample:
    pair_id: int
    orientation: int
    cell: str
    internal_consistent: bool
    externally_correct: bool
    anchor_x: int
    anchor_y: int
    branch_anchor: int
    alternate_anchor: int
    prompt_anchor: int
    announced_anchor: int
    operations: list[dict]
    template_id: int
    prompt: str
    announcement_text: str
    continuation_text: str
    reported_final: int
    true_final: int


def apply_chain(start: int, operations: Iterable[Operation]) -> tuple[list[int], int]:
    states = [start]
    x = start
    for op in operations:
        x = op.apply(x)
        states.append(x)
    return states, x


def render_prompt(anchor: int, operations: list[Operation], template_id: int) -> str:
    steps = ", then ".join(op.instruction() for op in operations)
    templates = [
        f"A calculator starts from the integer {anchor}. Apply these operations in order: {steps}. What is the final value? Show the intermediate states.",
        f"A counter initially reads {anchor}. In order, {steps}. Compute the final counter value and show each intermediate state.",
        f"Set variable v to {anchor}. Next, {steps}. What value does v have at the end? Show the intermediate values.",
        f"The starting state is {anchor}. Apply the following sequence exactly: {steps}. Report every intermediate state and the final value.",
    ]
    return templates[template_id % len(templates)]


def render_trajectory(
    branch_anchor: int,
    announced_anchor: int,
    operations: list[Operation],
    template_id: int,
) -> tuple[str, str, int]:
    states, final = apply_chain(branch_anchor, operations)
    announcement_templates = [
        f"Initial state: {announced_anchor}\n",
        f"Starting value: {announced_anchor}\n",
        f"v0 = {announced_anchor}\n",
        f"State at start: {announced_anchor}\n",
    ]
    announcement = announcement_templates[template_id % len(announcement_templates)]
    lines: list[str] = []
    for i, op in enumerate(operations, start=1):
        lhs = states[i - 1]
        rhs = states[i]
        lines.append(f"Step {i}: {lhs} {op.symbol} {op.value} = {rhs}")
    lines.append(f"Final answer: {final}")
    return announcement, "\n".join(lines), final


def sample_operations(rng: random.Random, min_anchor: int) -> list[Operation]:
    """Sample a short affine chain with positive, bounded intermediate values."""
    templates = [
        ("+", "*", "-"),
        ("*", "+", "-"),
        ("+", "-", "*"),
        ("-", "+", "*"),
    ]
    symbols = rng.choice(templates)
    ops: list[Operation] = []
    current_floor = min_anchor
    for symbol in symbols:
        if symbol == "+":
            value = rng.randint(2, 9)
        elif symbol == "*":
            value = rng.randint(2, 5)
        else:
            # Keep all branches comfortably positive.
            value = rng.randint(1, max(1, min(9, current_floor - 1)))
        ops.append(Operation(symbol, value))
        if symbol == "+":
            current_floor += value
        elif symbol == "-":
            current_floor -= value
        else:
            current_floor *= value
    return ops


def build_orientation(
    pair_id: int,
    orientation: int,
    branch_anchor: int,
    alternate_anchor: int,
    x: int,
    y: int,
    operations: list[Operation],
    template_id: int,
) -> list[Sample]:
    """Build A/B/C/D for one fixed downstream branch.

    A/CC: prompt=branch, announce=branch  -> consistent, final correct
    B/IC: prompt=branch, announce=alt     -> inconsistent, final correct
    C/CW: prompt=alt,    announce=branch  -> consistent, final wrong
    D/IW: prompt=alt,    announce=alt     -> inconsistent, final wrong

    Within an orientation, the downstream continuation is *identical in all four
    cells*. Only the prompt anchor and announcement anchor vary.
    """
    _, _, reported_final = render_trajectory(branch_anchor, branch_anchor, operations, template_id)
    _, true_final_branch = apply_chain(branch_anchor, operations)
    _, true_final_alt = apply_chain(alternate_anchor, operations)
    assert reported_final == true_final_branch
    assert true_final_alt != true_final_branch

    specs = [
        ("CC", True, True, branch_anchor, branch_anchor, true_final_branch),
        ("IC", False, True, branch_anchor, alternate_anchor, true_final_branch),
        ("CW", True, False, alternate_anchor, branch_anchor, true_final_alt),
        ("IW", False, False, alternate_anchor, alternate_anchor, true_final_alt),
    ]
    out: list[Sample] = []
    for cell, consistent, correct, prompt_anchor, announced_anchor, true_final in specs:
        announcement, continuation, reported = render_trajectory(
            branch_anchor, announced_anchor, operations, template_id
        )
        out.append(
            Sample(
                pair_id=pair_id,
                orientation=orientation,
                cell=cell,
                internal_consistent=consistent,
                externally_correct=correct,
                anchor_x=x,
                anchor_y=y,
                branch_anchor=branch_anchor,
                alternate_anchor=alternate_anchor,
                prompt_anchor=prompt_anchor,
                announced_anchor=announced_anchor,
                operations=[asdict(op) for op in operations],
                template_id=template_id,
                prompt=render_prompt(prompt_anchor, operations, template_id),
                announcement_text=announcement,
                continuation_text=continuation,
                reported_final=reported,
                true_final=true_final,
            )
        )
    return out


def build_pair(pair_id: int, rng: random.Random, anchor_min: int, anchor_max: int) -> list[Sample]:
    x, y = rng.sample(range(anchor_min, anchor_max + 1), 2)
    operations = sample_operations(rng, min(x, y))
    template_id = rng.randrange(4)
    samples = []
    samples.extend(build_orientation(pair_id, 0, x, y, x, y, operations, template_id))
    samples.extend(build_orientation(pair_id, 1, y, x, x, y, operations, template_id))
    return samples


def validate_pair(samples: list[Sample]) -> None:
    assert len(samples) == 8
    by_orientation: dict[int, dict[str, Sample]] = {}
    for s in samples:
        by_orientation.setdefault(s.orientation, {})[s.cell] = s
    assert set(by_orientation) == {0, 1}
    for cells in by_orientation.values():
        assert set(cells) == {"CC", "IC", "CW", "IW"}
        cc, ic, cw, iw = (cells[k] for k in ("CC", "IC", "CW", "IW"))
        # The downstream arithmetic is fixed across all four cells.
        assert len({s.continuation_text for s in cells.values()}) == 1
        assert len({s.reported_final for s in cells.values()}) == 1
        # Correctness factor: same output, only the prompt anchor changes.
        assert cc.announcement_text == cw.announcement_text
        assert ic.announcement_text == iw.announcement_text
        assert cc.prompt == ic.prompt
        assert cw.prompt == iw.prompt
        # Consistency factor: same prompt + same continuation, only announcement changes.
        assert cc.announcement_text != ic.announcement_text
        assert cw.announcement_text != iw.announcement_text
        assert cc.externally_correct and ic.externally_correct
        assert not cw.externally_correct and not iw.externally_correct
        assert cc.internal_consistent and cw.internal_consistent
        assert not ic.internal_consistent and not iw.internal_consistent
        assert cc.reported_final == cc.true_final == ic.true_final
        assert cw.reported_final != cw.true_final
        assert iw.reported_final != iw.true_final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-pairs", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--anchor-min", type=int, default=20)
    p.add_argument("--anchor-max", type=int, default=89)
    args = p.parse_args()
    if args.num_pairs < 1:
        raise SystemExit("--num-pairs must be positive")
    if args.anchor_max - args.anchor_min < 2:
        raise SystemExit("Anchor range is too small")

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for pair_id in range(args.num_pairs):
            pair = build_pair(pair_id, rng, args.anchor_min, args.anchor_max)
            validate_pair(pair)
            for s in pair:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} samples ({args.num_pairs} mirrored anchor pairs) -> {args.out}")


if __name__ == "__main__":
    main()
