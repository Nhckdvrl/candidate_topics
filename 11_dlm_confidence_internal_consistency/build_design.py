#!/usr/bin/env python3
"""Build Topic-11 v3: retroactive 2x2 factorial.

The trajectory is fixed. External correctness is manipulated only in the prompt.
Internal consistency is manipulated only in a *future suffix check*, after the
trajectory. Primary confidence is read on unchanged trajectory tokens that occur
before the suffix, so a consistency effect is necessarily retroactive under the
final-forward DLM score.

Anchor values in prompt/check are encoded as different arithmetic aliases rather
than copied literally, preventing a raw digit-match heuristic from defining the
factor.
"""

from __future__ import annotations
import argparse, json, random, re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DESIGN_VERSION = "v3_retroactive_semantic_alias"

ALIAS_BASES = [(7, 11), (8, 13), (9, 14), (6, 12)]


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
        raise ValueError(self.symbol)

    def instruction(self) -> str:
        return {"+": f"add {self.value}",
                "-": f"subtract {self.value}",
                "*": f"multiply by {self.value}"}[self.symbol]


@dataclass(frozen=True)
class Sample:
    design_version: str
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
    check_anchor: int
    operations: list[dict]
    template_id: int
    prompt: str
    trajectory_text: str
    check_text: str
    result_char_spans: list[list[int]]
    reported_final: int
    true_final: int


def apply_chain(start: int, operations: Iterable[Operation]) -> tuple[list[int], int]:
    states = [start]
    x = start
    for op in operations:
        x = op.apply(x)
        states.append(x)
    return states, x


def alias_expr(value: int, base: int) -> str:
    residual = value - base
    if residual <= 0:
        raise ValueError("alias requires value > base")
    return f"{base} + {residual}"


def render_prompt(anchor: int, operations: list[Operation], template_id: int) -> str:
    steps = ", then ".join(op.instruction() for op in operations)
    pbase, _ = ALIAS_BASES[template_id % len(ALIAS_BASES)]
    expr = alias_expr(anchor, pbase)
    templates = [
        f"Apply these operations in order: {steps}. Show every intermediate state and the final value. For verification, the ground-truth starting value is {expr}.",
        f"Execute this arithmetic program: {steps}. Write each intermediate value and the final answer. The correct initial value is {expr}.",
        f"Compute the state sequence for these operations: {steps}. Show all intermediate states and the final state. The specified starting value is {expr}.",
        f"Carry out the following transformations exactly: {steps}. Report each intermediate state and the final value. The task's initial value is {expr}.",
    ]
    return templates[template_id % len(templates)]


def render_check(anchor: int, template_id: int) -> str:
    _, cbase = ALIAS_BASES[template_id % len(ALIAS_BASES)]
    expr = alias_expr(anchor, cbase)
    templates = [
        f"Consistency check: the initial value used in the steps above equals {expr}.",
        f"Consistency check: the starting value used by the derivation above is {expr}.",
        f"Consistency check: the derivation above begins from a value equal to {expr}.",
        f"Consistency check: the computation above uses an initial value of {expr}.",
    ]
    return templates[template_id % len(templates)] + "\n"


def _append_line(lines: list[str], spans: list[list[int]], line: str, result: int) -> None:
    result_text = str(result)
    if not line.endswith(result_text):
        raise AssertionError("result must be trailing substring")
    cursor = sum(len(x) + 1 for x in lines)
    start = cursor + len(line) - len(result_text)
    spans.append([start, start + len(result_text)])
    lines.append(line)


def render_trajectory(branch_anchor: int, operations: list[Operation]) -> tuple[str, int, list[list[int]]]:
    states, final = apply_chain(branch_anchor, operations)
    lines: list[str] = []
    spans: list[list[int]] = []
    for i, op in enumerate(operations, start=1):
        lhs, rhs = states[i-1], states[i]
        _append_line(lines, spans, f"Step {i}: {lhs} {op.symbol} {op.value} = {rhs}", rhs)
    _append_line(lines, spans, f"Final answer: {final}", final)
    text = "\n".join(lines) + "\n"
    return text, final, spans


def sample_operations(rng: random.Random, min_anchor: int) -> list[Operation]:
    patterns = [("+", "*", "-"), ("*", "+", "-"), ("+", "-", "*"), ("-", "+", "*")]
    symbols = rng.choice(patterns)
    ops: list[Operation] = []
    floor = min_anchor
    for symbol in symbols:
        if symbol == "+":
            v = rng.randint(2, 9)
            floor += v
        elif symbol == "*":
            v = rng.randint(2, 5)
            floor *= v
        else:
            v = rng.randint(1, max(1, min(9, floor - 1)))
            floor -= v
        ops.append(Operation(symbol, v))
    return ops


def _standalone_integer_occurs(text: str, value: int) -> bool:
    return re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", text) is not None


def aliases_are_clean(x: int, y: int, ops: list[Operation], template_id: int) -> bool:
    """Reject accidental literal copies of either anchor in semantic aliases.

    Also reject a changed residual if it equals any numeric state or operation
    value appearing in either mirrored trajectory. This keeps the factor semantic
    rather than a hidden raw-number match.
    """
    pbase, cbase = ALIAS_BASES[template_id % len(ALIAS_BASES)]
    residues = {x-pbase, y-pbase, x-cbase, y-cbase}
    if min(residues) <= 0:
        return False
    states_x, _ = apply_chain(x, ops)
    states_y, _ = apply_chain(y, ops)
    forbidden = set(states_x) | set(states_y) | {x, y} | {op.value for op in ops}
    if residues & forbidden:
        return False
    for v in (x, y):
        for b in (pbase, cbase):
            for z in (x, y):
                if _standalone_integer_occurs(alias_expr(z, b), v):
                    return False
    return True


def build_orientation(pair_id: int, orientation: int, branch: int, alt: int,
                      x: int, y: int, ops: list[Operation], template_id: int) -> list[Sample]:
    trajectory, reported_final, spans = render_trajectory(branch, ops)
    _, true_branch = apply_chain(branch, ops)
    _, true_alt = apply_chain(alt, ops)
    assert reported_final == true_branch != true_alt

    specs = [
        ("CC", True,  True,  branch, branch, true_branch),
        ("IC", False, True,  branch, alt,    true_branch),
        ("CW", True,  False, alt,    branch, true_alt),
        ("IW", False, False, alt,    alt,    true_alt),
    ]
    out = []
    for cell, consistent, correct, prompt_anchor, check_anchor, true_final in specs:
        out.append(Sample(
            design_version=DESIGN_VERSION, pair_id=pair_id, orientation=orientation,
            cell=cell, internal_consistent=consistent, externally_correct=correct,
            anchor_x=x, anchor_y=y, branch_anchor=branch, alternate_anchor=alt,
            prompt_anchor=prompt_anchor, check_anchor=check_anchor,
            operations=[asdict(op) for op in ops], template_id=template_id,
            prompt=render_prompt(prompt_anchor, ops, template_id),
            trajectory_text=trajectory, check_text=render_check(check_anchor, template_id),
            result_char_spans=spans, reported_final=reported_final, true_final=true_final,
        ))
    return out


def build_pair(pair_id: int, rng: random.Random, anchor_min: int, anchor_max: int) -> list[Sample]:
    for _ in range(10000):
        x, y = rng.sample(range(anchor_min, anchor_max + 1), 2)
        template_id = rng.randrange(len(ALIAS_BASES))
        ops = sample_operations(rng, min(x, y))
        if aliases_are_clean(x, y, ops, template_id):
            pair = []
            pair += build_orientation(pair_id, 0, x, y, x, y, ops, template_id)
            pair += build_orientation(pair_id, 1, y, x, x, y, ops, template_id)
            validate_pair(pair)
            return pair
    raise RuntimeError("Could not sample a clean semantic-alias pair")


def validate_pair(samples: list[Sample]) -> None:
    assert len(samples) == 8
    grouped: dict[int, dict[str, Sample]] = {}
    for s in samples:
        assert s.design_version == DESIGN_VERSION
        grouped.setdefault(s.orientation, {})[s.cell] = s
        assert not _standalone_integer_occurs(s.prompt, s.prompt_anchor)
        assert not _standalone_integer_occurs(s.check_text, s.check_anchor)
        for start, end in s.result_char_spans:
            assert s.trajectory_text[start:end].lstrip("-").isdigit()
    assert set(grouped) == {0, 1}
    for cells in grouped.values():
        assert set(cells) == {"CC", "IC", "CW", "IW"}
        cc, ic, cw, iw = (cells[k] for k in ("CC", "IC", "CW", "IW"))
        assert len({s.trajectory_text for s in cells.values()}) == 1
        assert len({tuple(map(tuple, s.result_char_spans)) for s in cells.values()}) == 1
        assert cc.prompt == ic.prompt and cw.prompt == iw.prompt
        assert cc.check_text == cw.check_text and ic.check_text == iw.check_text
        assert cc.prompt != cw.prompt and ic.prompt != iw.prompt
        assert cc.check_text != ic.check_text and cw.check_text != iw.check_text
        assert cc.externally_correct and ic.externally_correct
        assert not cw.externally_correct and not iw.externally_correct
        assert cc.internal_consistent and cw.internal_consistent
        assert not ic.internal_consistent and not iw.internal_consistent
        assert cc.reported_final == cc.true_final == ic.true_final
        assert cw.reported_final != cw.true_final and iw.reported_final != iw.true_final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-pairs", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--anchor-min", type=int, default=20)
    p.add_argument("--anchor-max", type=int, default=89)
    a = p.parse_args()
    if a.num_pairs < 1 or a.anchor_max - a.anchor_min < 2:
        raise SystemExit("invalid design size/range")
    rng = random.Random(a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with a.out.open("w", encoding="utf-8") as f:
        for pid in range(a.num_pairs):
            for s in build_pair(pid, rng, a.anchor_min, a.anchor_max):
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} samples ({a.num_pairs} mirrored pairs) -> {a.out}")


if __name__ == "__main__":
    main()
