from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ForkExample:
    premise: str
    target: str
    candidate_a: str
    candidate_b: str
    viable: str
    candidate_a_target: str
    candidate_b_target: str
    alternative_target: str


def parse_equations(question: str) -> dict[str, str]:
    """Parse single-letter equations from the official ArithChain prompt."""
    rules: dict[str, str] = {}
    for lhs, rhs in re.findall(r"\b([a-z])\s*=\s*([^;,.?]+)", question):
        rhs = rhs.strip()
        # The numeric premise can be repeated in the final "If ..." clause.
        rules.setdefault(lhs, rhs)
    return rules


def parse_target(question: str) -> str:
    pats = [
        r"determine the value of\s+([a-z])\b",
        r"what is the resulting value of\s+([a-z])\b",
    ]
    for p in pats:
        m = re.search(p, question, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
    raise ValueError(f"Could not parse target from: {question}")


def parse_premise(question: str) -> str:
    m = re.search(r"\bIf\s+([a-z])\s*=\s*[+-]?\d+", question, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse premise from: {question}")
    return m.group(1).lower()


def referenced_vars(expr: str) -> set[str]:
    return set(re.findall(r"\b([a-z])\b", expr.lower()))


def direct_children(node: str, rules: dict[str, str]) -> list[str]:
    return sorted(lhs for lhs, rhs in rules.items() if lhs != node and node in referenced_vars(rhs))


def ancestors(target: str, rules: dict[str, str]) -> set[str]:
    seen: set[str] = set()
    stack = [target]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node in rules:
            stack.extend(v for v in referenced_vars(rules[node]) if v in rules)
    return seen


def unique_terminal_from(start: str, rules: dict[str, str]) -> str:
    """Follow one ArithChain branch to its unique leaf."""
    seen: set[str] = set()
    node = start
    while True:
        if node in seen:
            raise ValueError(f"Cycle encountered while following branch from {start}: {node}")
        seen.add(node)
        children = direct_children(node, rules)
        if not children:
            return node
        if len(children) != 1:
            raise ValueError(f"Expected a chain after first fork; {node} has children {children}")
        node = children[0]


def parse_first_fork(question: str) -> ForkExample:
    rules = parse_equations(question)
    target = parse_target(question)
    premise = parse_premise(question)
    children = direct_children(premise, rules)
    if len(children) != 2:
        raise ValueError(f"Expected exactly 2 first-fork children of {premise}; got {children}\n{question}")

    anc = ancestors(target, rules)
    viable = [c for c in children if c in anc]
    if len(viable) != 1:
        raise ValueError(
            f"Expected exactly one viable first branch; candidates={children}, "
            f"target={target}, ancestors={sorted(anc)}"
        )

    target_a = unique_terminal_from(children[0], rules)
    target_b = unique_terminal_from(children[1], rules)
    if target not in {target_a, target_b}:
        raise ValueError(
            f"Query target {target} is not a terminal of either first branch: "
            f"{children[0]}->{target_a}, {children[1]}->{target_b}"
        )
    alternative_target = target_b if target == target_a else target_a

    return ForkExample(
        premise=premise,
        target=target,
        candidate_a=children[0],
        candidate_b=children[1],
        viable=viable[0],
        candidate_a_target=target_a,
        candidate_b_target=target_b,
        alternative_target=alternative_target,
    )
