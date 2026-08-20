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


def parse_equations(question: str) -> dict[str, str]:
    """Parse single-letter equations from the official ArithChain prompt."""
    # Relations occur before the question clause and use ';' separators. We intentionally
    # accept both variable expressions and numeric premises.
    rules: dict[str, str] = {}
    for lhs, rhs in re.findall(r"\b([a-z])\s*=\s*([^;,.?]+)", question):
        rhs = rhs.strip()
        # Keep only the first occurrence of a lhs; the premise may be repeated in the "If ..." clause.
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


def parse_first_fork(question: str) -> ForkExample:
    rules = parse_equations(question)
    target = parse_target(question)
    premise = parse_premise(question)
    children = sorted(lhs for lhs, rhs in rules.items() if premise in referenced_vars(rhs) and lhs != premise)
    if len(children) != 2:
        raise ValueError(f"Expected exactly 2 first-fork children of {premise}; got {children}\n{question}")
    anc = ancestors(target, rules)
    viable = [c for c in children if c in anc]
    if len(viable) != 1:
        raise ValueError(f"Expected exactly one viable first branch; candidates={children}, target={target}, ancestors={sorted(anc)}")
    return ForkExample(premise=premise, target=target, candidate_a=children[0], candidate_b=children[1], viable=viable[0])
