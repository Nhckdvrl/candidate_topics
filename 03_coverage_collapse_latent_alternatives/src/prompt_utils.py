from __future__ import annotations

import re

ALPACA_PREFIX = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve the following math problem, and put your final answer within \\boxed{{}}.

### Input:
{question}

### Response:
"""

OPENING_SENTENCES = (
    "To find the target value, we compute the following variables step by step:",
    "We compute the following variables step by step to obtain the target value.",
    "The target value is found by sequentially computing the following variables.",
)


def decision_prefix(variant: int = 0) -> str:
    if not 0 <= variant < len(OPENING_SENTENCES):
        raise ValueError(f"prefix variant must be 0..{len(OPENING_SENTENCES)-1}, got {variant}")
    return OPENING_SENTENCES[variant] + "\n1."


def _replace_query_target(question: str, old_target: str, replacement: str) -> str:
    old_target = re.escape(old_target)
    patterns = (
        rf"(determine the value of\s+){old_target}\b",
        rf"(what is the resulting value of\s+){old_target}\b",
    )
    for pat in patterns:
        out, n = re.subn(pat, lambda m: m.group(1) + replacement, question, count=1, flags=re.IGNORECASE)
        if n:
            return out
    raise ValueError(f"Could not replace query target {old_target!r} in question: {question}")


def mask_query_target(question: str, target: str) -> str:
    """Remove only the query target identity while preserving every graph equation."""
    return _replace_query_target(question, target, "the requested variable")


def flip_query_target(question: str, target: str, alternative_target: str) -> str:
    """Matched counterfactual: keep the graph fixed and query the opposite branch leaf."""
    if target == alternative_target:
        raise ValueError("alternative_target must differ from target")
    return _replace_query_target(question, target, alternative_target)


def full_prompt(question: str, prefix_variant: int = 0) -> str:
    return ALPACA_PREFIX.format(question=question) + decision_prefix(prefix_variant)
