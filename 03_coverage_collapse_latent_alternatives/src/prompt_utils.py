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


def mask_query_target(question: str, target: str) -> str:
    """Remove the queried target identity while leaving the graph rules untouched.

    This is a negative control: without knowing which terminal node is queried, the
    globally viable first branch is not identifiable from the two symmetric chains.
    """
    target = re.escape(target)
    patterns = (
        rf"(determine the value of\s+){target}\b",
        rf"(what is the resulting value of\s+){target}\b",
    )
    out = question
    for pat in patterns:
        out, n = re.subn(pat, r"\1the requested variable", out, count=1, flags=re.IGNORECASE)
        if n:
            return out
    raise ValueError(f"Could not mask target {target!r} in question: {question}")


def full_prompt(question: str, prefix_variant: int = 0) -> str:
    return ALPACA_PREFIX.format(question=question) + decision_prefix(prefix_variant)
