from __future__ import annotations

ALPACA_PREFIX = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve the following math problem, and put your final answer within \\boxed{{}}.

### Input:
{question}

### Response:
"""

DECISION_PREFIX = "To find the target value, we compute the following variables step by step:\n1."


def full_prompt(question: str) -> str:
    return ALPACA_PREFIX.format(question=question) + DECISION_PREFIX
