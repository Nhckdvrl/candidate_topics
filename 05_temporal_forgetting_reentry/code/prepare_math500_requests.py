#!/usr/bin/env python3
"""Prepare MATH-500 requests matching the seed repository's lm-eval prompt."""
from __future__ import annotations

import argparse

from common import write_jsonl

# Exact doc_to_text from uw-nsl/Temporal_Forgetting's
# lm_eval/tasks/MATH-500/hendrycks_math_500.yaml.
PROMPT = (
    "Solve the following math problem. Present the final answer in the format: "
    "Final Answer: \\boxed{{your_answer}}.\nProblem: {problem}\nAnswer:"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, x in enumerate(ds):
        pid = str(x.get("unique_id") or f"math500_{i:04d}")
        rows.append(
            {
                "request_id": pid,
                "problem_id": pid,
                "prompt": PROMPT.format(problem=x["problem"]),
                "problem": x["problem"],
                "gold_answer": x["answer"],
                "gold_solution": x.get("solution"),
                "subject": x.get("subject"),
                "level": x.get("level"),
                "stop": ["Problem:"],
            }
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} MATH-500 requests")


if __name__ == "__main__":
    main()
