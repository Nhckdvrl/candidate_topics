#!/usr/bin/env python3
"""Prepare standard MATH-500 generation requests with stable problem IDs."""
from __future__ import annotations

import argparse

from common import write_jsonl

PROMPT = """Please reason step by step and put your final answer within \\boxed{{}}.\n\n{problem}"""


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
            }
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} MATH-500 requests")


if __name__ == "__main__":
    main()
