#!/usr/bin/env python3
"""Build exactly one corrective exposure per semantic item per cycle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mcq_utils import LABELS, cyclic_permutations, format_training_user_message, stable_int


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    rows = []
    pairs = read_jsonl(args.pairs)
    for pair in pairs:
        for group in ("high", "low"):
            item = pair[group]
            choices = list(item["choices"])
            k = len(choices)
            perms = cyclic_permutations(k)
            offset = (stable_int(f"{args.seed}:{item['id']}") % k)
            for cycle in range(1, args.cycles + 1):
                perm = perms[(offset + cycle - 1) % k]
                perm_choices = [choices[i] for i in perm]
                local_answer = perm.index(int(item["answer"]))
                user = format_training_user_message(item["question"], perm_choices)
                # Training target contains both current label and invariant semantic content.
                response = f"Answer: {LABELS[local_answer]}. {choices[int(item['answer'])]}"
                rows.append(
                    {
                        "id": item["id"],
                        "pair_id": pair["pair_id"],
                        "group": group,
                        "split": pair["split"],
                        "category": pair["category"],
                        "cycle": cycle,
                        "permutation": perm,
                        "prompt": user,
                        "response": response,
                        "base_p_correct": float(item["p_correct"]),
                        "wrong_concentration": float(item["wrong_concentration"]),
                        "old_wrong": int(item["top_wrong"]),
                    }
                )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "n_pairs": len(pairs),
                "n_rows": len(rows),
                "cycles": args.cycles,
                "rows_per_cycle": len(rows) // args.cycles if args.cycles else 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
