#!/usr/bin/env python3
"""Match forgotten items to never-correct controls before re-entry inference."""
from __future__ import annotations

import argparse
import math

from common import read_jsonl, write_jsonl, stable_hash_int


def tok_len(row: dict) -> int:
    return max(1, len(str(row.get("prompt", "")).split()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-length-ratio", type=float, default=1.6)
    ap.add_argument("--allow-cross-subject", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.groups)
    F = [r for r in rows if r["group"] == "forgotten"]
    N = [r for r in rows if r["group"] == "never_correct" and r.get("verified_correct_trace")]
    used: set[str] = set()
    pairs = []

    F = sorted(F, key=lambda r: (int(r.get("level", 0) or 0), tok_len(r)), reverse=True)
    for f in F:
        best = None
        flen = tok_len(f)
        flevel = int(f.get("level", 0) or 0)
        for n in N:
            nid = str(n["problem_id"])
            if nid in used:
                continue
            if not args.allow_cross_subject and f.get("subject") and n.get("subject"):
                if f["subject"] != n["subject"]:
                    continue
            nlen = tok_len(n)
            ratio = max(flen, nlen) / min(flen, nlen)
            if ratio > args.max_length_ratio:
                continue
            nlevel = int(n.get("level", 0) or 0)
            score = abs(flevel - nlevel) + 0.5 * abs(math.log(flen / nlen))
            if best is None or score < best[0]:
                best = (score, n)
        if best is None:
            continue
        _, n = best
        used.add(str(n["problem_id"]))
        pairs.append(
            {
                "pair_id": f"FN_{len(pairs):04d}",
                "split": "discovery" if stable_hash_int(str(f["problem_id"])) % 10 < 6 else "confirmation",
                "forgotten_problem_id": str(f["problem_id"]),
                "never_problem_id": str(n["problem_id"]),
                "subject_f": f.get("subject"),
                "subject_n": n.get("subject"),
                "level_f": f.get("level"),
                "level_n": n.get("level"),
                "prompt_tokens_proxy_f": flen,
                "prompt_tokens_proxy_n": tok_len(n),
                "match_score": best[0],
            }
        )

    write_jsonl(args.output, pairs)
    print(f"forgotten={len(F)} never={len(N)} pairs={len(pairs)}")


if __name__ == "__main__":
    main()
