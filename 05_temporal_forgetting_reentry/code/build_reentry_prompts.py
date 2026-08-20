#!/usr/bin/env python3
"""Create length-matched, answer-leakage-audited re-entry prompts.

Primary fractions are defined on the frozen old-self trace. For forgotten-item
controls (`other_correct`, `final_wrong`), we truncate at complete reasoning-step
boundaries to approximately the *same tokenizer token budget* as the old-self
prefix. This avoids a major confound in the initial implementation.

The 0% baseline is emitted exactly once per problem, not once per source.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

from common import (
    read_jsonl,
    write_jsonl,
    split_reasoning_steps,
    explicit_answer_leak_reasons,
    stable_hash_int,
)


def n_tokens(tok, text: str) -> int:
    if tok is None:
        return len(text.split())
    return len(tok(text, add_special_tokens=False).input_ids)


def prefix_from_fraction(trace: str, frac: float) -> str:
    steps = split_reasoning_steps(trace)
    if not steps or frac <= 0:
        return ""
    # Never consume the whole trace.
    n = max(1, math.ceil(frac * len(steps)))
    n = min(n, max(1, len(steps) - 1))
    return "\n".join(steps[:n])


def prefix_under_budget(tok, trace: str, target_tokens: int) -> str:
    steps = split_reasoning_steps(trace)
    if not steps or target_tokens <= 0:
        return ""
    kept = []
    for step in steps[:-1] if len(steps) > 1 else steps:
        trial = "\n".join(kept + [step])
        if kept and n_tokens(tok, trial) > target_tokens:
            break
        kept.append(step)
        if n_tokens(tok, trial) >= target_tokens:
            break
    return "\n".join(kept)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer", default="UWNSL/Qwen2.5-7B-deepscaler_4k_step_256")
    ap.add_argument("--fractions", default="0.10,0.25,0.50")
    ap.add_argument("--max-budget-ratio-error", type=float, default=0.30)
    args = ap.parse_args()

    if args.tokenizer.lower() == "whitespace":
        tok = None
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fracs = [float(x) for x in args.fractions.split(",") if x.strip()]
    rows = read_jsonl(args.groups)
    by_pid = {str(r["problem_id"]): r for r in rows}
    pairs = read_jsonl(args.pairs)
    pair_for_f = {p["forgotten_problem_id"]: p for p in pairs}
    pair_for_n = {p["never_problem_id"]: p for p in pairs}

    out = []
    audited = defaultdict(int)

    # Baseline exactly once per F/N/S problem used in the primary run.
    used_pids = set(pair_for_f) | set(pair_for_n) | {
        str(r["problem_id"]) for r in rows if r["group"] == "stable_correct"
    }
    for pid in sorted(used_pids):
        r = by_pid.get(pid)
        if not r or not r.get("prompt"):
            continue
        pair_id = None
        if pid in pair_for_f:
            pair_id = pair_for_f[pid]["pair_id"]
        elif pid in pair_for_n:
            pair_id = pair_for_n[pid]["pair_id"]
        out.append(
            {
                "request_id": f"{pid}__baseline",
                "problem_id": pid,
                "pair_id": pair_id,
                "split": (pair_for_f.get(pid) or pair_for_n.get(pid) or {}).get("split", "discovery" if stable_hash_int(pid) % 10 < 6 else "confirmation"),
                "group": r["group"],
                "source": "baseline",
                "prefix_fraction": 0.0,
                "prefix_tokens": 0,
                "prefix": "",
                "assistant_prefix": "",
                "problem": r.get("problem"),
                "prompt": r["prompt"].rstrip(),
                "gold_answer": r.get("gold_answer"),
            }
        )

    # Forgotten conditions use old-self prefix as the token-budget reference.
    for pid, pair in pair_for_f.items():
        r = by_pid[pid]
        old = r.get("old_correct_trace", "")
        if not old:
            continue
        sources = {
            "oldself": old,
            "other_correct": r.get("other_correct_trace", ""),
            "final_wrong": r.get("final_wrong_trace", ""),
        }
        for frac in fracs:
            old_prefix = prefix_from_fraction(old, frac)
            budget = n_tokens(tok, old_prefix)
            if budget <= 0:
                continue
            for source, trace in sources.items():
                if not trace:
                    continue
                pref = old_prefix if source == "oldself" else prefix_under_budget(tok, trace, budget)
                if not pref:
                    continue
                leaks = explicit_answer_leak_reasons(pref, r.get("gold_answer"))
                actual = n_tokens(tok, pref)
                ratio_err = abs(actual - budget) / max(1, budget)
                if leaks:
                    audited[f"rejected_leak_{source}"] += 1
                    continue
                if source != "oldself" and ratio_err > args.max_budget_ratio_error:
                    audited[f"rejected_length_{source}"] += 1
                    continue
                out.append(
                    {
                        "request_id": f"{pid}__{source}__{frac:.2f}",
                        "problem_id": pid,
                        "pair_id": pair["pair_id"],
                        "split": pair["split"],
                        "group": "forgotten",
                        "source": source,
                        "prefix_fraction": frac,
                        "target_prefix_tokens": budget,
                        "prefix_tokens": actual,
                        "prefix_budget_ratio_error": ratio_err,
                        "prefix": pref,
                        "assistant_prefix": pref.rstrip() + "\n",
                        "problem": r.get("problem"),
                        "prompt": r["prompt"].rstrip(),
                        "gold_answer": r.get("gold_answer"),
                    }
                )

            # N control gets the exact F token budget at the same fraction.
            n = by_pid.get(str(pair["never_problem_id"]))
            if n and n.get("verified_correct_trace"):
                pref = prefix_under_budget(tok, n["verified_correct_trace"], budget)
                if pref:
                    leaks = explicit_answer_leak_reasons(pref, n.get("gold_answer"))
                    actual = n_tokens(tok, pref)
                    ratio_err = abs(actual - budget) / max(1, budget)
                    if not leaks and ratio_err <= args.max_budget_ratio_error:
                        out.append(
                            {
                                "request_id": f"{n['problem_id']}__verified_correct__{frac:.2f}",
                                "problem_id": str(n["problem_id"]),
                                "pair_id": pair["pair_id"],
                                "split": pair["split"],
                                "group": "never_correct",
                                "source": "verified_correct",
                                "prefix_fraction": frac,
                                "target_prefix_tokens": budget,
                                "prefix_tokens": actual,
                                "prefix_budget_ratio_error": ratio_err,
                                "prefix": pref,
                                "assistant_prefix": pref.rstrip() + "\n",
                                "problem": n.get("problem"),
                                "prompt": n["prompt"].rstrip(),
                                "gold_answer": n.get("gold_answer"),
                            }
                        )
                    else:
                        audited["rejected_N_control"] += 1

    # Stable-correct positive controls use their own fraction, no matched F budget.
    for r in rows:
        if r["group"] != "stable_correct" or not r.get("old_correct_trace"):
            continue
        pid = str(r["problem_id"])
        for frac in fracs:
            pref = prefix_from_fraction(r["old_correct_trace"], frac)
            if not pref or explicit_answer_leak_reasons(pref, r.get("gold_answer")):
                continue
            out.append(
                {
                    "request_id": f"{pid}__oldself__{frac:.2f}",
                    "problem_id": pid,
                    "pair_id": None,
                    "split": "discovery" if stable_hash_int(pid) % 10 < 6 else "confirmation",
                    "group": "stable_correct",
                    "source": "oldself",
                    "prefix_fraction": frac,
                    "prefix_tokens": n_tokens(tok, pref),
                    "prefix": pref,
                    "assistant_prefix": pref.rstrip() + "\n",
                    "problem": r.get("problem"),
                    "prompt": r["prompt"].rstrip(),
                    "gold_answer": r.get("gold_answer"),
                }
            )

    write_jsonl(args.output, out)
    print(f"requests={len(out)} audits={dict(sorted(audited.items()))}")


if __name__ == "__main__":
    main()
