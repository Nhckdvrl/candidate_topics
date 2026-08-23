#!/usr/bin/env python3
"""Frozen G0 screen for the ChronoScope mechanism candidate.

This script does NOT run a model and does NOT search layers.  It consumes the JSON
produced by the official ChronoScope `source/hf_scope_benchmark.py` evaluator and
asks only whether the prerequisite failure cell is dense enough to justify any
mechanistic work.

Primary event (Gold Context only):

    the chain's first evaluated temporal turn is correct
    AND
    a later follow-up turn is wrong
    AND
    that wrong answer exactly/relaxed-matches the benchmark's present-day truth

The official benchmark calls the last condition `drift_top1`.  We additionally
require an earlier correct turn in the same chain so the event cannot be reduced
to "the model never knew the historical fact".

Recommended official run (make `max_bad_samples` larger than the number of turns
so the JSON contains a complete turn dump):

    python source/hf_scope_benchmark.py \
      --data merged_scope_benchmark.jsonl \
      --model Qwen/Qwen2.5-7B-Instruct \
      --out results/qwen25_7b_full.json \
      --max_chains 5000 \
      --self_max_chains 1 \
      --batch_size 64 \
      --max_new_tokens 24 \
      --dtype bfloat16 \
      --match_mode relaxed \
      --max_bad_samples 1000000 \
      --dump_examples

Then:

    python advisor_topic_search/g0/chronoscope_drift_g0.py \
      results/qwen25_7b_full.json \
      --out results/qwen25_7b_chronoscope_g0.json

This is intentionally a prerequisite screen, not evidence for a hidden-state
mechanism.  Promotion requires a substantial event density on at least one model
already included in the official benchmark, preferably replicated on a second
open model.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def gold_examples(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recover per-turn Gold-Context examples from either payload layout."""
    # Official --dump_examples currently places flattened examples here.
    flat = payload.get("examples") or []
    out = [x for x in flat if x.get("setting") == "gold"]
    if out:
        return out

    # Be permissive to older/newer payload layouts.
    res = payload.get("results", {}).get("gold", {})
    nested = res.get("examples") or res.get("bad_samples") or []
    return [dict(x, setting="gold") for x in nested]


def is_correct(ex: Dict[str, Any]) -> bool:
    # Official evaluator writes error_type="correct" for retained correct turns.
    if ex.get("error_type") == "correct":
        return True
    # A drift flag is always wrong by construction.
    if ex.get("drift_top1") is True:
        return False
    # Do not silently infer correctness from arbitrary strings.
    return False


def is_drift(ex: Dict[str, Any]) -> bool:
    return bool(ex.get("drift_top1")) or ex.get("error_type") == "drift"


def chain_key(ex: Dict[str, Any]) -> str:
    return str(ex.get("chain_id"))


def sorted_turns(xs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(xs, key=lambda x: int(x.get("turn_index", -1)))


def pct(x: int, n: int) -> float:
    return x / n if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--min_followup_pos",
        type=int,
        default=1,
        help="Only count drift at/after this official followup_pos (default: 1).",
    )
    ap.add_argument(
        "--warn_if_example_coverage_below",
        type=float,
        default=0.95,
        help="Warn if retained Gold examples cover less than this fraction of reported Gold turns.",
    )
    ap.add_argument("--max_cases", type=int, default=200)
    args = ap.parse_args()

    payload = load_json(args.result_json)
    examples = gold_examples(payload)
    if not examples:
        raise SystemExit(
            "No Gold-context per-turn examples found. Re-run the official evaluator "
            "with --dump_examples and a very large --max_bad_samples."
        )

    reported_turns = (
        payload.get("results", {})
        .get("gold", {})
        .get("overall", {})
        .get("n_turns")
    )
    coverage = None
    if isinstance(reported_turns, int) and reported_turns > 0:
        coverage = len(examples) / reported_turns
        if coverage < args.warn_if_example_coverage_below:
            raise SystemExit(
                f"Gold example coverage is only {coverage:.3f} "
                f"({len(examples)}/{reported_turns}). This screen requires an almost-complete "
                "turn dump; increase --max_bad_samples and re-run."
            )

    by_chain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        cid = chain_key(ex)
        if cid and cid != "None":
            by_chain[cid].append(ex)

    eligible_chains = 0
    chains_with_drift = 0
    total_later_turns = 0
    total_drift_events = 0
    first_to_drift_cases: List[Dict[str, Any]] = []

    fam_eligible = Counter()
    fam_drift_chains = Counter()
    fam_later_turns = Counter()
    fam_drift_events = Counter()
    pos_later_turns = Counter()
    pos_drift_events = Counter()

    # More stringent diagnostic: a drift event after at least one earlier correct
    # turn, not merely after an incorrect first turn.
    drift_after_any_prior_correct = 0
    later_turns_after_any_prior_correct = 0

    for cid, xs in by_chain.items():
        turns = sorted_turns(xs)
        if len(turns) < 2:
            continue

        first = turns[0]
        if not is_correct(first):
            continue

        fam = str(first.get("family", "unknown"))
        eligible_chains += 1
        fam_eligible[fam] += 1
        chain_has_drift = False
        seen_correct = True

        for ex in turns[1:]:
            pos = int(ex.get("followup_pos", 0) or 0)
            if pos < args.min_followup_pos:
                seen_correct = seen_correct or is_correct(ex)
                continue

            total_later_turns += 1
            fam_later_turns[str(ex.get("family", fam))] += 1
            pos_later_turns[pos] += 1

            if seen_correct:
                later_turns_after_any_prior_correct += 1

            if is_drift(ex):
                total_drift_events += 1
                chain_has_drift = True
                efam = str(ex.get("family", fam))
                fam_drift_events[efam] += 1
                pos_drift_events[pos] += 1
                if seen_correct:
                    drift_after_any_prior_correct += 1

                if len(first_to_drift_cases) < args.max_cases:
                    first_to_drift_cases.append(
                        {
                            "chain_id": cid,
                            "family": efam,
                            "first_turn": {
                                "turn_index": first.get("turn_index"),
                                "question": first.get("question"),
                                "gold": first.get("gold"),
                                "pred_top1": first.get("pred_top1"),
                            },
                            "drift_turn": {
                                "turn_index": ex.get("turn_index"),
                                "followup_pos": pos,
                                "question": ex.get("question"),
                                "historical_gold": ex.get("gold"),
                                "pred_top1": ex.get("pred_top1"),
                                "present_day": ex.get("present_day"),
                            },
                        }
                    )

            seen_correct = seen_correct or is_correct(ex)

        if chain_has_drift:
            chains_with_drift += 1
            fam_drift_chains[fam] += 1

    by_family = {}
    all_fams = sorted(set(fam_eligible) | set(fam_later_turns))
    for fam in all_fams:
        by_family[fam] = {
            "eligible_chains_first_turn_correct": fam_eligible[fam],
            "chains_with_present_drift": fam_drift_chains[fam],
            "chain_drift_rate": pct(fam_drift_chains[fam], fam_eligible[fam]),
            "later_turns": fam_later_turns[fam],
            "present_drift_events": fam_drift_events[fam],
            "turn_drift_rate": pct(fam_drift_events[fam], fam_later_turns[fam]),
        }

    by_followup_pos = {}
    for pos in sorted(pos_later_turns):
        by_followup_pos[str(pos)] = {
            "later_turns": pos_later_turns[pos],
            "present_drift_events": pos_drift_events[pos],
            "turn_drift_rate": pct(pos_drift_events[pos], pos_later_turns[pos]),
        }

    summary = {
        "model": payload.get("model"),
        "source_result": str(args.result_json),
        "reported_gold_turns": reported_turns,
        "retained_gold_examples": len(examples),
        "example_coverage": coverage,
        "primary_definition": (
            "Gold Context; chain starts with a retained correct temporal turn; "
            "later turn is wrong and matches benchmark present-day truth"
        ),
        "eligible_chains_first_turn_correct": eligible_chains,
        "chains_with_present_drift": chains_with_drift,
        "chain_drift_rate": pct(chains_with_drift, eligible_chains),
        "later_turns_after_first_correct": total_later_turns,
        "present_drift_events": total_drift_events,
        "turn_drift_rate": pct(total_drift_events, total_later_turns),
        "later_turns_after_any_prior_correct": later_turns_after_any_prior_correct,
        "drift_after_any_prior_correct": drift_after_any_prior_correct,
        "drift_rate_after_any_prior_correct": pct(
            drift_after_any_prior_correct, later_turns_after_any_prior_correct
        ),
        "by_family": by_family,
        "by_followup_pos": by_followup_pos,
        "cases": first_to_drift_cases,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"saved: {args.out}")
    else:
        print(text)

    print("\n=== ChronoScope frozen prerequisite ===")
    print("model:", summary["model"])
    print(
        "eligible chains:", eligible_chains,
        "chains with present drift:", chains_with_drift,
        "rate:", f"{summary['chain_drift_rate']:.4f}",
    )
    print(
        "later turns:", total_later_turns,
        "present-drift events:", total_drift_events,
        "rate:", f"{summary['turn_drift_rate']:.4f}",
    )


if __name__ == "__main__":
    main()
