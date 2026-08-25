#!/usr/bin/env python3
"""Frozen descriptive analysis for Topic 28 truthful-clue reversals.

No model inference, classifier, LLM judge, alternate scorer, or outcome-based
selection is used. The script reconstructs the same cleaned G0 trajectories,
then compares all official-score 1->0 boundaries with all eligible 1->1
boundaries using deterministic text and trajectory features.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import g0_progressive_reversal as g0


ANALYSIS_SEED = 20260825
BOOTSTRAP_REPS = 2000

TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
QUOTE_RE = re.compile(r"[\"“”‘’]|(?:^|\s)'[^']+'")
PAREN_RE = re.compile(r"\([^)]{2,}\)")
CAP_SPAN_RE = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b"
)

STOPWORDS = {
    "about", "after", "again", "against", "all", "also", "among", "and",
    "any", "are", "because", "been", "before", "being", "between", "both",
    "but", "can", "did", "does", "during", "each", "for", "from", "had",
    "has", "have", "her", "here", "him", "his", "how", "into", "its",
    "may", "more", "most", "not", "one", "only", "other", "over", "same",
    "she", "some", "such", "than", "that", "their", "them", "then", "these",
    "they", "this", "through", "under", "was", "were", "what", "when",
    "where", "which", "while", "who", "whose", "will", "with", "would",
}

CAPITALIZED_EXCLUSIONS = {
    "A", "According", "After", "An", "Another", "As", "At", "Before",
    "During", "For", "He", "Her", "His", "In", "It", "Its", "One", "On",
    "She", "That", "The", "Their", "These", "They", "This", "Those",
}

PRIMARY_BINARY_FEATURES = [
    "introduced_name_any",
    "has_year",
    "has_number",
    "has_quote",
    "has_parenthetical",
    "gold_exact_in_new",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/analysis1"))
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


def text_tokens(value: object) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(g0.normalize_answer(value)))


def content_tokens(value: object) -> tuple[str, ...]:
    return tuple(t for t in text_tokens(value) if len(t) >= 3 and t not in STOPWORDS)


def contains_token_sequence(text: object, phrase: object) -> bool:
    hay = text_tokens(text)
    needle = text_tokens(phrase)
    if not needle or len(needle) > len(hay):
        return False
    n = len(needle)
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def capitalized_spans(text: object) -> tuple[str, ...]:
    raw = "" if text is None else str(text)
    spans = []
    for m in CAP_SPAN_RE.finditer(raw):
        surface = m.group(0).strip()
        if surface in CAPITALIZED_EXCLUSIONS:
            continue
        norm = g0.normalize_answer(surface)
        if norm:
            spans.append(norm)
    return tuple(sorted(set(spans)))


def introduced_capitalized_spans(new_clue: object, before_text: object) -> tuple[str, ...]:
    before = set(capitalized_spans(before_text))
    return tuple(x for x in capitalized_spans(new_clue) if x not in before)


def token_coverage(answer: object, clue: object) -> float:
    toks = set(content_tokens(answer))
    if not toks:
        return 0.0
    clue_toks = set(content_tokens(clue))
    return float(len(toks & clue_toks) / len(toks))


def max_alias_coverage(aliases: Iterable[str], clue: object) -> float:
    vals = [token_coverage(a, clue) for a in aliases]
    return float(max(vals)) if vals else 0.0


def any_alias_exact(aliases: Iterable[str], clue: object) -> bool:
    return any(contains_token_sequence(clue, a) for a in aliases if text_tokens(a))


def build_atomic_clue_idf(questions: pd.DataFrame) -> dict[str, float]:
    docs: list[set[str]] = []
    for row in questions.itertuples(index=False):
        clue = g0.extract_added_clue(row.full_quiz_question, row.clue_spans, int(row.n_clues))
        docs.append(set(content_tokens(clue)))
    n = len(docs)
    freq: Counter[str] = Counter()
    for toks in docs:
        freq.update(toks)
    return {tok: float(math.log((n + 1) / (df + 1)) + 1.0) for tok, df in freq.items()}


def clue_specificity(text: object, idf: dict[str, float]) -> float:
    toks = content_tokens(text)
    if not toks:
        return 0.0
    default = float(math.log(len(idf) + 1) + 1.0)
    return float(np.mean([idf.get(t, default) for t in toks]))


def add_first_future_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Add first later correct clue and lag, preserving trajectory boundaries."""
    out = df.copy()
    first_future = np.full(len(out), np.nan, dtype=float)
    keys = ["config", "agent_type", "qid"]
    for _, idx in out.groupby(keys, sort=False).indices.items():
        positions = np.asarray(idx, dtype=int)
        next_correct_clue: float | None = None
        for pos in positions[::-1]:
            if next_correct_clue is not None:
                first_future[pos] = next_correct_clue
            if int(out.at[pos, "correct"]) == 1:
                next_correct_clue = float(out.at[pos, "clue_idx"])
    out["first_future_correct_clue"] = first_future
    out["recovery_lag_clues"] = out["first_future_correct_clue"] - out["clue_idx"]
    return out


def build_transition_table(clean: pd.DataFrame, idf: dict[str, float]) -> pd.DataFrame:
    keys = ["config", "agent_type", "qid"]
    df = clean.sort_values(keys + ["clue_idx"]).reset_index(drop=True).copy()
    grouped = df.groupby(keys, sort=False)

    for col in ["clue_idx", "correct", "prediction", "strict_alias_correct"]:
        df[f"prev_{col}"] = grouped[col].shift(1)
    df["prev_clue_text"] = grouped["clue_text"].shift(1)
    for col in ["clue_idx", "correct"]:
        df[f"next_{col}"] = grouped[col].shift(-1)
    df["final_correct"] = grouped["correct"].transform("last").astype(int)
    df = add_first_future_recovery(df)

    df["delta"] = df["clue_idx"] - df["prev_clue_idx"]
    eligible = df[(df["delta"] == 1) & (df["prev_correct"] == 1)].copy()
    eligible["reversal"] = (eligible["correct"] == 0).astype(int)
    eligible["new_clue_text"] = [
        g0.extract_added_clue(full, spans, int(idx))
        for full, spans, idx in zip(
            eligible["full_quiz_question"], eligible["clue_spans"], eligible["clue_idx"]
        )
    ]
    if eligible["new_clue_text"].isna().any():
        raise RuntimeError("Eligible transition has no extractable official new clue")

    eligible["total_clues"] = eligible["clue_spans"].map(len).astype(int)
    eligible["relative_arrival"] = eligible["clue_idx"] / eligible["total_clues"]
    eligible["relative_stage"] = pd.cut(
        eligible["relative_arrival"],
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["early", "middle", "late"],
        include_lowest=True,
    ).astype(str)
    eligible["immediate_recovery"] = (
        (eligible["next_clue_idx"] == eligible["clue_idx"] + 1)
        & (eligible["next_correct"] == 1)
    )
    eligible["eventual_recovery"] = eligible["first_future_correct_clue"].notna()
    eligible["final_state_correct"] = eligible["final_correct"] == 1

    features = [
        structural_features(new, before, aliases, idf)
        for new, before, aliases in zip(
            eligible["new_clue_text"], eligible["prev_clue_text"], eligible["alias_norms"]
        )
    ]
    feat_df = pd.DataFrame(features, index=eligible.index)
    eligible = pd.concat([eligible, feat_df], axis=1)

    eligible["word_count_bin"] = pd.qcut(
        eligible["new_clue_word_count"], q=4, duplicates="drop"
    ).astype(str)
    eligible["specificity_bin"] = pd.qcut(
        eligible["new_clue_specificity"], q=4, duplicates="drop"
    ).astype(str)
    eligible["introduced_name_bin"] = pd.cut(
        eligible["introduced_name_count"],
        bins=[-0.1, 0.5, 1.5, np.inf],
        labels=["0", "1", "2+"],
    ).astype(str)
    return eligible.reset_index(drop=True)


def structural_features(
    new_clue: object,
    before_text: object,
    aliases: Iterable[str],
    idf: dict[str, float],
) -> dict:
    introduced = introduced_capitalized_spans(new_clue, before_text)
    raw = str(new_clue)
    return {
        "new_clue_word_count": len(text_tokens(new_clue)),
        "new_clue_specificity": clue_specificity(new_clue, idf),
        "introduced_name_count": len(introduced),
        "introduced_name_any": bool(introduced),
        "introduced_name_surfaces": json.dumps(list(introduced), ensure_ascii=False),
        "has_year": bool(YEAR_RE.search(raw)),
        "has_number": bool(NUMBER_RE.search(raw)),
        "has_quote": bool(QUOTE_RE.search(raw)),
        "has_parenthetical": bool(PAREN_RE.search(raw)),
        "gold_exact_in_new": any_alias_exact(aliases, new_clue),
        "gold_content_coverage_new": max_alias_coverage(aliases, new_clue),
    }


def add_competitor_features(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    wrong_exact_new = []
    wrong_exact_before = []
    newly_shared = []
    wrong_cov = []
    advantage = []
    for row in ev.itertuples(index=False):
        w_new = contains_token_sequence(row.new_clue_text, row.prediction)
        w_before = contains_token_sequence(row.prev_clue_text, row.prediction)
        wrong_tokens = set(content_tokens(row.prediction))
        new_tokens = set(content_tokens(row.new_clue_text))
        before_tokens = set(content_tokens(row.prev_clue_text))
        new_shared = sorted((wrong_tokens & new_tokens) - before_tokens)
        wcov = token_coverage(row.prediction, row.new_clue_text)
        wrong_exact_new.append(w_new)
        wrong_exact_before.append(w_before)
        newly_shared.append(json.dumps(new_shared, ensure_ascii=False))
        wrong_cov.append(wcov)
        advantage.append(wcov - float(row.gold_content_coverage_new))
    ev["wrong_exact_in_new"] = wrong_exact_new
    ev["wrong_exact_in_before"] = wrong_exact_before
    ev["wrong_newly_exact_in_new"] = ev["wrong_exact_in_new"] & ~ev["wrong_exact_in_before"]
    ev["wrong_new_content_tokens"] = newly_shared
    ev["wrong_new_content_token_any"] = ev["wrong_new_content_tokens"] != "[]"
    ev["wrong_content_coverage_new"] = wrong_cov
    ev["wrong_minus_gold_coverage"] = advantage
    return ev


def grouped_rate(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        df.groupby(columns, dropna=False, observed=False)
        .agg(n=("reversal", "size"), reversals=("reversal", "sum"))
        .reset_index()
        .assign(reversal_rate=lambda x: x["reversals"] / x["n"])
    )


def cluster_bootstrap_binary_difference(
    df: pd.DataFrame,
    feature: str,
    reps: int = BOOTSTRAP_REPS,
    seed: int = ANALYSIS_SEED,
) -> tuple[float, float, float]:
    work = df[["qid", feature, "reversal"]].copy()
    work[feature] = work[feature].astype(bool)
    obs = work.groupby(feature)["reversal"].mean()
    if True not in obs.index or False not in obs.index:
        return float("nan"), float("nan"), float("nan")
    observed = float(obs[True] - obs[False])

    counts = (
        work.groupby(["qid", feature], observed=False)["reversal"]
        .agg(["sum", "count"])
        .reset_index()
    )
    qids = np.asarray(sorted(work["qid"].unique()))
    by_qid = {qid: g for qid, g in counts.groupby("qid", sort=False)}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        sampled = rng.choice(qids, size=len(qids), replace=True)
        totals = {False: [0, 0], True: [0, 0]}
        for qid in sampled:
            for _, raw_flag, n_reversals, n_total in by_qid[qid].itertuples(
                index=False, name=None
            ):
                flag = bool(raw_flag)
                totals[flag][0] += int(n_reversals)
                totals[flag][1] += int(n_total)
        if totals[True][1] and totals[False][1]:
            vals.append(
                totals[True][0] / totals[True][1]
                - totals[False][0] / totals[False][1]
            )
    lo, hi = np.quantile(np.asarray(vals, dtype=float), [0.025, 0.975])
    return observed, float(lo), float(hi)


def trigger_feature_summary(eligible: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in PRIMARY_BINARY_FEATURES:
        tab = grouped_rate(eligible, [feature])
        observed, lo, hi = cluster_bootstrap_binary_difference(eligible, feature)
        for row in tab.itertuples(index=False):
            rows.append(
                {
                    "feature": feature,
                    "value": bool(getattr(row, feature)),
                    "n": int(row.n),
                    "reversals": int(row.reversals),
                    "reversal_rate": float(row.reversal_rate),
                    "true_minus_false_rate": observed,
                    "cluster_bootstrap_95ci_low": lo,
                    "cluster_bootstrap_95ci_high": hi,
                }
            )
    for feature in ["word_count_bin", "specificity_bin", "introduced_name_bin"]:
        tab = grouped_rate(eligible, [feature])
        for row in tab.itertuples(index=False):
            rows.append(
                {
                    "feature": feature,
                    "value": str(getattr(row, feature)),
                    "n": int(row.n),
                    "reversals": int(row.reversals),
                    "reversal_rate": float(row.reversal_rate),
                    "true_minus_false_rate": np.nan,
                    "cluster_bootstrap_95ci_low": np.nan,
                    "cluster_bootstrap_95ci_high": np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    response_revision = g0.resolve_revision(g0.RESPONSES_DATASET)
    question_revision = g0.resolve_revision(g0.QUESTIONS_DATASET)
    questions = g0.load_questions(question_revision, args.cache_dir)
    responses, configs, n_available = g0.load_responses(
        response_revision, None, args.cache_dir
    )
    if len(configs) != n_available:
        raise RuntimeError("Frozen analysis requires every available response config")
    clean, audit = g0.clean_and_join(responses, questions, include_human=False)

    idf = build_atomic_clue_idf(questions)
    eligible = build_transition_table(clean, idf)
    if int(eligible["reversal"].sum()) != 8102:
        raise RuntimeError(
            "Analysis population does not reproduce frozen G0 reversal count: "
            f"{int(eligible['reversal'].sum())}"
        )
    if len(eligible) != 120353:
        raise RuntimeError(
            "Analysis population does not reproduce frozen G0 eligible count: "
            f"{len(eligible)}"
        )

    events = add_competitor_features(eligible[eligible["reversal"] == 1].copy())
    trigger = trigger_feature_summary(eligible)
    by_to = grouped_rate(eligible, ["clue_idx"])
    by_stage = grouped_rate(eligible, ["relative_stage"])
    by_config = grouped_rate(eligible, ["config"])
    by_category = grouped_rate(eligible, ["category"])

    recovery = {
        "events": int(len(events)),
        "immediate_recovery_events": int(events["immediate_recovery"].sum()),
        "immediate_recovery_rate": float(events["immediate_recovery"].mean()),
        "eventual_recovery_events": int(events["eventual_recovery"].sum()),
        "eventual_recovery_rate": float(events["eventual_recovery"].mean()),
        "final_state_correct_events": int(events["final_state_correct"].sum()),
        "final_state_correct_rate": float(events["final_state_correct"].mean()),
        "never_recovered_events": int((~events["eventual_recovery"]).sum()),
        "median_recovery_lag_clues": (
            float(events.loc[events["eventual_recovery"], "recovery_lag_clues"].median())
            if events["eventual_recovery"].any()
            else None
        ),
    }
    competitor = {
        "wrong_exact_in_new": int(events["wrong_exact_in_new"].sum()),
        "wrong_newly_exact_in_new": int(events["wrong_newly_exact_in_new"].sum()),
        "wrong_new_content_token_any": int(events["wrong_new_content_token_any"].sum()),
        "gold_exact_in_new": int(events["gold_exact_in_new"].sum()),
        "wrong_coverage_exceeds_gold": int(
            (events["wrong_minus_gold_coverage"] > 0).sum()
        ),
        "mean_wrong_content_coverage_new": float(events["wrong_content_coverage_new"].mean()),
        "mean_gold_content_coverage_new": float(events["gold_content_coverage_new"].mean()),
    }

    summary = {
        "topic": 28,
        "analysis": "frozen descriptive reversal structure",
        "eligible_transitions": int(len(eligible)),
        "reversal_events": int(events.shape[0]),
        "reversal_rate": float(eligible["reversal"].mean()),
        "competitor_introduction": competitor,
        "recovery_dynamics": recovery,
        "limitations": [
            "lexical competitor metrics are high-precision lower bounds, not semantic judgments",
            "descriptive associations are not causal effects",
            "released trajectories predate a controlled clue-order intervention",
        ],
    }
    receipt = {
        "responses_revision": response_revision,
        "questions_revision": question_revision,
        "n_available_configs": n_available,
        "n_loaded_configs": len(configs),
        "human_included": False,
        "analysis_seed": ANALYSIS_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "g0_cleaning_audit": audit,
    }

    with open(args.out_dir / "analysis_receipt.json", "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    with open(args.out_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    eligible.to_csv(args.out_dir / "eligible_transition_features.csv", index=False)
    events.to_csv(args.out_dir / "reversal_event_dynamics.csv", index=False)
    trigger.to_csv(args.out_dir / "trigger_feature_summary.csv", index=False)
    by_to.to_csv(args.out_dir / "rates_by_to_clue.csv", index=False)
    by_stage.to_csv(args.out_dir / "rates_by_relative_stage.csv", index=False)
    by_config.to_csv(args.out_dir / "rates_by_config.csv", index=False)
    by_category.to_csv(args.out_dir / "rates_by_category.csv", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote analysis artifacts to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
