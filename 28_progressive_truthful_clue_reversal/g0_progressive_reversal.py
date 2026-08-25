#!/usr/bin/env python3
"""Frozen G0 for Topic 28: truthful-clue answer reversals.

This script performs no model inference. It reconstructs cumulative QuizBowl
trajectories from the public CAIMIRA artifacts and measures consecutive
correct->wrong transitions after adding one official clue.

Primary correctness comes from the released `score` field. A strict normalized
alias match against `clean_answers` is computed only as a high-precision sanity
check; it never replaces the released score.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

RESPONSES_DATASET = "mgor/protobowl-11-13-agent-responses"
QUESTIONS_DATASET = "mgor/protobowl-11-13"
QUESTIONS_CONFIG = "progressive-clues"
QUESTIONS_SPLIT = "eval"
RESPONSES_SPLIT = "train"

QC_RE = re.compile(r"^(q\d+)_(\d+)$")
ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)

# Frozen paper-worthiness / support gates. These are deliberately not CLI knobs.
MIN_JOIN_COVERAGE = 0.98
MIN_ELIGIBLE_CORRECT_TRANSITIONS = 500
MIN_REVERSAL_EVENTS = 100
MIN_REVERSAL_RATE = 0.02
MIN_REVERSAL_QUESTIONS = 50
MIN_REVERSAL_CONFIGS = 5
MIN_STRICT_ALIAS_EVENTS = 30
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260825


@dataclass(frozen=True)
class DatasetReceipt:
    repo_id: str
    revision: str
    split: str
    config: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/g0"),
        help="Output directory. Generated artifacts are not intended for git.",
    )
    p.add_argument(
        "--response-configs",
        nargs="*",
        default=None,
        help="Optional exact response configs for engineering/debug runs only.",
    )
    p.add_argument(
        "--include-human",
        action="store_true",
        help=(
            "Engineering/debug only. Human rows were backfilled under a "
            "monotonicity assumption and are invalid for the scientific G0."
        ),
    )
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


def resolve_revision(repo_id: str) -> str:
    info = HfApi().dataset_info(repo_id=repo_id)
    if not info.sha:
        raise RuntimeError(f"Hugging Face returned no immutable SHA for {repo_id}")
    return str(info.sha)


def parse_qc_id(value: object) -> tuple[str | None, int | None]:
    m = QC_RE.fullmatch(str(value))
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def normalize_answer(text: object) -> str:
    """Frozen SQuAD-style normalization used only for strict alias sanity checks."""
    s = unicodedata.normalize("NFKC", "" if text is None else str(text)).lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def normalize_aliases(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        raw = list(value)
    else:
        raw = [value]
    aliases = sorted({normalize_answer(x) for x in raw if normalize_answer(x)})
    return tuple(aliases)


def alias_exact(prediction: object, aliases: Iterable[str]) -> bool:
    p = normalize_answer(prediction)
    return bool(p) and p in set(aliases)


def extract_added_clue(full_question: object, clue_spans: object, clue_idx: int) -> str | None:
    """Return the newly added clue for 1-based clue_idx using official spans."""
    if not isinstance(full_question, str):
        return None
    if not isinstance(clue_spans, (list, tuple, np.ndarray)):
        return None
    pos = clue_idx - 1
    if pos < 0 or pos >= len(clue_spans):
        return None
    span = clue_spans[pos]
    if not isinstance(span, (list, tuple, np.ndarray)) or len(span) != 2:
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= start <= end <= len(full_question)):
        return None
    return full_question[start:end].strip()


def load_questions(revision: str, cache_dir: Path | None) -> pd.DataFrame:
    ds = load_dataset(
        QUESTIONS_DATASET,
        QUESTIONS_CONFIG,
        split=QUESTIONS_SPLIT,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    required = {
        "qc_id",
        "clue_text",
        "n_clues",
        "clean_answers",
        "orig_qid",
        "full_quiz_question",
        "clue_spans",
        "orig_answer_string",
        "metadata",
    }
    missing = required.difference(ds.column_names)
    if missing:
        raise RuntimeError(f"Question artifact missing required columns: {sorted(missing)}")
    df = ds.select_columns(sorted(required)).to_pandas()

    if df["qc_id"].duplicated().any():
        dup = df.loc[df["qc_id"].duplicated(keep=False), "qc_id"].astype(str).unique()
        raise RuntimeError(f"Question artifact has duplicate qc_id values, e.g. {dup[:5].tolist()}")

    parsed = df["qc_id"].map(parse_qc_id)
    df["parsed_qid"] = [x[0] for x in parsed]
    df["parsed_clue_idx"] = [x[1] for x in parsed]
    bad = (
        df["parsed_qid"].isna()
        | df["parsed_clue_idx"].isna()
        | (df["orig_qid"].astype(str) != df["parsed_qid"].astype(str))
        | (pd.to_numeric(df["n_clues"], errors="coerce") != df["parsed_clue_idx"])
    )
    if bad.any():
        examples = df.loc[bad, ["qc_id", "orig_qid", "n_clues"]].head(5).to_dict("records")
        raise RuntimeError(f"Question qc_id metadata contract failed, examples={examples}")

    df["alias_norms"] = df["clean_answers"].map(normalize_aliases)
    if (df["alias_norms"].map(len) == 0).any():
        n = int((df["alias_norms"].map(len) == 0).sum())
        raise RuntimeError(f"{n} question rows have no usable clean_answers aliases")

    keep = [
        "qc_id",
        "clue_text",
        "n_clues",
        "clean_answers",
        "alias_norms",
        "orig_qid",
        "full_quiz_question",
        "clue_spans",
        "orig_answer_string",
        "metadata",
    ]
    return df[keep].copy()


def load_responses(
    revision: str,
    configs: list[str] | None,
    cache_dir: Path | None,
) -> tuple[pd.DataFrame, list[str], int]:
    all_configs = get_dataset_config_names(RESPONSES_DATASET, revision=revision)
    if configs is None:
        selected = list(all_configs)
    else:
        unknown = sorted(set(configs).difference(all_configs))
        if unknown:
            raise RuntimeError(f"Unknown response configs: {unknown}")
        selected = list(configs)

    frames: list[pd.DataFrame] = []
    required = {"agent_type", "qc_id", "answer", "prediction", "score"}
    for i, config in enumerate(selected, 1):
        print(f"[responses {i}/{len(selected)}] {config}", flush=True)
        ds = load_dataset(
            RESPONSES_DATASET,
            config,
            split=RESPONSES_SPLIT,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        missing = required.difference(ds.column_names)
        if missing:
            raise RuntimeError(f"Config {config} missing columns: {sorted(missing)}")
        part = ds.select_columns(sorted(required)).to_pandas()
        part["config"] = config
        frames.append(part)

    if not frames:
        raise RuntimeError("No response configs loaded")
    return pd.concat(frames, ignore_index=True), selected, len(all_configs)


def clean_and_join(
    responses: pd.DataFrame,
    questions: pd.DataFrame,
    include_human: bool,
) -> tuple[pd.DataFrame, dict]:
    raw_rows = len(responses)
    df = responses.copy()

    parsed = df["qc_id"].map(parse_qc_id)
    df["qid"] = [x[0] for x in parsed]
    df["clue_idx"] = [x[1] for x in parsed]
    bad_qc = df["qid"].isna() | df["clue_idx"].isna()
    n_bad_qc = int(bad_qc.sum())
    df = df.loc[~bad_qc].copy()
    df["clue_idx"] = df["clue_idx"].astype(int)

    score_num = pd.to_numeric(df["score"], errors="coerce")
    bad_score = ~score_num.isin([0.0, 1.0])
    n_bad_score = int(bad_score.sum())
    df = df.loc[~bad_score].copy()
    df["correct"] = score_num.loc[df.index].astype(int)

    human_mask = df["agent_type"].astype(str).str.lower().str.contains("human", regex=False)
    n_human = int(human_mask.sum())
    if not include_human:
        df = df.loc[~human_mask].copy()

    # Join exact cumulative-clue metadata.
    qcols = [
        "qc_id",
        "clue_text",
        "n_clues",
        "clean_answers",
        "alias_norms",
        "orig_qid",
        "full_quiz_question",
        "clue_spans",
        "orig_answer_string",
        "metadata",
    ]
    df = df.merge(questions[qcols], on="qc_id", how="left", validate="many_to_one")
    joined = df["orig_qid"].notna()
    join_coverage = float(joined.mean()) if len(df) else 0.0
    df = df.loc[joined].copy()

    # The parsed qid/clue index must agree with the official progressive item.
    contract_bad = (
        df["orig_qid"].astype(str) != df["qid"].astype(str)
    ) | (pd.to_numeric(df["n_clues"], errors="coerce") != df["clue_idx"])
    n_contract_bad = int(contract_bad.sum())
    df = df.loc[~contract_bad].copy()

    # Drop ambiguous trajectory cells. Identical duplicates are safe to collapse.
    df["prediction_norm"] = df["prediction"].map(normalize_answer)
    key = ["config", "agent_type", "qid", "clue_idx"]
    conflicts = (
        df.groupby(key, dropna=False)
        .agg(
            n_correct=("correct", "nunique"),
            n_prediction=("prediction_norm", "nunique"),
        )
        .reset_index()
    )
    conflicts = conflicts[(conflicts["n_correct"] > 1) | (conflicts["n_prediction"] > 1)]
    n_conflict_cells = len(conflicts)
    if n_conflict_cells:
        bad_keys = set(map(tuple, conflicts[key].itertuples(index=False, name=None)))
        keep = [tuple(x) not in bad_keys for x in df[key].itertuples(index=False, name=None)]
        df = df.loc[keep].copy()

    df = df.sort_values(key).drop_duplicates(key, keep="first").copy()
    df["strict_alias_correct"] = [
        alias_exact(pred, aliases)
        for pred, aliases in zip(df["prediction"], df["alias_norms"])
    ]

    audit = {
        "raw_response_rows": int(raw_rows),
        "bad_qc_rows_dropped": n_bad_qc,
        "nonbinary_score_rows_dropped": n_bad_score,
        "human_rows_seen": n_human,
        "human_rows_included": bool(include_human),
        "question_join_coverage_before_contract_filter": join_coverage,
        "metadata_contract_rows_dropped": n_contract_bad,
        "ambiguous_duplicate_cells_dropped": int(n_conflict_cells),
        "clean_joined_rows": int(len(df)),
    }
    return df, audit


def analyze_transitions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    group_cols = ["config", "agent_type", "qid"]
    events: list[dict] = []
    trajectories: list[dict] = []
    transition_counter: Counter[str] = Counter()
    n_gap_pairs = 0

    for keys, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("clue_idx").reset_index(drop=True)
        if len(g) < 2:
            continue

        n_primary_reversals = 0
        n_strict_reversals = 0
        ever_correct = bool((g["correct"] == 1).any())
        recovered_immediately = False
        recovered_eventually = False

        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            cur = g.iloc[i]
            delta = int(cur["clue_idx"] - prev["clue_idx"])
            if delta != 1:
                n_gap_pairs += 1
                continue

            pair = f"{int(prev['correct'])}->{int(cur['correct'])}"
            transition_counter[pair] += 1

            if int(prev["correct"]) == 1 and int(cur["correct"]) == 0:
                n_primary_reversals += 1
                strict = bool(prev["strict_alias_correct"] and not cur["strict_alias_correct"])
                n_strict_reversals += int(strict)

                if i + 1 < len(g):
                    nxt = g.iloc[i + 1]
                    if int(nxt["clue_idx"] - cur["clue_idx"]) == 1 and int(nxt["correct"]) == 1:
                        recovered_immediately = True
                if (g.loc[i + 1 :, "correct"] == 1).any():
                    recovered_eventually = True

                added = extract_added_clue(
                    cur["full_quiz_question"], cur["clue_spans"], int(cur["clue_idx"])
                )
                events.append(
                    {
                        "config": keys[0],
                        "agent_type": keys[1],
                        "qid": keys[2],
                        "from_clue": int(prev["clue_idx"]),
                        "to_clue": int(cur["clue_idx"]),
                        "prediction_before": prev["prediction"],
                        "prediction_after": cur["prediction"],
                        "official_answer": cur["answer"],
                        "orig_answer_string": cur["orig_answer_string"],
                        "clean_answers": json.dumps(list(cur["clean_answers"]), ensure_ascii=False),
                        "strict_alias_stable": strict,
                        "cumulative_text_before": prev["clue_text"],
                        "cumulative_text_after": cur["clue_text"],
                        "new_clue_text": added,
                        "category": (
                            cur["metadata"].get("category")
                            if isinstance(cur["metadata"], dict)
                            else None
                        ),
                        "subcategory": (
                            cur["metadata"].get("subcategory")
                            if isinstance(cur["metadata"], dict)
                            else None
                        ),
                    }
                )

        trajectories.append(
            {
                "config": keys[0],
                "agent_type": keys[1],
                "qid": keys[2],
                "n_states": int(len(g)),
                "ever_correct": ever_correct,
                "n_reversals": int(n_primary_reversals),
                "has_reversal": bool(n_primary_reversals),
                "n_strict_alias_reversals": int(n_strict_reversals),
                "recovered_immediately": recovered_immediately,
                "recovered_eventually": recovered_eventually,
            }
        )

    traj = pd.DataFrame(trajectories)
    ev = pd.DataFrame(events)

    cc = transition_counter["1->1"]
    cw = transition_counter["1->0"]
    eligible_from_correct = cc + cw
    reversal_rate = float(cw / eligible_from_correct) if eligible_from_correct else 0.0

    counts = {
        "transition_counts": dict(transition_counter),
        "eligible_consecutive_transitions_from_correct": int(eligible_from_correct),
        "official_reversal_events": int(cw),
        "reversal_rate_given_current_correct": reversal_rate,
        "gap_pairs_excluded_from_primary": int(n_gap_pairs),
    }
    return traj, ev, counts


def grouped_summary(traj: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if traj.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in traj.groupby(by, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        payload = dict(zip(by, keys))
        payload.update(
            {
                "n_trajectories": int(len(g)),
                "n_ever_correct": int(g["ever_correct"].sum()),
                "n_reversal_questions": int(g["has_reversal"].sum()),
                "n_reversal_events": int(g["n_reversals"].sum()),
                "n_strict_alias_reversal_events": int(g["n_strict_alias_reversals"].sum()),
                "n_immediate_recovery_questions": int(g["recovered_immediately"].sum()),
                "n_eventual_recovery_questions": int(g["recovered_eventually"].sum()),
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def clustered_bootstrap_rate(
    df: pd.DataFrame, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED
) -> tuple[float | None, float | None]:
    """Bootstrap reversal rate by whole trajectory, preserving within-trajectory pairs."""
    group_cols = ["config", "agent_type", "qid"]
    units: list[tuple[int, int]] = []
    for _, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("clue_idx")
        cc = cw = 0
        rows = list(g.itertuples(index=False))
        for prev, cur in zip(rows[:-1], rows[1:]):
            if int(cur.clue_idx) - int(prev.clue_idx) != 1:
                continue
            if int(prev.correct) == 1:
                if int(cur.correct) == 0:
                    cw += 1
                else:
                    cc += 1
        if cc + cw:
            units.append((cw, cc + cw))
    if not units:
        return None, None

    arr = np.asarray(units, dtype=np.int64)
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    n = len(arr)
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        sample = arr[idx]
        denom = sample[:, 1].sum()
        vals[r] = sample[:, 0].sum() / denom if denom else np.nan
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None, None
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_gates(
    audit: dict,
    counts: dict,
    events: pd.DataFrame,
    include_human: bool,
    debug_subset: bool,
) -> tuple[str, dict]:
    if include_human:
        return "DEBUG_ONLY_INCLUDE_HUMAN", {
            "debug_only": {"pass": False, "observed": True, "required": False}
        }
    if debug_subset:
        return "DEBUG_ONLY_CONFIG_SUBSET", {
            "debug_only": {"pass": False, "observed": True, "required": "all response configs"}
        }

    n_events = int(counts["official_reversal_events"])
    n_qids = int(events["qid"].nunique()) if not events.empty else 0
    n_configs = int(events["config"].nunique()) if not events.empty else 0
    n_strict = int(events["strict_alias_stable"].sum()) if not events.empty else 0

    artifact_gates = {
        "question_join_coverage": {
            "observed": float(audit["question_join_coverage_before_contract_filter"]),
            "required": f">={MIN_JOIN_COVERAGE}",
            "pass": float(audit["question_join_coverage_before_contract_filter"]) >= MIN_JOIN_COVERAGE,
        },
        "clean_joined_rows_positive": {
            "observed": int(audit["clean_joined_rows"]),
            "required": ">0",
            "pass": int(audit["clean_joined_rows"]) > 0,
        },
    }
    if not all(x["pass"] for x in artifact_gates.values()):
        return "STOP_ARTIFACT_CONTRACT", artifact_gates

    scientific_gates = {
        "eligible_correct_transitions": {
            "observed": int(counts["eligible_consecutive_transitions_from_correct"]),
            "required": f">={MIN_ELIGIBLE_CORRECT_TRANSITIONS}",
            "pass": int(counts["eligible_consecutive_transitions_from_correct"])
            >= MIN_ELIGIBLE_CORRECT_TRANSITIONS,
        },
        "official_reversal_events": {
            "observed": n_events,
            "required": f">={MIN_REVERSAL_EVENTS}",
            "pass": n_events >= MIN_REVERSAL_EVENTS,
        },
        "reversal_rate": {
            "observed": float(counts["reversal_rate_given_current_correct"]),
            "required": f">={MIN_REVERSAL_RATE}",
            "pass": float(counts["reversal_rate_given_current_correct"]) >= MIN_REVERSAL_RATE,
        },
        "unique_reversal_questions": {
            "observed": n_qids,
            "required": f">={MIN_REVERSAL_QUESTIONS}",
            "pass": n_qids >= MIN_REVERSAL_QUESTIONS,
        },
        "unique_reversal_configs": {
            "observed": n_configs,
            "required": f">={MIN_REVERSAL_CONFIGS}",
            "pass": n_configs >= MIN_REVERSAL_CONFIGS,
        },
        "strict_alias_reversal_events": {
            "observed": n_strict,
            "required": f">={MIN_STRICT_ALIAS_EVENTS}",
            "pass": n_strict >= MIN_STRICT_ALIAS_EVENTS,
        },
    }
    verdict = (
        "GO_REVERSAL_OBJECT"
        if all(x["pass"] for x in scientific_gates.values())
        else "STOP_REVERSAL_OBJECT"
    )
    return verdict, {**artifact_gates, **scientific_gates}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    response_revision = resolve_revision(RESPONSES_DATASET)
    question_revision = resolve_revision(QUESTIONS_DATASET)
    questions = load_questions(question_revision, args.cache_dir)
    responses, configs, n_available_configs = load_responses(
        response_revision, args.response_configs, args.cache_dir
    )

    clean, audit = clean_and_join(responses, questions, args.include_human)
    traj, events, counts = analyze_transitions(clean)
    if traj.empty:
        raise RuntimeError("No trajectories with >=2 states after cleaning")

    ci_lo, ci_hi = clustered_bootstrap_rate(clean)
    counts["clustered_bootstrap_95ci_reversal_rate"] = [ci_lo, ci_hi]

    by_config = grouped_summary(traj, ["config", "agent_type"])
    by_type = grouped_summary(traj, ["agent_type"])

    inventory = (
        clean.groupby(["config", "agent_type"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["agent_type", "config"])
    )

    debug_subset = len(configs) != n_available_configs
    verdict, gates = evaluate_gates(
        audit, counts, events, args.include_human, debug_subset
    )

    receipt = {
        "responses": DatasetReceipt(
            RESPONSES_DATASET, response_revision, RESPONSES_SPLIT
        ).__dict__,
        "questions": DatasetReceipt(
            QUESTIONS_DATASET, question_revision, QUESTIONS_SPLIT, QUESTIONS_CONFIG
        ).__dict__,
        "loaded_response_configs": configs,
        "n_loaded_response_configs": len(configs),
        "n_available_response_configs": n_available_configs,
        "config_subset_debug_only": bool(debug_subset),
        "frozen_gates": {
            "min_join_coverage": MIN_JOIN_COVERAGE,
            "min_eligible_correct_transitions": MIN_ELIGIBLE_CORRECT_TRANSITIONS,
            "min_reversal_events": MIN_REVERSAL_EVENTS,
            "min_reversal_rate": MIN_REVERSAL_RATE,
            "min_reversal_questions": MIN_REVERSAL_QUESTIONS,
            "min_reversal_configs": MIN_REVERSAL_CONFIGS,
            "min_strict_alias_events": MIN_STRICT_ALIAS_EVENTS,
        },
        "audit": audit,
    }

    summary = {
        "topic": 28,
        "primary_event": "consecutive cumulative clue states with official score 1->0",
        "human_rows_valid_for_scientific_g0": False,
        "counts": counts,
        "n_reversal_questions": int(events["qid"].nunique()) if not events.empty else 0,
        "n_reversal_configs": int(events["config"].nunique()) if not events.empty else 0,
        "n_strict_alias_reversal_events": (
            int(events["strict_alias_stable"].sum()) if not events.empty else 0
        ),
        "gates": gates,
        "verdict": verdict,
    }

    with open(args.out_dir / "dataset_receipt.json", "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    with open(args.out_dir / "transition_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    inventory.to_csv(args.out_dir / "agent_type_inventory.csv", index=False)
    by_config.to_csv(args.out_dir / "summary_by_config.csv", index=False)
    by_type.to_csv(args.out_dir / "summary_by_agent_type.csv", index=False)
    traj.to_csv(args.out_dir / "trajectory_flags.csv", index=False)
    events.to_csv(args.out_dir / "reversal_events.csv", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote G0 artifacts to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
