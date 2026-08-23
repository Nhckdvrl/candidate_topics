#!/usr/bin/env python3
"""Frozen feasibility screen for progressive Quiz Bowl correct->wrong reversals.

Dataset:
  mgor/protobowl-11-13-agent-responses

Purpose:
  Decide whether the candidate phenomenon exists at useful density BEFORE any
  hidden-state/mechanistic work. This script intentionally performs no model
  inference and uses only released response records and their released scores.

Primary event:
  Within one (dataset config, agent_type, original question) trajectory,
  correctness changes 1 -> 0 between two *consecutive* cumulative clue states.

Why consecutive only for the primary statistic:
  If clue indices jump (e.g. q10_2 -> q10_5), a reversal is still interesting,
  but multiple clues were added and missing states make the transition less
  clean. We report those separately instead of mixing them into the primary G0.

Human rows are excluded by default because the CAIMIRA processing backfills
human progressive responses under a monotonicity assumption. Use
--include-human only for debugging; do not use those rows for the project-level
reversal claim.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

DATASET = "mgor/protobowl-11-13-agent-responses"
QC_RE = re.compile(r"^(q\d+)_(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("quiz_reversal_g0"))
    p.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional exact HF config names. Default: all configs.",
    )
    p.add_argument(
        "--include-human",
        action="store_true",
        help="Include human_team rows. NOT recommended for the scientific G0.",
    )
    return p.parse_args()


def parse_qc_id(qc_id: str):
    m = QC_RE.match(str(qc_id))
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def load_all(configs: list[str] | None) -> pd.DataFrame:
    if configs is None:
        configs = get_dataset_config_names(DATASET)

    frames = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] loading {config}", flush=True)
        ds = load_dataset(DATASET, config, split="train")
        cols = [c for c in ["agent_type", "qc_id", "answer", "prediction", "score"] if c in ds.column_names]
        df = ds.select_columns(cols).to_pandas()
        df["config"] = config
        frames.append(df)

    if not frames:
        raise RuntimeError("No dataset configs were loaded.")
    return pd.concat(frames, ignore_index=True)


def clean_rows(df: pd.DataFrame, include_human: bool):
    n_raw = len(df)

    parsed = df["qc_id"].map(parse_qc_id)
    df = df.copy()
    df["qid"] = [x[0] for x in parsed]
    df["clue_idx"] = [x[1] for x in parsed]
    df = df[df["qid"].notna() & df["clue_idx"].notna()].copy()

    # Released score is the ground-truth correctness signal used by CAIMIRA.
    df["correct"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["correct"].isin([0, 1])].copy()
    df["correct"] = df["correct"].astype(int)
    df["clue_idx"] = df["clue_idx"].astype(int)

    if not include_human:
        df = df[df["agent_type"].astype(str) != "human_team"].copy()

    # A trajectory cell should contain one released prediction. If duplicate
    # rows disagree on correctness, drop the entire ambiguous cell instead of
    # choosing one post hoc. Identical duplicates are collapsed.
    key = ["config", "agent_type", "qid", "clue_idx"]
    conflict = (
        df.groupby(key, dropna=False)["correct"]
        .nunique()
        .rename("n_correct_values")
        .reset_index()
    )
    conflict = conflict[conflict["n_correct_values"] > 1]
    if len(conflict):
        bad = set(map(tuple, conflict[key].itertuples(index=False, name=None)))
        keep = [tuple(x) not in bad for x in df[key].itertuples(index=False, name=None)]
        df = df.loc[keep].copy()

    df = df.sort_values(key).drop_duplicates(key, keep="first")
    return df, {
        "raw_rows": int(n_raw),
        "clean_rows": int(len(df)),
        "conflicting_duplicate_cells_dropped": int(len(conflict)),
        "human_rows_included": bool(include_human),
    }


def analyze(df: pd.DataFrame):
    group_cols = ["config", "agent_type", "qid"]
    events = []
    trajectories = []

    for keys, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("clue_idx")
        clue = g["clue_idx"].to_numpy()
        corr = g["correct"].to_numpy()
        if len(g) < 2:
            continue

        ever_correct = bool((corr == 1).any())
        first_correct_pos = int(next((i for i, x in enumerate(corr) if x == 1), -1))
        any_primary_reversal = False
        any_gap_reversal = False
        recovered_after_primary = False
        primary_count = 0
        gap_count = 0

        for i in range(1, len(g)):
            prev_c, cur_c = int(corr[i - 1]), int(corr[i])
            delta = int(clue[i] - clue[i - 1])
            if prev_c == 1 and cur_c == 0:
                consecutive = delta == 1
                if consecutive:
                    primary_count += 1
                    any_primary_reversal = True
                else:
                    gap_count += 1
                    any_gap_reversal = True

                row_prev = g.iloc[i - 1]
                row_cur = g.iloc[i]
                events.append(
                    {
                        "config": keys[0],
                        "agent_type": keys[1],
                        "qid": keys[2],
                        "from_clue": int(clue[i - 1]),
                        "to_clue": int(clue[i]),
                        "clue_delta": delta,
                        "primary_consecutive": consecutive,
                        "answer": row_cur.get("answer", None),
                        "prediction_before": row_prev.get("prediction", None),
                        "prediction_after": row_cur.get("prediction", None),
                    }
                )

                if consecutive and (corr[i + 1 :] == 1).any():
                    recovered_after_primary = True

        trajectories.append(
            {
                "config": keys[0],
                "agent_type": keys[1],
                "qid": keys[2],
                "n_states": int(len(g)),
                "ever_correct": ever_correct,
                "primary_reversal": any_primary_reversal,
                "gap_reversal": any_gap_reversal,
                "n_primary_1to0": int(primary_count),
                "n_gap_1to0": int(gap_count),
                "recovered_after_primary_reversal": recovered_after_primary,
                "first_correct_clue": (
                    int(clue[first_correct_pos]) if first_correct_pos >= 0 else None
                ),
            }
        )

    traj = pd.DataFrame(trajectories)
    ev = pd.DataFrame(events)
    return traj, ev


def summarize(traj: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if traj.empty:
        raise RuntimeError("No trajectories with >=2 clue states after cleaning.")

    rows = []
    for (config, agent), g in traj.groupby(["config", "agent_type"], sort=False):
        ever = g[g["ever_correct"]]
        rev = g[g["primary_reversal"]]
        rows.append(
            {
                "config": config,
                "agent_type": agent,
                "n_trajectories": int(len(g)),
                "n_ever_correct": int(len(ever)),
                "n_primary_reversal_questions": int(len(rev)),
                "primary_reversal_rate_given_ever_correct": (
                    float(len(rev) / len(ever)) if len(ever) else None
                ),
                "n_gap_reversal_questions": int(g["gap_reversal"].sum()),
                "n_recovered_after_primary": int(
                    g["recovered_after_primary_reversal"].sum()
                ),
                "recovery_rate_given_primary_reversal": (
                    float(g["recovered_after_primary_reversal"].sum() / len(rev))
                    if len(rev)
                    else None
                ),
            }
        )

    by_config = pd.DataFrame(rows).sort_values(
        ["primary_reversal_rate_given_ever_correct", "n_ever_correct"],
        ascending=[False, False],
        na_position="last",
    )

    ever = traj[traj["ever_correct"]]
    rev = traj[traj["primary_reversal"]]
    global_summary = {
        "n_trajectories": int(len(traj)),
        "n_ever_correct": int(len(ever)),
        "n_primary_reversal_questions": int(len(rev)),
        "primary_reversal_rate_given_ever_correct": (
            float(len(rev) / len(ever)) if len(ever) else None
        ),
        "n_primary_1to0_events": int(traj["n_primary_1to0"].sum()),
        "n_gap_reversal_questions": int(traj["gap_reversal"].sum()),
        "n_recovered_after_primary": int(
            traj["recovered_after_primary_reversal"].sum()
        ),
        "recovery_rate_given_primary_reversal": (
            float(traj["recovered_after_primary_reversal"].sum() / len(rev))
            if len(rev)
            else None
        ),
    }
    return by_config, global_summary


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_all(args.configs)
    clean, audit = clean_rows(raw, args.include_human)
    traj, events = analyze(clean)
    by_config, global_summary = summarize(traj)

    by_config.to_csv(args.out_dir / "summary_by_config.csv", index=False)
    traj.to_csv(args.out_dir / "trajectory_flags.csv", index=False)
    events.to_csv(args.out_dir / "reversal_events.csv", index=False)

    payload = {
        "dataset": DATASET,
        "primary_definition": "consecutive cumulative clue states with correctness 1->0",
        "audit": audit,
        "global": global_summary,
        "note": (
            "This script measures phenomenon density only. It intentionally does "
            "not set a post-hoc GO threshold. Freeze the project-level minimum-worthy "
            "density before using these results to register a numbered topic."
        ),
    }
    with open(args.out_dir / "global_summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"\nWrote results to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
