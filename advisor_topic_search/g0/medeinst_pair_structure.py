#!/usr/bin/env python3
"""Artifact-only feasibility audit for MedEinst paired counterfactual structure.

This script does NOT test the Einstellung hypothesis. The seed paper already
establishes that phenomenon. It answers the prerequisite question for our
planned token/layer causal analysis:

    Are control -> trap edits sufficiently localized and well-paired that a
    counterfactual activation-patching experiment has a natural intervention
    unit?

Dataset:
    zhui711/MedEinst, split=test

Outputs:
    pair_metrics.csv
    summary.json
    most_diffuse_pairs.csv

No LLM inference, API calls, medical judge, or new annotations are used.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

DATASET = "zhui711/MedEinst"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("medeinst_pair_g0"))
    p.add_argument(
        "--top-k-diffuse",
        type=int,
        default=200,
        help="Save this many highest edit-fraction pairs for manual spot audit.",
    )
    return p.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text))


def diff_metrics(control: str, trap: str) -> dict:
    a = tokenize(control)
    b = tokenize(trap)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    opcodes = sm.get_opcodes()

    changed_a = 0
    changed_b = 0
    changed_blocks = 0
    max_block_a = 0
    max_block_b = 0
    equal_tokens = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            equal_tokens += i2 - i1
            continue
        changed_blocks += 1
        da = i2 - i1
        db = j2 - j1
        changed_a += da
        changed_b += db
        max_block_a = max(max_block_a, da)
        max_block_b = max(max_block_b, db)

    max_len = max(len(a), len(b), 1)
    union_edit = max(changed_a, changed_b)

    return {
        "control_tokens": len(a),
        "trap_tokens": len(b),
        "equal_tokens_lcs_blocks": equal_tokens,
        "changed_control_tokens": changed_a,
        "changed_trap_tokens": changed_b,
        "changed_token_fraction": union_edit / max_len,
        "sequence_match_ratio": sm.ratio(),
        "n_changed_blocks": changed_blocks,
        "max_changed_block_control": max_block_a,
        "max_changed_block_trap": max_block_b,
    }


def quantiles(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET, split="test")
    df = ds.to_pandas()

    required = {"case_id", "case_type", "narrative", "ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset missing required fields: {sorted(missing)}")

    rows = []
    malformed = []

    for case_id, g in df.groupby("case_id", sort=False):
        control = g[g["case_type"] == "control"]
        trap = g[g["case_type"] == "trap"]
        if len(control) != 1 or len(trap) != 1:
            malformed.append(
                {
                    "case_id": case_id,
                    "n_rows": int(len(g)),
                    "n_control": int(len(control)),
                    "n_trap": int(len(trap)),
                }
            )
            continue

        c = control.iloc[0]
        t = trap.iloc[0]
        m = diff_metrics(c["narrative"], t["narrative"])
        m.update(
            {
                "case_id": case_id,
                "control_ground_truth": c["ground_truth"],
                "trap_ground_truth": t["ground_truth"],
                "ground_truth_flips": bool(c["ground_truth"] != t["ground_truth"]),
                "same_age": bool(c.get("age", None) == t.get("age", None)),
                "same_sex": bool(c.get("sex", None) == t.get("sex", None)),
                "control_narrative": c["narrative"],
                "trap_narrative": t["narrative"],
            }
        )
        rows.append(m)

    pairs = pd.DataFrame(rows)
    malformed_df = pd.DataFrame(malformed)

    if pairs.empty:
        raise RuntimeError("No valid control/trap pairs found.")

    metric_cols = [
        "changed_token_fraction",
        "sequence_match_ratio",
        "n_changed_blocks",
        "max_changed_block_control",
        "max_changed_block_trap",
        "control_tokens",
        "trap_tokens",
    ]

    summary = {
        "dataset": DATASET,
        "split": "test",
        "n_rows": int(len(df)),
        "n_unique_case_ids": int(df["case_id"].nunique()),
        "n_valid_pairs": int(len(pairs)),
        "n_malformed_case_ids": int(len(malformed_df)),
        "ground_truth_flip_rate": float(pairs["ground_truth_flips"].mean()),
        "same_age_rate": float(pairs["same_age"].mean()),
        "same_sex_rate": float(pairs["same_sex"].mean()),
        "metrics": {col: quantiles(pairs[col]) for col in metric_cols},
        "interpretation": (
            "Use this only as a structure audit. Freeze a minimum acceptable "
            "pair-locality criterion before deciding project promotion. Do not "
            "choose a convenient threshold after inspecting the distribution."
        ),
    }

    pairs.to_csv(args.out_dir / "pair_metrics.csv", index=False)
    pairs.sort_values("changed_token_fraction", ascending=False).head(
        args.top_k_diffuse
    ).to_csv(args.out_dir / "most_diffuse_pairs.csv", index=False)

    if len(malformed_df):
        malformed_df.to_csv(args.out_dir / "malformed_case_ids.csv", index=False)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote results to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
