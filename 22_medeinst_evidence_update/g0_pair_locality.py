#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def toks(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")


def pair_metrics(control: dict, trap: dict) -> dict:
    a, b = toks(control["narrative"]), toks(trap["narrative"])
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    changed = blocks = largest = 0
    changes = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        blocks += 1
        span = max(i2 - i1, j2 - j1)
        changed += span
        largest = max(largest, span)
        changes.append(
            {
                "tag": tag,
                "control_token_span": [i1, i2],
                "trap_token_span": [j1, j2],
                "control_text": " ".join(a[i1:i2]),
                "trap_text": " ".join(b[j1:j2]),
            }
        )
    denom = max(1, max(len(a), len(b)))
    return {
        "case_id": control["case_id"],
        "control_gt": control["ground_truth"],
        "trap_gt": trap["ground_truth"],
        "gt_flips": control["ground_truth"] != trap["ground_truth"],
        "same_age": control["age"] == trap["age"],
        "same_sex": control["sex"] == trap["sex"],
        "control_tokens": len(a),
        "trap_tokens": len(b),
        "sequence_match_ratio": sm.ratio(),
        "changed_token_fraction": changed / denom,
        "change_blocks": blocks,
        "largest_change_span_fraction": largest / denom,
        "changes": changes,
    }


def q(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, round((len(ys) - 1) * p)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="zhui711/MedEinst")
    ap.add_argument("--split", default="test")
    ap.add_argument("--outdir", default="artifacts/g0_pair_locality")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.dataset, split=args.split)
    grouped = defaultdict(list)
    for row in ds:
        grouped[row["case_id"]].append(dict(row))

    metrics, malformed = [], []
    for case_id, rows in tqdm(grouped.items(), desc="MedEinst pair audit"):
        by = defaultdict(list)
        for row in rows:
            by[row["case_type"]].append(row)
        if len(by["control"]) != 1 or len(by["trap"]) != 1:
            malformed.append(
                {"case_id": case_id, "n_control": len(by["control"]), "n_trap": len(by["trap"])}
            )
            continue
        metrics.append(pair_metrics(by["control"][0], by["trap"][0]))

    csv_fields = [
        "case_id",
        "control_gt",
        "trap_gt",
        "gt_flips",
        "same_age",
        "same_sex",
        "control_tokens",
        "trap_tokens",
        "sequence_match_ratio",
        "changed_token_fraction",
        "change_blocks",
        "largest_change_span_fraction",
    ]
    if metrics:
        with (out / "pair_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for row in metrics:
                writer.writerow({k: row[k] for k in csv_fields})

    with (out / "pair_diffs.jsonl").open("w", encoding="utf-8") as f:
        for row in metrics:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out / "most_diffuse_200.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(metrics, key=lambda x: x["changed_token_fraction"], reverse=True)[:200]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out / "malformed_pairs.jsonl").open("w", encoding="utf-8") as f:
        for row in malformed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(metrics)
    changed = [m["changed_token_fraction"] for m in metrics]
    largest = [m["largest_change_span_fraction"] for m in metrics]
    flip = sum(m["gt_flips"] for m in metrics) / max(1, n)
    demographics = sum(m["same_age"] and m["same_sex"] for m in metrics) / max(1, n)
    gate = {
        "pair_count_ge_5000": n >= 5000,
        "malformed_pairs_eq_0": not malformed,
        "ground_truth_flip_rate_ge_0.99": flip >= 0.99,
        "age_sex_match_rate_ge_0.99": demographics >= 0.99,
        "median_changed_fraction_le_0.12": q(changed, 0.5) <= 0.12,
        "p90_changed_fraction_le_0.30": q(changed, 0.9) <= 0.30,
    }
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "raw_rows": len(ds),
        "valid_pairs": n,
        "malformed_pairs": len(malformed),
        "ground_truth_flip_rate": flip,
        "age_sex_match_rate": demographics,
        "changed_token_fraction": {
            "median": q(changed, 0.5),
            "p90": q(changed, 0.9),
            "p95": q(changed, 0.95),
        },
        "largest_change_span_fraction": {
            "median": q(largest, 0.5),
            "p90": q(largest, 0.9),
        },
        "interpretation": (
            "This gate tests whether the released counterfactual pairs are local enough for aligned intervention. "
            "It does not independently prove that every changed span is the decisive medical evidence; that validity comes from the seed benchmark construction."
        ),
        "gate": gate,
        "verdict": "PAIR_STRUCTURE_OK" if all(gate.values()) else "PAIR_STRUCTURE_NOT_CLEAN_ENOUGH",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
