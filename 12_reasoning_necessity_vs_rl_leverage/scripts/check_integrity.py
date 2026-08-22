#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Audit Topic 12 sweep before statistics")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--expect-layers", type=int, default=28)
    p.add_argument("--allow-missing-layers", action="store_true")
    p.add_argument("--max-baseline-fallback-rate", type=float, default=0.05)
    p.add_argument("--max-baseline-truncation-rate", type=float, default=0.10)
    p.add_argument("--warn-layer-fallback-rate", type=float, default=0.20)
    p.add_argument("--warn-layer-truncation-rate", type=float, default=0.20)
    p.add_argument("--min-informative-baseline-accuracy", type=float, default=0.10)
    p.add_argument("--max-informative-baseline-accuracy", type=float, default=0.95)
    p.add_argument(
        "--paper-table",
        default=str(CODE_ROOT / "data" / "qwen3_1p7b_table13_math.csv"),
    )
    p.add_argument(
        "--max-published-baseline-gap-pp",
        type=float,
        default=15.0,
        help="Fail if our baseline differs from Table-13 base by more than this many percentage points.",
    )
    return p.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keyed(rows):
    return {(r["task"], r["uid"]): r for r in rows}


def condition_health(rows):
    if not rows:
        return {"n": 0, "accuracy": None, "fallback_rate": None, "truncation_rate": None}
    return {
        "n": len(rows),
        "accuracy": float(np.mean([bool(r["correct"]) for r in rows])),
        "fallback_rate": float(np.mean([not bool(r["parse_ok"]) for r in rows])),
        "truncation_rate": float(np.mean([bool(r["truncated"]) for r in rows])),
    }


def published_base_scores(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base = next(r for r in rows if r["setting"] == "Base")
    return {
        "math500": float(base["math500"]) / 100.0,
        "gsm8k": float(base["gsm8k"]) / 100.0,
    }


def main():
    args = parse_args()
    root = Path(args.results_dir)
    problems: list[str] = []
    warnings: list[str] = []
    audit = {
        "status": None,
        "problems": problems,
        "warnings": warnings,
        "baseline_by_task": {},
        "layers": {},
    }
    paper_base = published_base_scores(Path(args.paper_table))

    baseline_path = root / "baseline.jsonl"
    if not baseline_path.exists():
        problems.append("missing baseline.jsonl")
        baseline = []
    else:
        baseline = read_jsonl(baseline_path)

    layer_paths = sorted(root.glob("layer_*.jsonl"))
    indices = []
    for path in layer_paths:
        m = re.search(r"layer_(\d+)\.jsonl$", path.name)
        if m:
            indices.append(int(m.group(1)))

    expected = list(range(args.expect_layers))
    missing = sorted(set(expected) - set(indices))
    extra = sorted(set(indices) - set(expected))
    audit["missing_layers"] = missing
    audit["extra_layers"] = extra
    if missing and not args.allow_missing_layers:
        problems.append(f"missing layer files: {missing}")
    if extra:
        problems.append(f"unexpected layer files: {extra}")

    if baseline:
        base_map = keyed(baseline)
        if len(base_map) != len(baseline):
            problems.append("baseline has duplicate (task, uid) rows")
        base_keys = set(base_map)

        for path in layer_paths:
            rows = read_jsonl(path)
            row_map = keyed(rows)
            if len(row_map) != len(rows):
                problems.append(f"{path.name}: duplicate (task, uid) rows")
            keys = set(row_map)
            if keys != base_keys:
                problems.append(
                    f"{path.name}: ledger mismatch "
                    f"(missing={len(base_keys-keys)}, extra={len(keys-base_keys)})"
                )

        for task in sorted({r["task"] for r in baseline}):
            rows = [r for r in baseline if r["task"] == task]
            health = condition_health(rows)
            audit["baseline_by_task"][task] = health
            print(
                f"baseline {task}: n={health['n']} acc={health['accuracy']:.3f} "
                f"fallback={health['fallback_rate']:.2%} "
                f"trunc={health['truncation_rate']:.2%}"
            )
            if health["fallback_rate"] > args.max_baseline_fallback_rate:
                problems.append(
                    f"{task}: baseline grader fallback {health['fallback_rate']:.2%} "
                    f"> {args.max_baseline_fallback_rate:.2%}"
                )
            if health["truncation_rate"] > args.max_baseline_truncation_rate:
                problems.append(
                    f"{task}: baseline truncation {health['truncation_rate']:.2%} "
                    f"> {args.max_baseline_truncation_rate:.2%}"
                )
            if not (
                args.min_informative_baseline_accuracy
                <= health["accuracy"]
                <= args.max_informative_baseline_accuracy
            ):
                warnings.append(
                    f"{task}: baseline accuracy {health['accuracy']:.3f} is outside "
                    "the predeclared informative range; accuracy-drop necessity may "
                    "have too little paired variance."
                )
            if task in paper_base:
                gap_pp = 100.0 * abs(health["accuracy"] - paper_base[task])
                audit["baseline_by_task"][task]["published_table13_base_accuracy"] = paper_base[task]
                audit["baseline_by_task"][task]["published_gap_pp"] = gap_pp
                if gap_pp > args.max_published_baseline_gap_pp:
                    problems.append(
                        f"{task}: baseline differs from Table-13 base by {gap_pp:.1f} pp "
                        f"> {args.max_published_baseline_gap_pp:.1f} pp; external RL "
                        "curve is not sufficiently protocol-matched."
                    )

    for path in layer_paths:
        rows = read_jsonl(path)
        if not rows:
            problems.append(f"{path.name}: empty")
            continue
        health = condition_health(rows)
        audit["layers"][path.stem] = health
        if health["fallback_rate"] > args.warn_layer_fallback_rate:
            warnings.append(
                f"{path.name}: parser fallback {health['fallback_rate']:.2%}; "
                "inspect whether the necessity effect reflects broad generation damage."
            )
        if health["truncation_rate"] > args.warn_layer_truncation_rate:
            warnings.append(
                f"{path.name}: truncation {health['truncation_rate']:.2%}; "
                "this is retained as an outcome, not silently filtered."
            )

    audit["status"] = "FAIL" if problems else "PASS"
    (root / "integrity_report.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    print("\nIntegrity audit")
    for w in warnings:
        print(f"WARNING: {w}")
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        raise SystemExit(2)

    print(
        "PASS: complete matched ledger and baseline measurement checks passed. "
        "Layer-induced parser/truncation pathologies are reported, not censored."
    )


if __name__ == "__main__":
    main()
