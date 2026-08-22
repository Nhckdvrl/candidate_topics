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
    p.add_argument("--baseline-only", action="store_true", help="Audit baseline/protocol before spending compute on layer sweep.")
    p.add_argument("--max-baseline-fallback-rate", type=float, default=0.05)
    p.add_argument("--max-baseline-truncation-rate", type=float, default=0.10)
    p.add_argument("--max-input-truncation-rate", type=float, default=0.0)
    p.add_argument("--warn-layer-fallback-rate", type=float, default=0.20)
    p.add_argument("--warn-layer-truncation-rate", type=float, default=0.20)
    p.add_argument("--min-informative-baseline-accuracy", type=float, default=0.10)
    p.add_argument("--max-informative-baseline-accuracy", type=float, default=0.95)
    p.add_argument("--paper-table", default=str(CODE_ROOT / "data" / "qwen3_1p7b_table13_math.csv"))
    p.add_argument("--published-gap-z", type=float, default=2.0,
                   help="Allowed baseline gap is max(floor, z * binomial SE around paper score).")
    p.add_argument("--published-gap-floor-pp", type=float, default=5.0)
    return p.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keyed(rows):
    return {(r["task"], r["uid"]): r for r in rows}


def condition_health(rows):
    if not rows:
        return {"n": 0, "accuracy": None, "fallback_rate": None, "truncation_rate": None, "input_truncation_rate": None}
    return {
        "n": len(rows),
        "accuracy": float(np.mean([bool(r["correct"]) for r in rows])),
        "fallback_rate": float(np.mean([not bool(r["parse_ok"]) for r in rows])),
        "truncation_rate": float(np.mean([bool(r["truncated"]) for r in rows])),
        "input_truncation_rate": float(np.mean([bool(r.get("input_truncated", False)) for r in rows])),
    }


def published_base_scores(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base = next(r for r in rows if r["setting"] == "Base")
    return {"math500": float(base["math500"]) / 100.0, "gsm8k": float(base["gsm8k"]) / 100.0}


def main():
    args = parse_args()
    root = Path(args.results_dir)
    problems: list[str] = []
    warnings: list[str] = []
    audit = {"status": None, "problems": problems, "warnings": warnings, "baseline_by_task": {}, "layers": {}}

    contract_path = root / "run_contract.json"
    if not contract_path.exists():
        problems.append("missing run_contract.json; cannot prove conditions share one protocol")
        contract = {}
    else:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_id = contract.get("contract_id")
    audit["contract_id"] = contract_id
    audit["contract"] = contract
    if not contract_id:
        problems.append("run contract has no contract_id")
    if contract and not contract.get("model_source_is_local", False):
        requested = contract.get("requested_model_revision")
        resolved = contract.get("resolved_model_commit")
        if requested and not resolved:
            problems.append("remote model revision was requested but resolved commit is missing")
        elif requested and resolved and requested != resolved:
            problems.append(f"resolved model commit {resolved} != requested pinned revision {requested}")
    elif contract.get("model_source_is_local", False) and not contract.get("resolved_model_commit"):
        warnings.append(
            "local model snapshot has no HF commit metadata; ensure it was staged from the pinned "
            "Qwen revision before interpreting results."
        )

    paper_base = published_base_scores(Path(args.paper_table))
    baseline_path = root / "baseline.jsonl"
    baseline = read_jsonl(baseline_path) if baseline_path.exists() else []
    if not baseline:
        problems.append("missing or empty baseline.jsonl")

    layer_paths = sorted(root.glob("layer_*.jsonl"))
    indices = []
    for path in layer_paths:
        m = re.search(r"layer_(\d+)\.jsonl$", path.name)
        if m:
            indices.append(int(m.group(1)))
    if args.baseline_only:
        missing, extra = [], []
    else:
        expected = list(range(args.expect_layers))
        missing = sorted(set(expected) - set(indices))
        extra = sorted(set(indices) - set(expected))
        if missing and not args.allow_missing_layers:
            problems.append(f"missing layer files: {missing}")
        if extra:
            problems.append(f"unexpected layer files: {extra}")
    audit["missing_layers"] = missing
    audit["extra_layers"] = extra

    if baseline:
        base_map = keyed(baseline)
        if len(base_map) != len(baseline):
            problems.append("baseline has duplicate (task, uid) rows")
        base_keys = set(base_map)
        if contract_id and any(r.get("contract_id") != contract_id for r in baseline):
            problems.append("baseline rows do not all match run contract")

        for path in ([] if args.baseline_only else layer_paths):
            rows = read_jsonl(path)
            row_map = keyed(rows)
            if len(row_map) != len(rows):
                problems.append(f"{path.name}: duplicate (task, uid) rows")
            keys = set(row_map)
            if keys != base_keys:
                problems.append(f"{path.name}: ledger mismatch (missing={len(base_keys-keys)}, extra={len(keys-base_keys)})")
            if contract_id and any(r.get("contract_id") != contract_id for r in rows):
                problems.append(f"{path.name}: rows do not all match run contract")

        for task in sorted({r["task"] for r in baseline}):
            rows = [r for r in baseline if r["task"] == task]
            health = condition_health(rows)
            audit["baseline_by_task"][task] = health
            print(
                f"baseline {task}: n={health['n']} acc={health['accuracy']:.3f} "
                f"fallback={health['fallback_rate']:.2%} out_trunc={health['truncation_rate']:.2%} "
                f"in_trunc={health['input_truncation_rate']:.2%}"
            )
            if health["fallback_rate"] > args.max_baseline_fallback_rate:
                problems.append(f"{task}: baseline grader fallback {health['fallback_rate']:.2%} > {args.max_baseline_fallback_rate:.2%}")
            if health["truncation_rate"] > args.max_baseline_truncation_rate:
                problems.append(f"{task}: baseline output truncation {health['truncation_rate']:.2%} > {args.max_baseline_truncation_rate:.2%}")
            if health["input_truncation_rate"] > args.max_input_truncation_rate:
                problems.append(f"{task}: input truncation {health['input_truncation_rate']:.2%}; task statements must not be silently clipped")
            if not (args.min_informative_baseline_accuracy <= health["accuracy"] <= args.max_informative_baseline_accuracy):
                warnings.append(f"{task}: baseline accuracy {health['accuracy']:.3f} outside informative range")

            if task in paper_base:
                p0 = paper_base[task]
                n = health["n"]
                se = float(np.sqrt(max(p0 * (1.0 - p0), 1e-12) / max(n, 1)))
                allowed_pp = max(args.published_gap_floor_pp, 100.0 * args.published_gap_z * se)
                gap_pp = 100.0 * abs(health["accuracy"] - p0)
                health["published_table13_base_accuracy"] = p0
                health["published_gap_pp"] = gap_pp
                health["published_allowed_gap_pp"] = allowed_pp
                if gap_pp > allowed_pp:
                    problems.append(
                        f"{task}: baseline differs from Table-13 base by {gap_pp:.1f} pp > "
                        f"allowed {allowed_pp:.1f} pp. Protocol is not sufficiently matched; "
                        "do not compare our I curve to the external C curve."
                    )

    for path in ([] if args.baseline_only else layer_paths):
        rows = read_jsonl(path)
        if not rows:
            problems.append(f"{path.name}: empty")
            continue
        health = condition_health(rows)
        audit["layers"][path.stem] = health
        if health["input_truncation_rate"] > args.max_input_truncation_rate:
            problems.append(f"{path.name}: input truncation detected")
        if health["fallback_rate"] > args.warn_layer_fallback_rate:
            warnings.append(f"{path.name}: parser fallback {health['fallback_rate']:.2%}; possible broad generation damage")
        if health["truncation_rate"] > args.warn_layer_truncation_rate:
            warnings.append(f"{path.name}: output truncation {health['truncation_rate']:.2%}; retained as causal outcome")

    audit["status"] = "FAIL" if problems else "PASS"
    (root / "integrity_report.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("\nIntegrity audit")
    for w in warnings:
        print(f"WARNING: {w}")
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        raise SystemExit(2)
    print("PASS: one frozen contract, exact ledger match, usable baseline, no silent input truncation, and external-score compatibility.")


if __name__ == "__main__":
    main()
