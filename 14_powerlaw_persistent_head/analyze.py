#!/usr/bin/env python3
"""Aggregate locked contrasts and produce a compact, non-tunable decision report.

The primary scalar is normalized AUC of uniformly evaluated token accuracy over a
fixed optimizer-step budget. This rewards learning speed without choosing a lucky
checkpoint. Fixed 5pp/10pp engineering margins are triage rules, not p-values, and
must not be tuned after seeing results.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

CONDITIONS = ["uniform", "static", "balanced_slow", "balanced_fast"]


def read_metrics(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    return rows


def normalized_auc(rows: List[Dict[str, float]], key: str) -> float:
    x = np.asarray([r["step"] for r in rows], dtype=np.float64)
    y = np.asarray([r[key] for r in rows], dtype=np.float64)
    if np.any(np.diff(x) < 0):
        raise ValueError("steps are not monotone")
    if x[-1] <= x[0]:
        return float(y[-1])
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(y, x) / (x[-1] - x[0]))


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "auc_token": normalized_auc(rows, "token_accuracy"),
        "auc_exact": normalized_auc(rows, "exact_accuracy"),
        "final_token": rows[-1]["token_accuracy"],
        "final_exact": rows[-1]["exact_accuracy"],
        "final_loss": rows[-1]["eval_loss"],
        "last_step": rows[-1]["step"],
    }


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def validate_seed_integrity(seed_dir: Path, present: List[str]) -> Dict[str, object]:
    """Fail fast if paired runs differ in anything except the intervention."""
    invariant_keys = [
        "profile", "seed", "mapping_seed", "schedule_seed", "stream_seed",
        "alpha", "batch_size", "d_model", "layers", "heads", "ff_mult",
        "peak_lr", "lr_schedule", "warmup_steps", "weight_decay", "beta1", "beta2", "eps",
        "precision", "cycles_resolved", "block_steps_resolved", "eval_every_resolved",
        "eval_examples_resolved", "bin_eval_examples_resolved", "total_steps",
    ]
    configs = {c: read_json(seed_dir / c / "config.json") for c in present}
    ref = configs[present[0]]
    mismatches = []
    for cond in present[1:]:
        for key in invariant_keys:
            if configs[cond].get(key) != ref.get(key):
                mismatches.append({"condition": cond, "key": key, "ref": ref.get(key), "value": configs[cond].get(key)})
    if mismatches:
        raise SystemExit(f"paired-run config mismatch in {seed_dir}: {mismatches}")

    if "balanced_slow" in present and "balanced_fast" in present:
        sa = read_json(seed_dir / "balanced_slow" / "schedule_audit.json")
        fa = read_json(seed_dir / "balanced_fast" / "schedule_audit.json")
        checks = {
            "slow_rank_occupancy_balanced": bool(sa.get("occupancy_is_exactly_balanced")),
            "fast_rank_occupancy_balanced": bool(fa.get("occupancy_is_exactly_balanced")),
            "slow_realized_counts_equal": bool(sa.get("realized_skill_counts_equal")),
            "fast_realized_counts_equal": bool(fa.get("realized_skill_counts_equal")),
            "same_realized_count": sa.get("realized_skill_count_min") == fa.get("realized_skill_count_min"),
            "persistence_intervention_present": float(sa["lag1_log_weight_corr"]) > float(fa["lag1_log_weight_corr"]) + 0.25,
        }
        if not all(checks.values()):
            raise SystemExit(f"schedule integrity failed in {seed_dir}: {checks}")
        return checks
    return {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("outputs/pilot"))
    p.add_argument("--write-json", type=Path, default=None)
    args = p.parse_args()

    per_seed: dict[str, dict[str, Dict[str, float]]] = defaultdict(dict)
    integrity: dict[str, Dict[str, object]] = {}
    for seed_dir in sorted(args.root.glob("seed*")):
        seed = seed_dir.name
        present = []
        for cond in CONDITIONS:
            mp = seed_dir / cond / "metrics.csv"
            if mp.exists():
                per_seed[seed][cond] = summarize(read_metrics(mp))
                present.append(cond)
        if present:
            integrity[seed] = validate_seed_integrity(seed_dir, present)

    if not per_seed:
        raise SystemExit(f"No metrics found under {args.root}")

    anchor_reports = []
    full_reports = []
    for seed, d in sorted(per_seed.items()):
        if "uniform" in d and "static" in d:
            anchor_reports.append({
                "seed": seed,
                "anchor_auc_gain_static_minus_uniform": d["static"]["auc_token"] - d["uniform"]["auc_token"],
                "anchor_final_gain_static_minus_uniform": d["static"]["final_token"] - d["uniform"]["final_token"],
            })
        if all(c in d for c in CONDITIONS):
            full_reports.append({
                "seed": seed,
                "anchor_auc_gain_static_minus_uniform": d["static"]["auc_token"] - d["uniform"]["auc_token"],
                "anchor_final_gain_static_minus_uniform": d["static"]["final_token"] - d["uniform"]["final_token"],
                "persistence_auc_slow_minus_fast": d["balanced_slow"]["auc_token"] - d["balanced_fast"]["auc_token"],
                "persistence_final_slow_minus_fast": d["balanced_slow"]["final_token"] - d["balanced_fast"]["final_token"],
                "local_asym_auc_slow_minus_uniform": d["balanced_slow"]["auc_token"] - d["uniform"]["auc_token"],
                "local_asym_auc_fast_minus_uniform": d["balanced_fast"]["auc_token"] - d["uniform"]["auc_token"],
                "metrics": d,
            })

    # Frozen engineering margins. They are deliberately simple and must never be
    # swept/tuned post hoc. Primary scientific evidence remains the raw paired curves.
    ANCHOR_MEAN = 0.10
    ANCHOR_EACH = 0.05
    PERSIST_MEAN = 0.05
    LOCAL_MEAN = 0.05

    report: Dict[str, object] = {
        "root": str(args.root),
        "primary_metric": "normalized AUC of uniform-test token accuracy over fixed compute",
        "engineering_margins": {
            "anchor_mean_static_minus_uniform": ANCHOR_MEAN,
            "anchor_each_seed_static_minus_uniform": ANCHOR_EACH,
            "persistence_abs_mean_slow_minus_fast": PERSIST_MEAN,
            "local_asymmetry_each_balanced_minus_uniform_mean": LOCAL_MEAN,
        },
        "n_seeds_with_anchor": len(anchor_reports),
        "n_complete_seeds": len(full_reports),
        "anchor_seed_reports": anchor_reports,
        "integrity_checks": integrity,
    }

    if anchor_reports:
        anchor = np.asarray([r["anchor_auc_gain_static_minus_uniform"] for r in anchor_reports], dtype=np.float64)
        anchor_pass = bool(anchor.mean() >= ANCHOR_MEAN and np.all(anchor > ANCHOR_EACH))
        report["anchor_mean_auc_gain"] = float(anchor.mean())
        report["anchor_pass"] = anchor_pass
    else:
        anchor_pass = False
        report["anchor_pass"] = None

    if not full_reports:
        if not anchor_reports:
            decision = "INCOMPLETE_NO_ANCHOR_PAIR"
        elif anchor_pass:
            decision = "ANCHOR_ONLY_PASS_RUN_BALANCED_CONDITIONS"
        else:
            decision = "ANCHOR_ONLY_FAIL_DO_NOT_INTERPRET_PERSISTENCE"
        report["decision"] = decision
    else:
        def vals(k: str) -> np.ndarray:
            return np.asarray([r[k] for r in full_reports], dtype=np.float64)

        anchor_full = vals("anchor_auc_gain_static_minus_uniform")
        persist = vals("persistence_auc_slow_minus_fast")
        slow_u = vals("local_asym_auc_slow_minus_uniform")
        fast_u = vals("local_asym_auc_fast_minus_uniform")

        full_anchor_pass = bool(anchor_full.mean() >= ANCHOR_MEAN and np.all(anchor_full > ANCHOR_EACH))
        same_nonzero_sign = bool(np.all(persist > 0) or np.all(persist < 0))
        strong_persistence = bool(abs(persist.mean()) >= PERSIST_MEAN and same_nonzero_sign)
        local_asymmetry = bool(slow_u.mean() >= LOCAL_MEAN and fast_u.mean() >= LOCAL_MEAN)

        if not full_anchor_pass:
            decision = "KILL_PREREQUISITE_NOT_REPRODUCED"
        elif strong_persistence and persist.mean() > 0:
            decision = "KEEP_PERSISTENT_HEAD_SIGNAL"
        elif strong_persistence and persist.mean() < 0:
            decision = "KEEP_REVERSE_PERSISTENCE_SIGNAL"
        elif local_asymmetry:
            decision = "KEEP_INSTANTANEOUS_ASYMMETRY_SIGNAL"
        else:
            decision = "KILL_NO_INTERESTING_BALANCED_SCHEDULE_EFFECT"

        report.update({
            "means": {
                "anchor_auc_gain_static_minus_uniform": float(anchor_full.mean()),
                "persistence_auc_slow_minus_fast": float(persist.mean()),
                "local_asym_auc_slow_minus_uniform": float(slow_u.mean()),
                "local_asym_auc_fast_minus_uniform": float(fast_u.mean()),
            },
            "gate_flags": {
                "anchor_pass": full_anchor_pass,
                "strong_persistence": strong_persistence,
                "local_asymmetry": local_asymmetry,
            },
            "decision": decision,
            "seed_reports": full_reports,
        })

    if args.root.name == "smoke":
        report["decision_before_smoke_override"] = report.get("decision")
        report["decision"] = "SMOKE_ONLY_DO_NOT_INTERPRET"

    print(json.dumps(report, indent=2))
    out = args.write_json or (args.root / "decision.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
