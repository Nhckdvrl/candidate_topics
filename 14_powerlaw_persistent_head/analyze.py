#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from core import LOCKED_FULL_SEEDS, LOCKED_PAPER_ANCHOR_SEEDS, PROFILES, PROTOCOL_VERSION

PRIMARY_ARMS = ("uniform", "static", "slow", "fast")
ANCHOR_ARMS = ("uniform", "static")

CORE_ANCHOR_MEDIAN = 0.03
CORE_ANCHOR_POS_FRAC = 0.80
PERSIST_ABS_MEDIAN = 0.10
PERSIST_SIGN_FRAC = 0.80
NEAR_ZERO_MEDIAN = 0.03
NEAR_ZERO_PER_SEED = 0.06
NEAR_ZERO_FRAC = 0.80
PAPER_ANCHOR_MEDIAN_AUC = 0.10
PAPER_ANCHOR_POS_FRAC = 2 / 3
PAPER_ANCHOR_STATIC_FINAL = 0.50


def parse_seed_list(text: str | None, profile: str) -> list[int]:
    if text:
        return [int(x) for x in text.split(",") if x.strip()]
    if profile == "full":
        return list(LOCKED_FULL_SEEDS)
    if profile == "paper_anchor":
        return list(LOCKED_PAPER_ANCHOR_SEEDS)
    return [0]


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"empty metrics: {path}")
    return rows


def auc(rows: list[dict[str, float]], key: str) -> float:
    x = np.asarray([r["step"] for r in rows], dtype=np.float64)
    y = np.asarray([r[key] for r in rows], dtype=np.float64)
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("metric steps must be strictly increasing and contain >=2 points")
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(y, x) / (x[-1] - x[0]))


def one(run: Path) -> dict[str, Any]:
    rows = read_csv(run / "metrics.csv")
    cfg = json.loads((run / "config.json").read_text())
    done = json.loads((run / "done.json").read_text())
    return {
        "rows": rows,
        "auc_exact": auc(rows, "exact_accuracy"),
        "auc_token": auc(rows, "token_accuracy"),
        "final_exact": rows[-1]["exact_accuracy"],
        "final_token": rows[-1]["token_accuracy"],
        "branch_digest": cfg["branch_digest"],
        "cfg": cfg,
        "done": done,
    }


def same_values(d: dict[str, dict[str, Any]], arms: tuple[str, ...], keys: list[str]) -> list[str]:
    bad = []
    ref = d[arms[0]]["cfg"]
    for arm in arms[1:]:
        cfg = d[arm]["cfg"]
        for key in keys:
            if cfg.get(key) != ref.get(key):
                bad.append(f"{arm}:{key}")
    return bad


def validate_seed(seed: int, d: dict[str, dict[str, Any]], profile: str, arms: tuple[str, ...]) -> list[str]:
    bad: list[str] = []
    if any(a not in d for a in arms):
        return [f"seed{seed}:missing_arms"]

    invariant_keys = [
        "protocol_version",
        "profile",
        "profile_resolved",
        "seed",
        "batch_size",
        "eval_batch_size",
        "alpha",
        "mapping_seed_base",
        "mapping_seed_effective",
        "stream_seed",
        "peak_lr",
        "lr_schedule",
        "paper_lr_warmup_steps",
        "weight_decay",
        "betas",
        "eps",
        "precision",
        "d_model",
        "layers",
        "heads",
        "ff_mult",
        "eval_seed",
        "branch_digest",
    ]
    bad.extend(f"seed{seed}:{x}" for x in same_values(d, arms, invariant_keys))

    ref_steps = [int(r["step"]) for r in d[arms[0]]["rows"]]
    expected_last = int(PROFILES[profile].core_steps)
    if ref_steps[0] != 0 or ref_steps[-1] != expected_last:
        bad.append(f"seed{seed}:metric_grid_endpoints")
    for arm in arms:
        x = d[arm]
        cfg = x["cfg"]
        done = x["done"]
        if cfg.get("protocol_version") != PROTOCOL_VERSION:
            bad.append(f"seed{seed}:{arm}:stale_protocol")
        if cfg.get("profile") != profile or int(cfg.get("seed", -1)) != seed:
            bad.append(f"seed{seed}:{arm}:profile_or_seed")
        if done.get("run_signature") != cfg.get("run_signature"):
            bad.append(f"seed{seed}:{arm}:run_signature")
        if done.get("branch_digest") != cfg.get("branch_digest"):
            bad.append(f"seed{seed}:{arm}:done_branch_digest")
        steps = [int(r["step"]) for r in x["rows"]]
        if steps != ref_steps:
            bad.append(f"seed{seed}:{arm}:metric_grid")

    if profile != "paper_anchor":
        ps = d["slow"]["cfg"].get("schedule")
        pf = d["fast"]["cfg"].get("schedule")
        if not ps or not pf:
            bad.append(f"seed{seed}:missing_schedule")
        else:
            if ps.get("multiset_digest") != pf.get("multiset_digest"):
                bad.append(f"seed{seed}:schedule_multiset")
            if ps.get("temporal_digest") == pf.get("temporal_digest"):
                bad.append(f"seed{seed}:schedule_order")
    return bad


def decide_full(anchor: np.ndarray, diffs: np.ndarray) -> tuple[str, dict[str, float]]:
    summary = {
        "median_anchor": float(np.median(anchor)),
        "anchor_positive_fraction": float(np.mean(anchor > 0)),
        "median_slow_minus_fast": float(np.median(diffs)),
        "slow_positive_fraction": float(np.mean(diffs > 0)),
        "slow_negative_fraction": float(np.mean(diffs < 0)),
        "near_zero_fraction": float(np.mean(np.abs(diffs) <= NEAR_ZERO_PER_SEED)),
    }
    anchor_ok = summary["median_anchor"] >= CORE_ANCHOR_MEDIAN and summary["anchor_positive_fraction"] >= CORE_ANCHOR_POS_FRAC
    pos = summary["median_slow_minus_fast"] >= PERSIST_ABS_MEDIAN and summary["slow_positive_fraction"] >= PERSIST_SIGN_FRAC
    neg = summary["median_slow_minus_fast"] <= -PERSIST_ABS_MEDIAN and summary["slow_negative_fraction"] >= PERSIST_SIGN_FRAC
    eq = abs(summary["median_slow_minus_fast"]) <= NEAR_ZERO_MEDIAN and summary["near_zero_fraction"] >= NEAR_ZERO_FRAC
    if not anchor_ok:
        return "CORE_ANCHOR_WEAK_NO_PERSISTENCE_CONCLUSION", summary
    if pos:
        return "PASS_PERSISTENT_HEAD_HELPS", summary
    if neg:
        return "PASS_RAPID_SWITCHING_HELPS", summary
    if eq:
        return "KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT", summary
    return "INCONCLUSIVE_FIXED_PROTOCOL_NO_TUNING", summary


def decide_paper_anchor(anchor: np.ndarray, static_final: np.ndarray) -> tuple[str, dict[str, float]]:
    summary = {
        "median_anchor_auc": float(np.median(anchor)),
        "anchor_positive_fraction": float(np.mean(anchor > 0)),
        "median_static_final_exact": float(np.median(static_final)),
    }
    ok = (
        summary["median_anchor_auc"] >= PAPER_ANCHOR_MEDIAN_AUC
        and summary["anchor_positive_fraction"] >= PAPER_ANCHOR_POS_FRAC
        and summary["median_static_final_exact"] >= PAPER_ANCHOR_STATIC_FINAL
    )
    return ("PAPER_ANCHOR_REPRODUCED" if ok else "TECHNICAL_SEED_REPRODUCTION_FAILED_DEBUG_BEFORE_SCIENCE"), summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--seeds", default=None)
    args = p.parse_args()
    requested = parse_seed_list(args.seeds, args.profile)
    arms = ANCHOR_ARMS if args.profile == "paper_anchor" else PRIMARY_ARMS

    rep: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "profile": args.profile,
        "requested_seeds": requested,
        "seeds": {},
        "decision": None,
    }
    bad: list[str] = []
    mapping_seeds: list[int] = []
    anchor, diffs, static_final = [], [], []

    for seed in requested:
        sd = args.root / f"seed{seed}"
        d: dict[str, dict[str, Any]] = {}
        for arm in arms:
            if all((sd / arm / name).exists() for name in ("metrics.csv", "config.json", "done.json")):
                d[arm] = one(sd / arm)
        if any(a not in d for a in arms):
            bad.append(f"seed{seed}:incomplete")
            continue
        bad.extend(validate_seed(seed, d, args.profile, arms))
        mapping_seeds.append(int(d[arms[0]]["cfg"]["mapping_seed_effective"]))
        ag = d["static"]["auc_exact"] - d["uniform"]["auc_exact"]
        anchor.append(ag)
        seed_report = {
            "anchor_static_minus_uniform_auc_exact": ag,
            "metrics": {
                arm: {k: d[arm][k] for k in ("auc_exact", "auc_token", "final_exact", "final_token")}
                for arm in arms
            },
            "mapping_seed_effective": d[arms[0]]["cfg"]["mapping_seed_effective"],
        }
        if args.profile == "paper_anchor":
            static_final.append(d["static"]["final_exact"])
        else:
            de = d["slow"]["auc_exact"] - d["fast"]["auc_exact"]
            diffs.append(de)
            seed_report["slow_minus_fast_auc_exact"] = de
        rep["seeds"][f"seed{seed}"] = seed_report

    if len(mapping_seeds) > 1 and len(set(mapping_seeds)) != len(mapping_seeds):
        bad.append("mapping_seeds_not_unique_across_replications")

    if bad:
        rep["decision"] = "TECHNICAL_INVALID_INTEGRITY_OR_INCOMPLETE"
        rep["integrity_failures"] = bad
    elif args.profile == "smoke":
        rep["decision"] = "SMOKE_ONLY_DO_NOT_INTERPRET"
    elif args.profile == "pilot":
        rep["decision"] = "PILOT_SIGNAL_ONLY_DO_NOT_CONCLUDE"
    elif args.profile == "paper_anchor":
        if requested != list(LOCKED_PAPER_ANCHOR_SEEDS):
            rep["decision"] = "INCOMPLETE_NEED_LOCKED_PAPER_ANCHOR_SEEDS"
        else:
            rep["decision"], rep["summary"] = decide_paper_anchor(np.asarray(anchor), np.asarray(static_final))
    elif args.profile == "full":
        if requested != list(LOCKED_FULL_SEEDS):
            rep["decision"] = "INCOMPLETE_NEED_LOCKED_FULL_SEEDS"
        else:
            rep["decision"], rep["summary"] = decide_full(np.asarray(anchor), np.asarray(diffs))
    else:
        rep["decision"] = "DIAGNOSTIC_ONLY"

    print(json.dumps(rep, indent=2))
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "decision.json").write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
