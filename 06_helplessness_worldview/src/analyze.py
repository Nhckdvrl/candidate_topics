from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_test_step1(rows: Iterable[dict]) -> dict[tuple[str, str], dict[int, float]]:
    cells: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r["phase"] == "test" and int(r["trial"]) == 1 and bool(r["valid_action"]):
            cells[(r["condition"], r["diversity"])][int(r["pair_id"])] = float(bool(r["active"]))
    return cells


def paired_helplessness(cells: dict[tuple[str, str], dict[int, float]], diversity: str) -> dict[int, float]:
    c = cells[("controllable", diversity)]
    u = cells[("uncontrollable", diversity)]
    ids = sorted(set(c) & set(u))
    return {i: c[i] - u[i] for i in ids}


def diversity_amplification(cells: dict[tuple[str, str], dict[int, float]]) -> float:
    d = paired_helplessness(cells, "distributed")
    c = paired_helplessness(cells, "concentrated")
    ids = sorted(set(d) & set(c))
    if not ids:
        return float("nan")
    return float(np.mean([d[i] - c[i] for i in ids]))


def bootstrap_amplification(cells: dict[tuple[str, str], dict[int, float]], seed: int = 0, n_boot: int = 5000) -> tuple[float, float, float]:
    d = paired_helplessness(cells, "distributed")
    c = paired_helplessness(cells, "concentrated")
    ids = np.asarray(sorted(set(d) & set(c)), dtype=int)
    if len(ids) == 0:
        return float("nan"), float("nan"), float("nan")
    deltas = np.asarray([d[int(i)] - c[int(i)] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        vals[i] = rng.choice(deltas, size=len(deltas), replace=True).mean()
    return float(deltas.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def invalid_rate(rows: Iterable[dict], phase: str | None = None) -> float:
    xs = [not bool(r["valid_action"]) for r in rows if phase is None or r["phase"] == phase]
    return float(np.mean(xs)) if xs else float("nan")


def training_active_late(rows: Iterable[dict], condition: str, diversity: str, fraction: float = 0.2) -> float:
    by_pair: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["phase"] == "train" and r["condition"] == condition and r["diversity"] == diversity:
            by_pair[int(r["pair_id"])].append(r)
    vals = []
    for rs in by_pair.values():
        rs = sorted(rs, key=lambda x: (int(x["episode"]), int(x["trial"])))
        k = max(1, int(round(len(rs) * fraction)))
        vals.extend(float(bool(x["active"])) for x in rs[-k:] if bool(x["valid_action"]))
    return float(np.mean(vals)) if vals else float("nan")


def cell_training_success(rows: Iterable[dict], condition: str, diversity: str) -> float:
    vals = [float(bool(r["success"])) for r in rows if r["phase"] == "train" and r["condition"] == condition and r["diversity"] == diversity]
    return float(np.mean(vals)) if vals else float("nan")


def late_effective_action_rate(rows: Iterable[dict], condition: str, diversity: str, fraction: float = 0.2) -> float:
    by_pair: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["phase"] == "train" and r["condition"] == condition and r["diversity"] == diversity and bool(r["valid_action"]):
            by_pair[int(r["pair_id"])].append(r)
    vals = []
    for rs in by_pair.values():
        rs = sorted(rs, key=lambda x: (int(x["episode"]), int(x["trial"])))
        k = max(1, int(round(len(rs) * fraction)))
        for x in rs[-k:]:
            vals.append(float(x["latent_action"] == x["effective_action"]))
    return float(np.mean(vals)) if vals else float("nan")


def yoke_mismatch_count(rows: Iterable[dict]) -> int:
    grouped: dict[tuple[str, int, int, int], dict[str, bool]] = defaultdict(dict)
    for r in rows:
        if r["phase"] != "train":
            continue
        key = (r["diversity"], int(r["pair_id"]), int(r["episode"]), int(r["trial"]))
        grouped[key][r["condition"]] = bool(r["success"])
    mismatches = 0
    for x in grouped.values():
        if "controllable" in x and "uncontrollable" in x and x["controllable"] != x["uncontrollable"]:
            mismatches += 1
    return mismatches


def pooled_transfer(cells: dict[tuple[str, str], dict[int, float]]) -> float:
    effects = []
    for div in ("concentrated", "distributed"):
        effects.extend(paired_helplessness(cells, div).values())
    return float(np.mean(effects)) if effects else float("nan")


def summarize(rows: list[dict], n_boot: int = 5000) -> dict:
    cells = extract_test_step1(rows)
    amp, lo, hi = bootstrap_amplification(cells, seed=20260821, n_boot=n_boot)
    out = {
        "invalid_rate_all": invalid_rate(rows),
        "invalid_rate_test": invalid_rate(rows, "test"),
        "step1_active": {},
        "paired_helplessness": {},
        "pooled_transfer_C_minus_U": pooled_transfer(cells),
        "diversity_amplification": amp,
        "diversity_amplification_bootstrap95": [lo, hi],
        "late_training_active": {},
        "late_effective_action_rate": {},
        "training_success_rate": {},
        "yoke_mismatch_count": yoke_mismatch_count(rows),
    }
    for cond in ("controllable", "uncontrollable"):
        for div in ("concentrated", "distributed"):
            vals = list(cells.get((cond, div), {}).values())
            key = f"{cond}:{div}"
            out["step1_active"][key] = float(np.mean(vals)) if vals else float("nan")
            out["late_training_active"][key] = training_active_late(rows, cond, div)
            out["late_effective_action_rate"][key] = late_effective_action_rate(rows, cond, div)
            out["training_success_rate"][key] = cell_training_success(rows, cond, div)
    for div in ("concentrated", "distributed"):
        arr = list(paired_helplessness(cells, div).values())
        out["paired_helplessness"][div] = float(np.mean(arr)) if arr else float("nan")
    c1 = out["training_success_rate"]["controllable:concentrated"]
    c10 = out["training_success_rate"]["controllable:distributed"]
    out["master_success_gap_distributed_minus_concentrated"] = float(c10 - c1)
    return out


def decision(summary: dict) -> dict:
    invalid = summary["invalid_rate_test"]
    if summary.get("yoke_mismatch_count", 0) != 0:
        return {"status": "TECHNICAL_STOP", "reason": "master/yoked training outcomes are not exact matches"}
    master_gap = abs(summary.get("master_success_gap_distributed_minus_concentrated", float("nan")))
    if np.isfinite(master_gap) and master_gap > 0.10:
        return {"status": "EXPOSURE_IMBALANCE", "reason": ">10pp master success-rate difference across diversity conditions; interaction is not cleanly interpretable"}
    pooled = summary["pooled_transfer_C_minus_U"]
    amp = summary["diversity_amplification"]
    lo, hi = summary["diversity_amplification_bootstrap95"]
    if np.isfinite(invalid) and invalid > 0.01:
        return {"status": "TECHNICAL_STOP", "reason": "test invalid-action rate > 1%"}
    if np.isfinite(pooled) and pooled < 0.02:
        return {"status": "SCIENTIFIC_STOP_OR_REDESIGN", "reason": "<2pp pooled transfer of uncontrollability to novel test"}
    if np.isfinite(amp) and amp < 0.02 and lo <= 0 <= hi:
        return {"status": "BOUNDARY_RESULT", "reason": "transfer may exist, but little evidence that diversity amplifies it"}
    if np.isfinite(amp) and amp >= 0.05:
        return {"status": "CONTINUE_TO_CONFIRMATION", "reason": ">=5pp diversity amplification in locked primary endpoint"}
    return {"status": "AMBIGUOUS_INCREASE_POWER_ONLY", "reason": "effect is directionally plausible but preflight/pilot is underpowered"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--out", default=None)
    p.add_argument("--bootstrap", type=int, default=5000)
    args = p.parse_args()
    rows = load_rows(args.input)
    s = summarize(rows, n_boot=args.bootstrap)
    s["decision"] = decision(s)
    text = json.dumps(s, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
