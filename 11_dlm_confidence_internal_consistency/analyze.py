#!/usr/bin/env python3
"""Analyze Topic-11 locked factorial scores and emit a preregistered verdict."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CELLS = ("CC", "IC", "CW", "IW")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def orientation_effect(c: dict[str, float]) -> dict[str, float]:
    cons_correct = c["CC"] - c["IC"]
    cons_wrong = c["CW"] - c["IW"]
    delta_consistency = 0.5 * (cons_correct + cons_wrong)
    corr_consistent = c["CC"] - c["CW"]
    corr_inconsistent = c["IC"] - c["IW"]
    delta_correctness = 0.5 * (corr_consistent + corr_inconsistent)
    return {
        "consistency_when_correct": cons_correct,
        "consistency_when_wrong": cons_wrong,
        "delta_consistency": delta_consistency,
        "correctness_when_consistent": corr_consistent,
        "correctness_when_inconsistent": corr_inconsistent,
        "delta_correctness": delta_correctness,
        "coherent_wrong_minus_incoherent_correct": c["CW"] - c["IC"],
        "prompt_announcement_match_interaction": cons_correct - cons_wrong,
    }


def build_pair_effects(rows: list[dict[str, Any]], metric: str) -> tuple[list[int], dict[str, np.ndarray], dict[int, dict[int, dict[str, float]]]]:
    nested: dict[int, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        nested[int(r["pair_id"])][int(r["orientation"])][r["cell"]] = float(r[metric])

    pair_ids: list[int] = []
    by_key: dict[str, list[float]] = defaultdict(list)
    clean_nested: dict[int, dict[int, dict[str, float]]] = {}
    for pair_id in sorted(nested):
        if set(nested[pair_id]) != {0, 1}:
            continue
        if any(set(nested[pair_id][o]) != set(CELLS) for o in (0, 1)):
            continue
        e0 = orientation_effect(nested[pair_id][0])
        e1 = orientation_effect(nested[pair_id][1])
        pair_ids.append(pair_id)
        clean_nested[pair_id] = nested[pair_id]
        for key in e0:
            by_key[key].append(0.5 * (e0[key] + e1[key]))
        by_key["mirror_consistency_product"].append(e0["delta_consistency"] * e1["delta_consistency"])

    return pair_ids, {k: np.asarray(v, dtype=np.float64) for k, v in by_key.items()}, clean_nested


def bootstrap_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    n = len(x)
    chunk = 1000
    cursor = 0
    while cursor < n_boot:
        k = min(chunk, n_boot - cursor)
        idx = rng.integers(0, n, size=(k, n))
        means[cursor : cursor + k] = x[idx].mean(axis=1)
        cursor += k
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_effects(effects: dict[str, np.ndarray], seed: int, n_boot: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, (key, x) in enumerate(sorted(effects.items())):
        if key == "mirror_consistency_product":
            out[key] = {
                "fraction_same_sign": float(np.mean(x > 0)),
                "fraction_opposite_sign": float(np.mean(x < 0)),
            }
            continue
        rng = np.random.default_rng(seed + i * 7919)
        lo, hi = bootstrap_ci(x, rng, n_boot)
        out[key] = {
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "ci95": [lo, hi],
            "fraction_positive": float(np.mean(x > 0)),
            "n_pairs": int(len(x)),
        }
    return out


def cell_means(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    vals: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        vals[r["cell"]].append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def verdict(tail: dict[str, Any], full: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    t_cons = tail["delta_consistency"]["ci95"]
    t_cmp = tail["coherent_wrong_minus_incoherent_correct"]["ci95"]
    f_cons = full["delta_consistency"]["ci95"]

    if t_cons[1] <= 0:
        return "KILL_NO_INTERNAL_CONSISTENCY_SIGNAL", [
            "The 95% CI for the consistency effect on unchanged continuation tokens is non-positive."
        ]
    if t_cons[0] <= 0:
        return "INCONCLUSIVE_DO_NOT_TUNE", [
            "Continuation-token consistency effect is not stably above zero."
        ]
    if f_cons[0] <= 0:
        return "INCONCLUSIVE_PROTOCOL_MISMATCH", [
            "Tail-only consistency is positive, but the paper-default full-output mean is not stably positive."
        ]

    reasons.append("Internal consistency raises confidence on unchanged continuation tokens and under the paper-default full-output mean.")
    if t_cmp[0] > 0:
        reasons.append("Coherent-but-wrong stably beats incoherent-but-correct; equivalently, the consistency main effect exceeds the correctness main effect.")
        return "GO_STRONG_STRUCTURAL_SIGNAL", reasons
    reasons.append("A real consistency signal exists, but coherent-wrong does not stably outrank incoherent-correct.")
    return "MIXED_INTERNAL_SIGNAL_ONLY", reasons


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Topic 11 G-0 Result",
        "",
        f"**Verdict:** `{summary['verdict']}`",
        "",
        f"Eligible mirrored anchor pairs: **{summary['n_pairs']}**",
        "",
        "## Locked primary results",
        "",
        "| Metric | Effect | Mean | 95% bootstrap CI | Positive pairs |",
        "|---|---|---:|---:|---:|",
    ]
    for metric in ("confidence_tail", "confidence_full"):
        s = summary[metric]["effects"]
        for effect in ("delta_consistency", "delta_correctness", "coherent_wrong_minus_incoherent_correct", "prompt_announcement_match_interaction"):
            e = s[effect]
            lines.append(
                f"| {metric} | {effect} | {e['mean']:.6f} | [{e['ci95'][0]:.6f}, {e['ci95'][1]:.6f}] | {e['fraction_positive']:.3f} |"
            )
    lines += ["", "## Cell means", ""]
    for metric in ("confidence_tail", "confidence_full", "confidence_announcement"):
        means = summary[metric]["cell_means"]
        lines.append(
            f"- **{metric}:** CC={means.get('CC', float('nan')):.6f}, IC={means.get('IC', float('nan')):.6f}, CW={means.get('CW', float('nan')):.6f}, IW={means.get('IW', float('nan')):.6f}"
        )
    lines += ["", "## Decision rationale", ""]
    lines.extend(f"- {r}" for r in summary["reasons"])
    lines += [
        "",
        "The verdict is intentionally driven by the continuation-only score first. The continuation tokens are text-identical across the consistency manipulation, so a confidence difference there cannot be explained by directly scoring a different downstream token string.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260822)
    args = p.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit("No score rows found")

    summary: dict[str, Any] = {}
    n_pairs_ref: int | None = None
    for metric in ("confidence_tail", "confidence_full", "confidence_announcement"):
        pair_ids, effects, _ = build_pair_effects(rows, metric)
        if n_pairs_ref is None:
            n_pairs_ref = len(pair_ids)
        elif len(pair_ids) != n_pairs_ref:
            raise RuntimeError("Metric pair sets disagree")
        summary[metric] = {
            "cell_means": cell_means(rows, metric),
            "effects": summarize_effects(effects, args.seed, args.bootstrap),
        }

    summary["n_pairs"] = int(n_pairs_ref or 0)
    summary["design"] = {
        "primary_protocol": "paper-default full-output mean same-position probability",
        "identification_guardrail": "mean probability on unchanged continuation tokens",
        "unit_of_resampling": "mirrored anchor pair",
        "bootstrap": args.bootstrap,
        "seed": args.seed,
    }
    summary["verdict"], summary["reasons"] = verdict(
        summary["confidence_tail"]["effects"], summary["confidence_full"]["effects"]
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"verdict": summary["verdict"], "n_pairs": summary["n_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
