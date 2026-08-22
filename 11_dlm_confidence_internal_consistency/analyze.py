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
PRIMARY_METRIC = "confidence_result_late"


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


def build_pair_effects(
    rows: list[dict[str, Any]], metric: str
) -> tuple[list[int], dict[str, np.ndarray], dict[int, dict[int, dict[str, float]]]]:
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


def sign_flip_pvalue(x: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    """One-sided randomization p-value for mean(x) > 0 under sign symmetry."""
    if len(x) == 0:
        return float("nan")
    observed = float(x.mean())
    n = len(x)
    exceed = 0
    done = 0
    chunk = 1000
    while done < n_perm:
        k = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(k, n))
        null_means = (signs * x[None, :]).mean(axis=1)
        exceed += int(np.sum(null_means >= observed))
        done += k
    return float((exceed + 1) / (n_perm + 1))


def summarize_effects(effects: dict[str, np.ndarray], seed: int, n_boot: int, n_perm: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, (key, x) in enumerate(sorted(effects.items())):
        if key == "mirror_consistency_product":
            out[key] = {
                "fraction_same_sign": float(np.mean(x > 0)),
                "fraction_opposite_sign": float(np.mean(x < 0)),
            }
            continue
        rng_boot = np.random.default_rng(seed + i * 7919)
        rng_perm = np.random.default_rng(seed + 1000003 + i * 7919)
        lo, hi = bootstrap_ci(x, rng_boot, n_boot)
        out[key] = {
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "ci95": [lo, hi],
            "fraction_positive": float(np.mean(x > 0)),
            "p_signflip_one_sided": sign_flip_pvalue(x, rng_perm, n_perm),
            "n_pairs": int(len(x)),
        }
    return out


def cell_means(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    vals: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        vals[r["cell"]].append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def summarize_protocol(rows: list[dict[str, Any]], seed: int, n_boot: int, n_perm: int) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Protocol probe is empty")
    gaps = np.asarray([float(r["gap"]) for r in rows], dtype=np.float64)
    correct = np.asarray([float(r["confidence_correct"]) for r in rows], dtype=np.float64)
    wrong = np.asarray([float(r["confidence_wrong"]) for r in rows], dtype=np.float64)
    lo, hi = bootstrap_ci(gaps, np.random.default_rng(seed + 424242), n_boot)
    return {
        "n_pairs": int(len(rows)),
        "correct_mean": float(correct.mean()),
        "wrong_mean": float(wrong.mean()),
        "gap_mean": float(gaps.mean()),
        "gap_ci95": [lo, hi],
        "fraction_positive": float(np.mean(gaps > 0)),
        "p_signflip_one_sided": sign_flip_pvalue(gaps, np.random.default_rng(seed + 434343), n_perm),
        "reference_seed_paper": "LLaDA arithmetic probe: ~25.6pp gap at n=500; used here only as a qualitative positive control",
    }


def verdict(protocol: dict[str, Any], metrics: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    p_ci = protocol["gap_ci95"]
    if p_ci[0] <= 0:
        return "INVALID_PROTOCOL_DO_NOT_INTERPRET", [
            "The seed-paper arithmetic positive control did not show a stable positive correct-vs-wrong result-token gap."
        ]

    result = metrics["confidence_result_late"]["effects"]
    tail = metrics["confidence_tail"]["effects"]
    full = metrics["confidence_full"]["effects"]
    r_cons = result["delta_consistency"]["ci95"]
    r_cmp = result["coherent_wrong_minus_incoherent_correct"]["ci95"]
    f_cons = full["delta_consistency"]["ci95"]
    t_cons = tail["delta_consistency"]["ci95"]

    reasons.append("Seed-paper arithmetic positive control passed with a stable positive result-token gap.")
    if r_cons[1] <= 0:
        reasons.append("Unchanged downstream result tokens show no positive internal-consistency effect.")
        return "KILL_NO_INTERNAL_CONSISTENCY_SIGNAL", reasons
    if r_cons[0] <= 0:
        reasons.append("Result-token consistency effect is not stably above zero.")
        return "INCONCLUSIVE_DO_NOT_TUNE", reasons

    reasons.append("Internal consistency raises confidence on text-identical downstream result tokens.")
    if f_cons[0] <= 0:
        reasons.append("The paper-default full-output confidence does not show a stable consistency main effect.")
        return "MIXED_LOCAL_RESULT_SIGNAL_ONLY", reasons

    reasons.append("The paper-default full-output score also shows a stable consistency main effect.")
    if t_cons[0] > 0:
        reasons.append("The effect remains positive when averaged over the entire unchanged continuation, not only result tokens.")
    else:
        reasons.append("The all-continuation effect is diluted/uncertain; this is reported as breadth evidence, not used to rescue the primary result-token test.")

    if r_cmp[0] > 0:
        reasons.append("Coherent-but-wrong stably outranks incoherent-but-correct on unchanged result tokens.")
        return "GO_STRONG_STRUCTURAL_SIGNAL", reasons

    reasons.append("A real internal-consistency signal exists, but it does not stably dominate external correctness in CW-vs-IC.")
    return "MIXED_INTERNAL_SIGNAL_ONLY", reasons


def render_markdown(summary: dict[str, Any]) -> str:
    p = summary["protocol_probe"]
    lines = [
        "# Topic 11 G-0 Result",
        "",
        f"**Verdict:** `{summary['verdict']}`",
        "",
        f"Eligible mirrored anchor pairs: **{summary['n_pairs']}**",
        "",
        "## Scoring-protocol positive control",
        "",
        f"Correct result-token confidence: **{p['correct_mean']:.6f}**",
        f"Wrong result-token confidence: **{p['wrong_mean']:.6f}**",
        f"Paired gap: **{p['gap_mean']:.6f}** (95% CI [{p['gap_ci95'][0]:.6f}, {p['gap_ci95'][1]:.6f}])",
        f"One-sided sign-flip p: **{p['p_signflip_one_sided']:.6g}**",
        "",
        "If this positive control fails, the factorial verdict is invalid rather than negative.",
        "",
        "## Locked factorial results",
        "",
        "| Metric | Effect | Mean | 95% bootstrap CI | Positive pairs | sign-flip p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for metric in ("confidence_result_late", "confidence_result", "confidence_tail", "confidence_full"):
        s = summary[metric]["effects"]
        for effect in (
            "delta_consistency",
            "delta_correctness",
            "coherent_wrong_minus_incoherent_correct",
            "prompt_announcement_match_interaction",
        ):
            e = s[effect]
            lines.append(
                f"| {metric} | {effect} | {e['mean']:.6f} | [{e['ci95'][0]:.6f}, {e['ci95'][1]:.6f}] | {e['fraction_positive']:.3f} | {e['p_signflip_one_sided']:.4g} |"
            )
    lines += ["", "## Cell means", ""]
    for metric in ("confidence_result_first", "confidence_result_late", "confidence_result", "confidence_final", "confidence_tail", "confidence_full", "confidence_announcement"):
        means = summary[metric]["cell_means"]
        lines.append(
            f"- **{metric}:** CC={means.get('CC', float('nan')):.6f}, IC={means.get('IC', float('nan')):.6f}, CW={means.get('CW', float('nan')):.6f}, IW={means.get('IW', float('nan')):.6f}"
        )
    lines += ["", "## Decision rationale", ""]
    lines.extend(f"- {r}" for r in summary["reasons"])
    lines += [
        "",
        "Primary identification uses only late downstream arithmetic-result token positions (step 2 onward), excluding the first result adjacent to the manipulated announcement. Their token strings and positions are identical across the four cells within an orientation. `confidence_result_first` diagnoses purely local mismatch sensitivity; `confidence_tail` tests breadth; `confidence_full` preserves comparability with the seed paper's sequence-level score.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--protocol-probe", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260822)
    args = p.parse_args()

    rows = read_jsonl(args.input)
    probe_rows = read_jsonl(args.protocol_probe)
    if not rows:
        raise SystemExit("No score rows found")
    if not probe_rows:
        raise SystemExit("No protocol-probe rows found")

    summary: dict[str, Any] = {}
    n_pairs_ref: int | None = None
    for metric in ("confidence_result_first", "confidence_result_late", "confidence_result", "confidence_final", "confidence_tail", "confidence_full", "confidence_announcement"):
        pair_ids, effects, _ = build_pair_effects(rows, metric)
        if n_pairs_ref is None:
            n_pairs_ref = len(pair_ids)
        elif len(pair_ids) != n_pairs_ref:
            raise RuntimeError("Metric pair sets disagree")
        summary[metric] = {
            "cell_means": cell_means(rows, metric),
            "effects": summarize_effects(effects, args.seed, args.bootstrap, args.permutations),
        }

    summary["n_pairs"] = int(n_pairs_ref or 0)
    summary["protocol_probe"] = summarize_protocol(probe_rows, args.seed, args.bootstrap, args.permutations)
    summary["design"] = {
        "primary_identification_metric": PRIMARY_METRIC,
        "paper_compatible_metric": "confidence_full",
        "breadth_guardrail": "confidence_tail",
        "unit_of_resampling": "mirrored anchor pair",
        "bootstrap": args.bootstrap,
        "sign_flip_permutations": args.permutations,
        "seed": args.seed,
    }
    summary["verdict"], summary["reasons"] = verdict(summary["protocol_probe"], summary)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"verdict": summary["verdict"], "n_pairs": summary["n_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
