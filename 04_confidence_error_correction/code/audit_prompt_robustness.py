#!/usr/bin/env python3
"""Audit G-1v2 measurement reliability across prompts or balanced families."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from mcq_utils import js_divergence


def read_by_id(path: str) -> dict[str, dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["id"])] = r
    return out


def safe_spearman(a, b) -> float:
    v = spearmanr(a, b).statistic
    return float(v) if math.isfinite(float(v)) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--alternate", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--audit-kind",
        choices=["prompt", "permutation_family"],
        default="prompt",
    )
    args = ap.parse_args()

    a = read_by_id(args.primary)
    b = read_by_id(args.alternate)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("no overlapping IDs")

    c1 = np.array([float(a[i]["wrong_concentration"]) for i in ids])
    c2 = np.array([float(b[i]["wrong_concentration"]) for i in ids])
    p1 = np.array([float(a[i]["p_correct"]) for i in ids])
    p2 = np.array([float(b[i]["p_correct"]) for i in ids])
    top_agree = np.mean([int(a[i]["top_wrong"]) == int(b[i]["top_wrong"]) for i in ids])
    js = np.array(
        [
            js_divergence(a[i]["semantic_probs"], b[i]["semantic_probs"])
            for i in ids
        ],
        dtype=float,
    )
    rho_c = safe_spearman(c1, c2)
    rho_p = safe_spearman(p1, p2)

    # Exact top-wrong agreement is diagnostic only in v2: a truly diffuse wrong
    # distribution can legitimately swap top-1 identity under tiny perturbations.
    pass_default = bool(
        rho_c >= 0.70
        and rho_p >= 0.90
        and float(np.median(js)) <= 0.05
    )

    report = {
        "measurement_version": "g1v2_logmean",
        "audit_kind": args.audit_kind,
        "n_overlap": len(ids),
        "wrong_concentration_spearman": rho_c,
        "p_correct_spearman": rho_p,
        "semantic_top_wrong_agreement_diagnostic_only": float(top_agree),
        "median_semantic_js": float(np.median(js)),
        "p90_semantic_js": float(np.quantile(js, 0.90)),
        "pass_default": pass_default,
        "criteria": {
            "wrong_concentration_spearman_min": 0.70,
            "p_correct_spearman_min": 0.90,
            "median_semantic_js_max": 0.05,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
