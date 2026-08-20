#!/usr/bin/env python3
"""Compare primary vs alternate prompt measurements on the same item IDs."""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read_by_id(path: str) -> dict[str, dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["id"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--alternate", required=True)
    ap.add_argument("--output", required=True)
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
    rho_c = spearmanr(c1, c2).statistic
    rho_p = spearmanr(p1, p2).statistic

    report = {
        "n_overlap": len(ids),
        "wrong_concentration_spearman": float(rho_c),
        "p_correct_spearman": float(rho_p),
        "semantic_top_wrong_agreement": float(top_agree),
        "pass_default": bool(rho_c >= 0.70 and top_agree >= 0.75),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
