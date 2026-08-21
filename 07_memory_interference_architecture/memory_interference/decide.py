from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import load_rows
from .metrics import bootstrap_model_gap, summarize

TRANSFORMER = "transformer_1.3b"
EDITABLE = "gated_deltanet_1.3b"


def decide(rows, n_boot=5000):
    valid = [r for r in rows if not r.get("skipped", False)]
    total = len(rows)
    skip_rate = 1.0 - (len(valid) / total if total else 0.0)
    cells = [x for x in summarize(rows) if x["num_updates"] != "AUC"]
    by_model = {}
    for model in {x["model"] for x in cells}:
        by_model[model] = sorted(
            [x for x in cells if x["model"] == model], key=lambda z: z["num_updates"]
        )

    if TRANSFORMER not in by_model or EDITABLE not in by_model:
        return {"decision": "INVALID", "reason": "primary models missing", "skip_rate": skip_rate}
    if skip_rate > 0.05:
        return {"decision": "INVALID", "reason": "more than 5% rows skipped", "skip_rate": skip_rate}

    t_levels = by_model[TRANSFORMER]
    g_levels = by_model[EDITABLE]
    t_mean = float(np.mean([x["I"] for x in t_levels]))
    g_mean = float(np.mean([x["I"] for x in g_levels]))
    gap = bootstrap_model_gap(rows, TRANSFORMER, EDITABLE, n_boot=n_boot, seed=20260821)
    sign_flip_levels = sum(
        1
        for t, g in zip(t_levels, g_levels)
        if t["num_updates"] == g["num_updates"] and t["I"] > 0 and g["I"] < 0
    )

    base = {
        "transformer_mean_I": t_mean,
        "gated_deltanet_mean_I": g_mean,
        "transformer_minus_gated_deltanet": gap,
        "sign_flip_levels": sign_flip_levels,
        "skip_rate": skip_rate,
    }
    if t_mean <= 0:
        return {"decision": "PARADIGM_FAIL", "reason": "matched Transformer did not reproduce PI>RI", **base}
    if gap["estimate"] >= 0.10 and gap["lo"] > 0:
        if sign_flip_levels >= 3:
            return {"decision": "STRONG_GO", "reason": "robust architecture gap plus repeated sign reversal", **base}
        return {"decision": "GO_TO_LOCKED_CONFIRMATION", "reason": "robust preregistered Transformer-vs-GDN gap", **base}
    if abs(gap["estimate"]) < 0.05:
        return {"decision": "KILL", "reason": "primary architecture gap is practically small in the cheap pilot", **base}
    return {
        "decision": "INCONCLUSIVE_DO_NOT_TUNE",
        "reason": "effect is neither a clean GO nor a clean KILL; do not change metric/prompt/model roster post hoc",
        **base,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--bootstrap", type=int, default=5000)
    args = p.parse_args()
    rows = load_rows(Path(args.run_dir) / "results.jsonl")
    result = decide(rows, n_boot=args.bootstrap)
    out = Path(args.run_dir) / "decision.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
