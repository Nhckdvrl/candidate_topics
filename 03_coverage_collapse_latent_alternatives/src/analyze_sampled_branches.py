from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_STEP_RE = re.compile(r"(?:^|\n)\s*1\.\s*([a-z])\b", flags=re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Measure first-fork commitment and sampled coverage from an isolated run.")
    p.add_argument("--forks", default="artifacts/forks.jsonl")
    p.add_argument("--run-root", required=True)
    p.add_argument("--split", default="arithchain_2_10_g0")
    p.add_argument("--late-tag", default="e16")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-pass-drop", type=float, default=0.03)
    p.add_argument("--min-entropy-drop", type=float, default=0.05)
    p.add_argument("--min-parse-rate", type=float, default=0.90)
    return p.parse_args()


def load_forks(path: Path) -> dict[int, dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {int(r["problem_id"]): r for r in rows}


def first_branch(text: str) -> str | None:
    m = FIRST_STEP_RE.search(str(text))
    return m.group(1).lower() if m else None


def binary_entropy(n_a: int, n_b: int) -> float:
    n = n_a + n_b
    if n == 0:
        return float("nan")
    h = 0.0
    for c in (n_a, n_b):
        if c:
            p = c / n
            h -= p * math.log(p)
    return h


def pass_at_k_from_counts(n: int, c: int, k: int) -> float:
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    miss = 1.0
    for i in range(k):
        miss *= (n - c - i) / (n - i)
    return 1.0 - miss


def paired_bootstrap_delta(ref, late, col, n_boot, seed):
    m = ref[["problem_id", col]].merge(late[["problem_id", col]], on="problem_id", suffixes=("_ref", "_late"))
    vals = (m[f"{col}_ref"] - m[f"{col}_late"]).to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boots = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975))


def main():
    args = parse_args()
    forks = load_forks(Path(args.forks))
    root = Path(args.run_root)
    tag_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("e")])
    if not tag_dirs:
        raise FileNotFoundError(f"No checkpoint tag dirs in {root}")

    sample_rows = []
    for tag_dir in tag_dirs:
        raw_files = sorted(tag_dir.glob(f"{args.split}.generations*.csv"))
        processed = tag_dir / f"{args.split}.processed_generation.csv"
        if len(raw_files) != 1:
            raise RuntimeError(
                f"Expected exactly one raw generation CSV in {tag_dir}; found {len(raw_files)}. "
                "Use a fresh RUN_ID to prevent stale-run contamination."
            )
        if not processed.exists():
            raise FileNotFoundError(f"Missing upstream processed generation file: {processed}")

        df = pd.read_csv(processed)
        for _, r in df.iterrows():
            pid = int(r["question_id"])
            if pid not in forks:
                raise KeyError(f"question_id={pid} missing from fork labels")
            f = forks[pid]
            b = first_branch(r["response"])
            is_candidate = b in {f["candidate_a"], f["candidate_b"]}
            sample_rows.append({
                "tag": tag_dir.name,
                "problem_id": pid,
                "first_branch": b,
                "parsed_candidate": int(is_candidate),
                "chose_globally_viable": int(is_candidate and b == f["viable"]),
                "chose_candidate_a": int(is_candidate and b == f["candidate_a"]),
                "is_correct": int(bool(r["is_correct"])),
            })

    samples = pd.DataFrame(sample_rows)
    per_problem = []
    for (tag, pid), g in samples.groupby(["tag", "problem_id"], sort=True):
        f = forks[int(pid)]
        n_total = len(g)
        n_parsed = int(g["parsed_candidate"].sum())
        n_viable = int(g["chose_globally_viable"].sum())
        n_a = int(g["chose_candidate_a"].sum())
        n_b = n_parsed - n_a
        c = int(g["is_correct"].sum())
        per_problem.append({
            "tag": tag,
            "problem_id": int(pid),
            "candidate_a": f["candidate_a"],
            "candidate_b": f["candidate_b"],
            "globally_viable": f["viable"],
            "n_samples": n_total,
            "n_correct": c,
            "n_candidate_parsed": n_parsed,
            "parse_rate": n_parsed / n_total,
            "p_viable_first_total": n_viable / n_total,
            "p_viable_first_given_candidate": n_viable / n_parsed if n_parsed else np.nan,
            "first_branch_entropy_given_candidate": binary_entropy(n_a, n_b),
            "pass_at_1": c / n_total,
            "pass_at_half": pass_at_k_from_counts(n_total, c, max(1, n_total // 2)),
            "pass_at_n": float(c > 0),
        })

    pp = pd.DataFrame(per_problem)
    summary = pp.groupby("tag", as_index=False).agg(
        problems=("problem_id", "nunique"),
        mean_parse_rate=("parse_rate", "mean"),
        mean_p_viable_first=("p_viable_first_total", "mean"),
        mean_p_viable_given_candidate=("p_viable_first_given_candidate", "mean"),
        mean_first_branch_entropy=("first_branch_entropy_given_candidate", "mean"),
        pass_at_1=("pass_at_1", "mean"),
        pass_at_half=("pass_at_half", "mean"),
        pass_at_n=("pass_at_n", "mean"),
    )

    if args.late_tag not in set(summary.tag):
        raise ValueError(f"late tag {args.late_tag!r} not found")
    early = summary[summary.tag != args.late_tag].sort_values(["pass_at_half", "pass_at_n"], ascending=False)
    if early.empty:
        raise ValueError("Need at least one non-late checkpoint")
    reference_tag = str(early.iloc[0].tag)

    ref_pp = pp[pp.tag == reference_tag]
    late_pp = pp[pp.tag == args.late_tag]
    pass_drop, pass_lo, pass_hi = paired_bootstrap_delta(
        ref_pp, late_pp, "pass_at_half", args.bootstrap, args.seed
    )
    ent_drop, ent_lo, ent_hi = paired_bootstrap_delta(
        ref_pp, late_pp, "first_branch_entropy_given_candidate", args.bootstrap, args.seed + 1
    )

    ref_parse = float(summary.loc[summary.tag == reference_tag, "mean_parse_rate"].iloc[0])
    late_parse = float(summary.loc[summary.tag == args.late_tag, "mean_parse_rate"].iloc[0])

    reasons = []
    if not np.isfinite(pass_lo) or pass_lo <= 0.0:
        reasons.append("coverage drop is not positive under paired bootstrap")
    if pass_drop < args.min_pass_drop:
        reasons.append(f"coverage drop {pass_drop:.4f} is below minimum practical effect {args.min_pass_drop:.4f}")
    if not np.isfinite(ent_lo) or ent_lo <= 0.0:
        reasons.append("first-fork entropy does not reliably decrease")
    if ent_drop < args.min_entropy_drop:
        reasons.append(f"entropy drop {ent_drop:.4f} is below minimum practical effect {args.min_entropy_drop:.4f}")
    if min(ref_parse, late_parse) < args.min_parse_rate:
        reasons.append(
            f"first-branch parser coverage too low: reference={ref_parse:.3f}, late={late_parse:.3f}, "
            f"required>={args.min_parse_rate:.2f}"
        )

    gate = {
        "status": "continue_to_latent" if not reasons else "stop_or_redesign",
        "reference_tag": reference_tag,
        "late_tag": args.late_tag,
        "pass_at_half_drop_ref_minus_late": pass_drop,
        "pass_at_half_drop_ci95": [pass_lo, pass_hi],
        "branch_entropy_drop_ref_minus_late": ent_drop,
        "branch_entropy_drop_ci95": [ent_lo, ent_hi],
        "reference_parse_rate": ref_parse,
        "late_parse_rate": late_parse,
        "thresholds": {
            "min_pass_drop": args.min_pass_drop,
            "min_entropy_drop": args.min_entropy_drop,
            "min_parse_rate": args.min_parse_rate,
        },
        "reasons": reasons,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out / "first_branch_samples.csv", index=False)
    pp.to_csv(out / "first_branch_per_problem.csv", index=False)
    summary.to_csv(out / "first_branch_summary.csv", index=False)
    (out / "gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
