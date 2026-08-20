from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_STEP_RE = re.compile(r"(?:^|\n)\s*1\.\s*([a-z])\b", flags=re.IGNORECASE)
EPOCH_RE = re.compile(r"sft_forward_ep(\d+)$")


def parse_args():
    p = argparse.ArgumentParser(description="Measure first-fork output accessibility from reasoning_forks sampled generations.")
    p.add_argument("--forks", default="artifacts/forks.jsonl")
    p.add_argument("--upstream", default="external/reasoning_forks")
    p.add_argument("--base-rel", default="inference_runs/candidate_topic_forward/qwen2.5-0.5b")
    p.add_argument("--output-dir", default="artifacts/behavior")
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


def main():
    args = parse_args()
    forks = load_forks(Path(args.forks))
    root = Path(args.upstream) / args.base_rel
    run_dirs = sorted([p for p in root.glob("sft_forward_ep*") if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No forward sample dirs under {root}")

    sample_rows = []
    for run_dir in run_dirs:
        m = EPOCH_RE.search(run_dir.name)
        if not m:
            continue
        epoch = int(m.group(1))
        files = sorted(run_dir.glob("arithchain_2_10.generations_seed*.csv"))
        if not files:
            raise FileNotFoundError(f"No generation CSV in {run_dir}")
        for file in files:
            df = pd.read_csv(file)
            for _, r in df.iterrows():
                pid = int(r["question_id"])
                f = forks[pid]
                b = first_branch(r["response"])
                is_candidate = b in {f["candidate_a"], f["candidate_b"]}
                sample_rows.append({
                    "epoch": epoch,
                    "problem_id": pid,
                    "first_branch": b,
                    "parsed_candidate": int(is_candidate),
                    "chose_globally_viable": int(is_candidate and b == f["viable"]),
                    "chose_candidate_a": int(is_candidate and b == f["candidate_a"]),
                })

    samples = pd.DataFrame(sample_rows)
    per_problem = []
    for (epoch, pid), g in samples.groupby(["epoch", "problem_id"], sort=True):
        f = forks[int(pid)]
        n_total = len(g)
        n_parsed = int(g["parsed_candidate"].sum())
        n_viable = int(g["chose_globally_viable"].sum())
        n_a = int(g["chose_candidate_a"].sum())
        n_b = n_parsed - n_a
        per_problem.append({
            "epoch": int(epoch),
            "problem_id": int(pid),
            "candidate_a": f["candidate_a"],
            "candidate_b": f["candidate_b"],
            "globally_viable": f["viable"],
            "n_samples": n_total,
            "n_candidate_parsed": n_parsed,
            "parse_rate": n_parsed / n_total,
            "p_viable_first_total": n_viable / n_total,
            "p_viable_first_given_candidate": n_viable / n_parsed if n_parsed else np.nan,
            "first_branch_entropy_given_candidate": binary_entropy(n_a, n_b),
        })
    pp = pd.DataFrame(per_problem)
    summary = pp.groupby("epoch", as_index=False).agg(
        problems=("problem_id", "nunique"),
        mean_parse_rate=("parse_rate", "mean"),
        mean_p_viable_first=("p_viable_first_total", "mean"),
        mean_p_viable_given_candidate=("p_viable_first_given_candidate", "mean"),
        mean_first_branch_entropy=("first_branch_entropy_given_candidate", "mean"),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out / "first_branch_samples.csv", index=False)
    pp.to_csv(out / "first_branch_per_problem.csv", index=False)
    summary.to_csv(out / "first_branch_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
