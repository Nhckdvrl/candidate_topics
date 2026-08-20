from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from graph_parser import parse_first_fork


def main():
    p = argparse.ArgumentParser(description="Recover exact first-fork viability labels from reasoning_forks ArithChain data.")
    p.add_argument("--input", required=True, help="Path to official datasets/arithchain_2_10/test.parquet")
    p.add_argument("--output", default="artifacts/forks.jsonl")
    p.add_argument("--limit", type=int, default=1000)
    args = p.parse_args()

    df = pd.read_parquet(args.input).head(args.limit).reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # question_id in upstream test.json is positional. Do not inherit a parquet index.
    for problem_id, row in enumerate(df.to_dict(orient="records")):
        f = parse_first_fork(row["question"])
        rows.append({
            "problem_id": problem_id,
            "question": row["question"],
            "target": f.target,
            "premise": f.premise,
            "candidate_a": f.candidate_a,
            "candidate_b": f.candidate_b,
            "viable": f.viable,
            "label_a_viable": int(f.viable == f.candidate_a),
        })
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} exact fork labels to {out}")


if __name__ == "__main__":
    main()
