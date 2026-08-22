"""Evaluate ordinary exact-grid accuracy for the G1/v4 9x9 corpus."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def parse_grid(text: str) -> list[list[int]] | None:
    # The seed protocol requests a Python-style 2D array. Parse only the first
    # bracketed object and never execute model output.
    start = text.find("[")
    if start < 0:
        return None
    try:
        value = ast.literal_eval(text[start:])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, list) or len(value) != 9:
        return None
    if any(not isinstance(row, list) or len(row) != 9 for row in value):
        return None
    if any(not isinstance(x, int) for row in value for x in row):
        return None
    return value


def exact_match(text: str, solution: list[int]) -> bool:
    grid = parse_grid(text)
    return grid == [solution[r * 9 : (r + 1) * 9] for r in range(9)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--data", default="data/g1_v4/test.jsonl", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = read_rows(args.data)
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    by_id = {row["id"]: row for row in predictions}
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("prediction IDs do not exactly match the frozen evaluation set")
    results = []
    for row in rows:
        pred = by_id[row["id"]]
        ok = exact_match(pred.get("prediction", ""), row["solution"])
        results.append({"id": row["id"], "exact": ok, "prediction": pred.get("prediction", "")})
    summary = {
        "protocol_version": "g1-v4-dream7b-9x9-seed-aligned",
        "n": len(results),
        "exact": sum(r["exact"] for r in results),
        "exact_rate": sum(r["exact"] for r in results) / len(results),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
