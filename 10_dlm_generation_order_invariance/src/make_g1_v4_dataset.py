"""Generate the auditable seed-aligned Dream-7B 9x9 Sudoku corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from sudoku import make_unique_puzzle


PROMPT = """Fill in the zeros in the matrix according to the following rules:
Given initial 9×9 matrix:
{puzzle}
Rules: Each row must contain the numbers 1-9 exactly once, each column must contain the numbers 1-9 exactly once, and each 3x3 subgrid must contain the numbers 1-9 exactly once.
Strict Requirement: Directly output the completed matrix in the format of a 2D array. Do not output any reasoning process, and do not include any explanatory text."""


def matrix_text(grid: tuple[int, ...]) -> str:
    return "[" + ",\n".join(
        "[" + ",".join(str(grid[r * 9 + c]) for c in range(9)) + "]"
        for r in range(9)
    ) + "]"


def zero_grid_text(grid: tuple[int, ...]) -> str:
    return "\n".join(
        " ".join(str(grid[r * 9 + c]) for c in range(9))
        for r in range(9)
    )


def record(puzzle: tuple[int, ...], solution: tuple[int, ...], index: int, split: str) -> dict:
    return {
        "id": f"g1-v4-sudoku-{index:03d}",
        "split": split,
        "puzzle": list(puzzle),
        "solution": list(solution),
        "blanks": puzzle.count(0),
        "prompt": PROMPT.format(puzzle=zero_grid_text(puzzle)),
        "response": matrix_text(solution),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/g1_v4")
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    blank_cycle = (35, 40, 45, 50, 55)
    rows = []
    for i in range(150):
        puzzle, solution = make_unique_puzzle(rng, blanks=blank_cycle[i % len(blank_cycle)])
        rows.append(record(puzzle, solution, i, "train" if i < 50 else "test"))
    for split in ("train", "test"):
        path = out / f"{split}.jsonl"
        with path.open("w") as f:
            for row in rows:
                if row["split"] == split:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"{path} sha256={sha256(path)}")
    meta = {
        "protocol_version": "g1-v4-dream7b-9x9-seed-aligned",
        "generator_seed": args.seed,
        "blank_count_cycle": list(blank_cycle),
        "total": len(rows),
        "train": 50,
        "test": 100,
        "train_sha256": sha256(out / "train.jsonl"),
        "test_sha256": sha256(out / "test.jsonl"),
        "provenance": "reconstructed-from-public-description",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
