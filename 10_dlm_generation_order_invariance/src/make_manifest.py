from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from schema import PuzzleSpec, write_jsonl
from sudoku import blank_indices, candidate_counts, make_unique_puzzle, random_spatial_transform


def encode_grid(grid: tuple[int, ...]) -> str:
    return "".join(str(v) for v in grid)


def build_manifest(seed: int, n_discovery: int, n_confirmation: int, blanks: int, transforms_per_puzzle: int) -> list[PuzzleSpec]:
    rng = random.Random(seed)
    records: list[PuzzleSpec] = []
    total = n_discovery + n_confirmation
    for k in range(total):
        puzzle, solution = make_unique_puzzle(rng, blanks=blanks)
        transforms = [random_spatial_transform(rng).as_dict() for _ in range(transforms_per_puzzle)]
        records.append(
            PuzzleSpec(
                puzzle_id=f"sudoku-{k:04d}",
                puzzle=encode_grid(puzzle),
                solution=encode_grid(solution),
                split="discovery" if k < n_discovery else "confirmation",
                blank_indices=list(blank_indices(puzzle)),
                candidate_counts={str(i): c for i, c in candidate_counts(puzzle).items()},
                transforms=transforms,
            )
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="LOCKED_CONFIG.json")
    ap.add_argument("--out", default="data/manifest.jsonl")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    records = build_manifest(
        seed=cfg["manifest_seed"],
        n_discovery=cfg["n_discovery_puzzles"],
        n_confirmation=cfg["n_confirmation_puzzles"],
        blanks=cfg["blanks_per_puzzle"],
        transforms_per_puzzle=cfg["transforms_per_puzzle"],
    )
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} puzzles to {args.out}")


if __name__ == "__main__":
    main()
