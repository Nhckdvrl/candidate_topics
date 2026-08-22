from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from sudoku4 import blank_indices, random_spatial_transform


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="LOCKED_CONFIG_V3.json")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="data/manifest_v3_4x4.jsonl")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    dataset = Path(args.dataset)
    if sha256(dataset) != cfg["dataset_sha256"]:
        raise RuntimeError("dataset SHA256 does not match LOCKED_CONFIG_V3.json")
    rows = list(csv.DictReader(dataset.open(newline="")))
    n_total = int(cfg["n_discovery_puzzles"]) + int(cfg["n_confirmation_puzzles"])
    if n_total > len(rows):
        raise RuntimeError("locked manifest budget exceeds published dataset")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows[:n_total]):
            split = "discovery" if i < cfg["n_discovery_puzzles"] else "confirmation"
            rng = random.Random(int(cfg["manifest_seed"]) + i)
            transforms = [random_spatial_transform(rng).as_dict() for _ in range(cfg["transforms_per_puzzle"])]
            payload = {
                "protocol_version": cfg["protocol_version"],
                "puzzle_id": f"sudoku4-{i:04d}",
                "split": split,
                "puzzle": row["Puzzle"],
                "solution": row["Solution"],
                "blank_indices": blank_indices(row["Puzzle"]),
                "transforms": transforms,
            }
            f.write(json.dumps(payload) + "\n")
    print(f"wrote {n_total} published 4x4 puzzles to {out} protocol={cfg['protocol_version']}")


if __name__ == "__main__":
    main()
