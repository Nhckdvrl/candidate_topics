from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from instrumented_llada import _exact_digit_token_ids
from sudoku import is_valid_solution, make_unique_puzzle, random_spatial_transform, solve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="LOCKED_CONFIG.json")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    rng = random.Random(cfg["manifest_seed"])
    puzzle, solution = make_unique_puzzle(rng, blanks=cfg["blanks_per_puzzle"])
    assert is_valid_solution(solution)
    assert solve(puzzle, limit=2) == [solution]
    for _ in range(20):
        t = random_spatial_transform(rng)
        assert solve(t.apply(puzzle), limit=2) == [t.apply(solution)]
        assert len(set(t.map_index(i) for i in range(81))) == 81
    print("[ok] sudoku solver/generator/isomorphisms")

    if not args.skip_model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        ids = _exact_digit_token_ids(tok)
        print(f"[ok] exact single-token digits: {ids}")
        print(f"[ok] mask_id={cfg['mask_id']}")
    print("preflight passed")


if __name__ == "__main__":
    main()
