from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from instrumented_llada import _exact_digit_token_ids, build_sequence, make_prompt
from sudoku import blank_indices, format_puzzle, is_valid_solution, make_unique_puzzle, random_spatial_transform, solve


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
        assert {t.map_index(i) for i in blank_indices(puzzle)} == set(blank_indices(t.apply(puzzle)))
    print("[ok] sudoku solver/generator/isomorphisms")

    if not args.skip_model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        ids = _exact_digit_token_ids(tok)
        if getattr(tok, "mask_token_id", None) is not None:
            assert int(tok.mask_token_id) == int(cfg["mask_id"]), (tok.mask_token_id, cfg["mask_id"])
        mask_repr = tok.decode([cfg["mask_id"]], skip_special_tokens=False)
        if not mask_repr:
            raise RuntimeError("configured mask token decodes to an empty string")

        prompt_ids = make_prompt(tok, format_puzzle(puzzle))
        seq, _, cell_positions = build_sequence(tok, prompt_ids, puzzle, cfg["mask_id"])
        assert len(cell_positions) == 81 and len(set(cell_positions)) == 81
        blank_pos = {cell_positions[i] for i in blank_indices(puzzle)}
        actual_mask_pos = {i for i, token in enumerate(seq) if int(token) == int(cfg["mask_id"])}
        if actual_mask_pos != blank_pos:
            raise RuntimeError(
                "mask token appears outside the intended blank cell slots; "
                f"extra={sorted(actual_mask_pos - blank_pos)[:5]} missing={sorted(blank_pos - actual_mask_pos)[:5]}"
            )
        print(f"[ok] exact single-token digits: {ids}")
        print(f"[ok] mask_id={cfg['mask_id']} token={mask_repr!r}")
        print(f"[ok] fixed template has {len(seq)} tokens and exactly {len(blank_pos)} mutable masks")
    print("preflight passed")


if __name__ == "__main__":
    main()
