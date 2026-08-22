from __future__ import annotations

import argparse
import json
from pathlib import Path

from instrumented_llada import decode_fixed_slots, load_model_and_tokenizer
from schema import TraceRecord, read_jsonl
from sudoku import SudokuTransform, blank_indices, format_puzzle, is_valid_solution


def decode_grid(s: str) -> tuple[int, ...]:
    if len(s) != 81 or any(ch not in "0123456789" for ch in s):
        raise ValueError("grid string must contain exactly 81 digits")
    return tuple(int(ch) for ch in s)


def encode_grid(grid) -> str:
    return "".join(str(int(v)) for v in grid)


def record_key(rec: dict) -> tuple[str, str, str, str]:
    return (rec["split"], rec["puzzle_id"], rec["variant_id"], rec["remasking"])


def run_variant(model, tokenizer, rec: dict, puzzle, solution, transform, variant_id, cfg, remasking, seed):
    result = decode_fixed_slots(
        model,
        tokenizer,
        puzzle,
        format_puzzle(puzzle),
        mask_id=cfg["mask_id"],
        remasking=remasking,
        temperature=cfg["temperature"],
        seed=seed,
    )
    predicted = tuple(result.predicted_digits)
    return TraceRecord(
        puzzle_id=rec["puzzle_id"],
        variant_id=variant_id,
        split=rec["split"],
        remasking=remasking,
        puzzle=encode_grid(puzzle),
        solution=encode_grid(solution),
        transform=transform.as_dict() if transform is not None else None,
        blank_indices=list(blank_indices(puzzle)),
        predicted_digits=list(predicted),
        finalization_step={str(k): v for k, v in result.finalization_step.items()},
        confidence_at_finalization={str(k): v for k, v in result.confidence_at_finalization.items()},
        valid_solution=is_valid_solution(predicted),
        exact_solution=predicted == tuple(solution),
        metadata={
            "seed": seed,
            "model_id": cfg["model_id"],
            "temperature": cfg["temperature"],
            "protocol_version": cfg["protocol_version"],
            "native_digit_argmax_fraction": result.native_digit_argmax_fraction,
        },
    )


def _existing_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for rec in read_jsonl(path):
        key = record_key(rec)
        if key in keys:
            raise RuntimeError(f"duplicate trace already present in {path}: {key}")
        keys.add(key)
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="LOCKED_CONFIG.json")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", default="results/g0_traces.jsonl")
    ap.add_argument("--split", choices=["discovery", "confirmation"], default="discovery")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="skip already completed trace keys")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("choose at most one of --resume and --overwrite")

    cfg = json.loads(Path(args.config).read_text())
    rows = [r for r in read_jsonl(args.manifest) if r["split"] == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]
    model, tokenizer = load_model_and_tokenizer(cfg["model_id"], device=args.device, dtype=cfg["dtype"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"{out} already exists; use --resume or --overwrite explicitly")
    done = _existing_keys(out) if args.resume else set()
    mode = "w" if args.overwrite or not out.exists() else "a"

    n_repeat = 0 if args.skip_controls or args.split != "discovery" else int(cfg.get("same_serialization_repeat_puzzles", 0))
    n_random = 0 if args.skip_controls or args.split != "discovery" else int(cfg.get("random_control_puzzles", 0))

    with out.open(mode, encoding="utf-8") as f:
        def emit(trace: TraceRecord) -> None:
            key = record_key(json.loads(trace.to_json()))
            if key in done:
                return
            f.write(trace.to_json() + "\n")
            f.flush()
            done.add(key)

        for row_n, rec in enumerate(rows):
            puzzle = decode_grid(rec["puzzle"])
            solution = decode_grid(rec["solution"])
            base_seed = cfg["decode_seed"] + row_n * 1000

            base = run_variant(model, tokenizer, rec, puzzle, solution, None, "identity", cfg, "low_confidence", base_seed)
            emit(base)

            if row_n < n_repeat:
                repeat = run_variant(
                    model, tokenizer, rec, puzzle, solution, None,
                    "identity-repeat", cfg, "low_confidence", base_seed + 777,
                )
                emit(repeat)

            for t_idx, t_dict in enumerate(rec["transforms"]):
                t = SudokuTransform.from_dict(t_dict)
                tp, ts = t.apply(puzzle), t.apply(solution)
                tr = run_variant(
                    model, tokenizer, rec, tp, ts, t,
                    f"iso-{t_idx}", cfg, "low_confidence", base_seed,
                )
                emit(tr)

            if row_n < n_random:
                random_rec = run_variant(
                    model, tokenizer, rec, puzzle, solution, None,
                    "random-control", cfg, "random", base_seed + 900,
                )
                emit(random_rec)

            print(f"[{row_n + 1}/{len(rows)}] {rec['puzzle_id']}")


if __name__ == "__main__":
    main()
