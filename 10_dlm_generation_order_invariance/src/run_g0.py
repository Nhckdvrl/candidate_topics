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


def variant_key(rec: dict, variant_id: str, remasking: str) -> tuple[str, str, str, str]:
    return (rec["split"], rec["puzzle_id"], variant_id, remasking)


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
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip already completed trace keys without rerunning inference")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("choose at most one of --resume and --overwrite")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards>=1 and 0<=shard_index<num_shards")

    cfg = json.loads(Path(args.config).read_text())
    split_rows = [r for r in read_jsonl(args.manifest) if r["split"] == args.split]
    indexed_rows = [(i, r) for i, r in enumerate(split_rows) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        indexed_rows = indexed_rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"{out} already exists; use --resume or --overwrite explicitly")
    done = _existing_keys(out) if args.resume else set()
    mode = "w" if args.overwrite or not out.exists() else "a"

    n_repeat = 0 if args.skip_controls or args.split != "discovery" else int(cfg.get("same_serialization_repeat_puzzles", 0))
    n_random = 0 if args.skip_controls or args.split != "discovery" else int(cfg.get("random_control_puzzles", 0))

    # Do not pay the model-loading cost if this shard is already complete.
    pending = False
    for split_idx, rec in indexed_rows:
        keys = [variant_key(rec, "identity", "low_confidence")]
        if split_idx < n_repeat:
            keys.append(variant_key(rec, "identity-repeat", "low_confidence"))
        keys.extend(variant_key(rec, f"iso-{i}", "low_confidence") for i in range(len(rec["transforms"])))
        if split_idx < n_random:
            keys.append(variant_key(rec, "random-control", "random"))
        if any(k not in done for k in keys):
            pending = True
            break
    if not pending:
        print("all requested trace keys already complete; nothing to do")
        return

    model, tokenizer = load_model_and_tokenizer(cfg["model_id"], device=args.device, dtype=cfg["dtype"])

    with out.open(mode, encoding="utf-8") as f:
        def emit(trace: TraceRecord) -> None:
            payload = json.loads(trace.to_json())
            key = record_key(payload)
            if key in done:
                raise RuntimeError(f"internal error: attempted to emit already-completed key {key}")
            f.write(trace.to_json() + "\n")
            f.flush()
            done.add(key)

        def maybe_run(rec, puzzle, solution, transform, variant_id, remasking, seed):
            key = variant_key(rec, variant_id, remasking)
            if key in done:
                return False
            emit(run_variant(model, tokenizer, rec, puzzle, solution, transform, variant_id, cfg, remasking, seed))
            return True

        for local_n, (split_idx, rec) in enumerate(indexed_rows):
            puzzle = decode_grid(rec["puzzle"])
            solution = decode_grid(rec["solution"])
            base_seed = cfg["decode_seed"] + split_idx * 1000

            maybe_run(rec, puzzle, solution, None, "identity", "low_confidence", base_seed)

            if split_idx < n_repeat:
                maybe_run(rec, puzzle, solution, None, "identity-repeat", "low_confidence", base_seed + 777)

            for t_idx, t_dict in enumerate(rec["transforms"]):
                t = SudokuTransform.from_dict(t_dict)
                tp, ts = t.apply(puzzle), t.apply(solution)
                maybe_run(rec, tp, ts, t, f"iso-{t_idx}", "low_confidence", base_seed)

            if split_idx < n_random:
                maybe_run(rec, puzzle, solution, None, "random-control", "random", base_seed + 900)

            print(f"[{local_n + 1}/{len(indexed_rows)} shard={args.shard_index}/{args.num_shards}] {rec['puzzle_id']}")


if __name__ == "__main__":
    main()
