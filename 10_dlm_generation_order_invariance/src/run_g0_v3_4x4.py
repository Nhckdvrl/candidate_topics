from __future__ import annotations

import argparse
import json
from pathlib import Path

from instrumented_llada import load_model_and_tokenizer
from run_published_4x4 import decode_one
from schema import read_jsonl
from sudoku4 import Sudoku4Transform


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="LOCKED_CONFIG_V3.json")
    ap.add_argument("--manifest", default="data/manifest_v3_4x4.jsonl")
    ap.add_argument("--split", choices=["discovery", "confirmation"], default="discovery")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/g0_v3_4x4.jsonl")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num-shards>=1 and 0<=shard-index<num-shards")
    cfg = json.loads(Path(args.config).read_text())
    rows = [r for r in read_jsonl(args.manifest) if r["split"] == args.split]
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit is not None:
        rows = rows[: args.limit]
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")
    model, tokenizer = load_model_and_tokenizer(cfg["model_id"], device=args.device, dtype=cfg["dtype"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for local_i, row in enumerate(rows):
            source_i = int(row["puzzle_id"].split("-")[-1])
            variants = [("identity", row["puzzle"], row["solution"], None)]
            if args.split == "discovery" and source_i < int(cfg.get("same_serialization_repeat_puzzles", 0)):
                variants.append(("identity-repeat", row["puzzle"], row["solution"], None))
            for i, transform_dict in enumerate(row["transforms"]):
                t = Sudoku4Transform.from_dict(transform_dict)
                variants.append((f"iso-{i}", t.apply(row["puzzle"]), t.apply(row["solution"]), t))
            for variant_id, puzzle, solution, transform in variants:
                result = decode_one(model, tokenizer, puzzle, solution, cfg)
                result.update({
                    "protocol_version": cfg["protocol_version"],
                    "puzzle_id": row["puzzle_id"],
                    "source_puzzle": row["puzzle"],
                    "source_solution": row["solution"],
                    "split": args.split,
                    "variant_id": variant_id,
                    "transform": transform.as_dict() if transform is not None else None,
                })
                f.write(json.dumps(result) + "\n")
                f.flush()
            print(f"[{local_i + 1}/{len(rows)} shard={args.shard_index}/{args.num_shards}] {row['puzzle_id']}", flush=True)


if __name__ == "__main__":
    main()
