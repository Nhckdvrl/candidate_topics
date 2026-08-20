from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a fixed byte-chunk Pile corpus for G0.")
    p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--split", default="train")
    p.add_argument("--num-examples", type=int, default=1000)
    p.add_argument("--chunk-bytes", type=int, default=1024)
    p.add_argument("--min-bytes", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="artifacts/corpus/pile_chunks_seed42.jsonl")
    return p.parse_args()


def valid_utf8_chunk(raw: bytes, start: int, chunk_bytes: int) -> str:
    piece = raw[start : start + chunk_bytes]
    return piece.decode("utf-8", errors="ignore").strip()


def reservoir_add(reservoir: list[dict], item: dict, seen: int, k: int, rng: np.random.Generator) -> None:
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = int(rng.integers(0, seen + 1))
    if j < k:
        reservoir[j] = item


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0 or args.chunk_bytes <= 0 or args.min_bytes <= 0:
        raise ValueError("num-examples/chunk-bytes/min-bytes must be positive")
    if args.min_bytes > args.chunk_bytes:
        raise ValueError("min-bytes cannot exceed chunk-bytes")

    ds = load_dataset(args.dataset, split=args.split)
    if "text" not in ds.column_names:
        raise ValueError(f"Dataset must contain a text column, got {ds.column_names}")

    rng = np.random.default_rng(args.seed)
    reservoir: list[dict] = []
    seen = 0

    for row_id, row in enumerate(ds):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        raw = text.encode("utf-8")
        for chunk_index, start in enumerate(range(0, len(raw), args.chunk_bytes)):
            chunk = valid_utf8_chunk(raw, start, args.chunk_bytes)
            n_bytes = len(chunk.encode("utf-8"))
            if n_bytes < args.min_bytes:
                continue
            item = {
                "source_row_id": row_id,
                "chunk_index": chunk_index,
                "byte_length": n_bytes,
                "text": chunk,
            }
            reservoir_add(reservoir, item, seen, args.num_examples, rng)
            seen += 1

    if len(reservoir) < args.num_examples:
        raise RuntimeError(f"Only found {len(reservoir)} eligible chunks; requested {args.num_examples}")

    # Stable order after reservoir sampling makes all checkpoint files directly comparable.
    reservoir.sort(key=lambda x: (x["source_row_id"], x["chunk_index"]))
    for example_id, item in enumerate(reservoir):
        item["example_id"] = example_id

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in reservoir)
    out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "num_examples": len(reservoir),
        "eligible_chunks_seen": seen,
        "chunk_bytes": args.chunk_bytes,
        "min_bytes": args.min_bytes,
        "seed": args.seed,
        "mean_bytes": float(np.mean([x["byte_length"] for x in reservoir])),
        "sha256": digest,
        "output": str(out),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
