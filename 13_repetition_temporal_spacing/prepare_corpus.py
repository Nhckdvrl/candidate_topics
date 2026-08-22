#!/usr/bin/env python3
"""Prepare a fixed-length, exact-document corpus for the Topic-13 G-0.

We intentionally keep only documents with >= seq_len-1 Qwen3 tokens, take exactly
seq_len-1 tokens, and append EOS. Every training atom is therefore exactly one
fixed-length document block. This removes variable-document-length timing as a
confound when repeated identities are moved between temporal slots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def split_bucket(doc_id: str, seed: int = 0, modulo: int = 1000) -> int:
    b = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()[:8]
    return int.from_bytes(b, "big") % modulo


def id64(doc_id: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest()[:8], "big"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-blocks", type=int, required=True)
    p.add_argument("--eval-blocks", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--dataset", default="HuggingFaceTB/smollm-corpus")
    p.add_argument("--subset", default="fineweb-edu-dedup")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--eval-permille", type=int, default=25)
    p.add_argument("--shuffle-buffer", type=int, default=10000)
    p.add_argument("--stream-seed", type=int, default=0)
    p.add_argument("--tokenize-batch", type=int, default=64)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "corpus_meta.json"
    train_path = args.out_dir / "train_blocks.npy"
    eval_path = args.out_dir / "eval_blocks.npy"
    ids_path = args.out_dir / "train_doc_ids.npy"
    eval_ids_path = args.out_dir / "eval_doc_ids.npy"

    if meta_path.exists() and train_path.exists() and eval_path.exists() and ids_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("train_blocks") >= args.train_blocks and old.get("eval_blocks") >= args.eval_blocks and old.get("seq_len") == args.seq_len and old.get("tokenizer") == args.tokenizer:
            print("reusing prepared corpus:", meta_path)
            return

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.eos_token_id is None:
        raise RuntimeError("Qwen tokenizer has no EOS token")
    if len(tok) != 151670:
        raise RuntimeError(f"Expected Qwen3 vocab size 151670, got {len(tok)}")

    train_mm = np.lib.format.open_memmap(train_path, mode="w+", dtype=np.uint32, shape=(args.train_blocks, args.seq_len))
    eval_mm = np.lib.format.open_memmap(eval_path, mode="w+", dtype=np.uint32, shape=(args.eval_blocks, args.seq_len))
    train_ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.uint64, shape=(args.train_blocks,))
    eval_ids = np.lib.format.open_memmap(eval_ids_path, mode="w+", dtype=np.uint64, shape=(args.eval_blocks,))

    ds = load_dataset(args.dataset, args.subset, split="train", streaming=True)
    ds = ds.shuffle(seed=args.stream_seed, buffer_size=args.shuffle_buffer)

    n_train = n_eval = scanned = long_enough = 0
    batch_rows = []

    def consume_batch(rows):
        nonlocal n_train, n_eval, long_enough
        if not rows:
            return
        texts = [r[1] for r in rows]
        encoded = tok(texts, add_special_tokens=False, truncation=True, max_length=args.seq_len - 1)["input_ids"]
        for (doc_id, _), ids in zip(rows, encoded):
            if len(ids) < args.seq_len - 1:
                continue
            long_enough += 1
            block = np.asarray(ids + [tok.eos_token_id], dtype=np.uint32)
            assert len(block) == args.seq_len
            is_eval = split_bucket(doc_id, args.split_seed) < args.eval_permille
            if is_eval and n_eval < args.eval_blocks:
                eval_mm[n_eval] = block
                eval_ids[n_eval] = id64(doc_id)
                n_eval += 1
            elif (not is_eval) and n_train < args.train_blocks:
                train_mm[n_train] = block
                train_ids[n_train] = id64(doc_id)
                n_train += 1

    for row in ds:
        if n_train >= args.train_blocks and n_eval >= args.eval_blocks:
            break
        scanned += 1
        text = row.get("text", "")
        doc_id = str(row.get("id", f"row-{scanned}"))
        batch_rows.append((doc_id, text))
        if len(batch_rows) >= args.tokenize_batch:
            consume_batch(batch_rows)
            batch_rows = []
        if scanned % 10000 == 0:
            print(f"scanned={scanned:,} long={long_enough:,} train={n_train:,}/{args.train_blocks:,} eval={n_eval:,}/{args.eval_blocks:,}", flush=True)
    if (n_train < args.train_blocks or n_eval < args.eval_blocks) and batch_rows:
        consume_batch(batch_rows)

    train_mm.flush(); eval_mm.flush(); train_ids.flush(); eval_ids.flush()
    if n_train != args.train_blocks or n_eval != args.eval_blocks:
        raise RuntimeError(f"Dataset stream exhausted early: train={n_train}, eval={n_eval}")
    if len(np.unique(np.asarray(train_ids))) != args.train_blocks:
        raise RuntimeError("duplicate document IDs entered the prepared train corpus")
    if np.intersect1d(np.asarray(train_ids), np.asarray(eval_ids)).size:
        raise RuntimeError("train/eval document ID collision")

    meta = {
        "dataset": args.dataset,
        "subset": args.subset,
        "tokenizer": args.tokenizer,
        "vocab_size": len(tok),
        "seq_len": args.seq_len,
        "train_blocks": args.train_blocks,
        "eval_blocks": args.eval_blocks,
        "train_tokens": args.train_blocks * args.seq_len,
        "eval_tokens": args.eval_blocks * args.seq_len,
        "split_seed": args.split_seed,
        "eval_permille": args.eval_permille,
        "stream_seed": args.stream_seed,
        "shuffle_buffer": args.shuffle_buffer,
        "tokenize_batch": args.tokenize_batch,
        "scanned_documents": scanned,
        "long_enough_documents": long_enough,
        "fixed_length_note": "first seq_len-1 Qwen3 tokens + EOS; short documents excluded"
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
