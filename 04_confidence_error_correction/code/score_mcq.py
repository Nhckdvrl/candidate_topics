#!/usr/bin/env python3
"""Permutation-robust semantic MCQ scoring for Topic 04.

The scorer:
1. applies the model's chat template;
2. scores answer-label probabilities;
3. rotates every semantic option through every label position;
4. maps probabilities back to semantic option identity;
5. averages across rotations.

Supports deterministic dataset sharding for embarrassingly parallel inference.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcq_utils import (
    build_prompt_records,
    label_token_sequences,
    map_permuted_probs_to_semantic,
    score_multi_token_labels_slow,
    score_single_token_labels_batched,
    semantic_metrics,
)


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_items(model, tokenizer, items: list[dict], batch_size: int, template: str, device) -> list[dict]:
    if not items:
        return []
    ks = {len(x["choices"]) for x in items}
    if len(ks) != 1:
        raise ValueError(f"score_items requires a fixed K per invocation; got {sorted(ks)}")
    k = next(iter(ks))
    records = build_prompt_records(tokenizer, items, template)
    label_seqs = label_token_sequences(tokenizer, k)
    single_token = all(len(x) == 1 for x in label_seqs)
    print(f"K={k} prompts={len(records)} label_token_lengths={[len(x) for x in label_seqs]}")

    if single_token:
        local_probs = score_single_token_labels_batched(
            model, tokenizer, records, k, batch_size, device
        )
    else:
        print("WARNING: answer labels are multi-token; using exact slow fallback")
        local_probs = score_multi_token_labels_slow(model, tokenizer, records, k, device)

    mapped_by_item: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for rec, probs in zip(records, local_probs):
        mapped = map_permuted_probs_to_semantic(probs, rec.permutation)
        mapped_by_item[rec.item_index].append((rec.permutation_index, mapped))

    out = []
    for i, item in enumerate(items):
        mapped = [x[1] for x in sorted(mapped_by_item[i], key=lambda z: z[0])]
        if len(mapped) != len(item["choices"]):
            raise RuntimeError(f"Incomplete permutations for {item['id']}")
        avg = [sum(row[j] for row in mapped) / len(mapped) for j in range(k)]
        metrics = semantic_metrics(avg, int(item["answer"]))

        top_wrong_by_perm = []
        for semantic in mapped:
            a = int(item["answer"])
            top_wrong_by_perm.append(
                max((j for j in range(k) if j != a), key=lambda j: semantic[j])
            )
        modal_wrong, count = Counter(top_wrong_by_perm).most_common(1)[0]

        qtok = len(tokenizer.encode(str(item["question"]), add_special_tokens=False))
        atok = len(
            tokenizer.encode(str(item["choices"][int(item["answer"])]), add_special_tokens=False)
        )

        out.append(
            {
                **item,
                "choice_count": k,
                "semantic_probs": avg,
                **metrics,
                "modal_top_wrong": int(modal_wrong),
                "top_wrong_stability": count / len(top_wrong_by_perm),
                "top_wrong_by_perm": top_wrong_by_perm,
                "permutation_probs": mapped,
                "question_token_count": qtok,
                "correct_answer_token_count": atok,
                "prompt_template": template,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--prompt-template", choices=["primary", "alternate"], default="primary")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("bad shard index")

    items = read_jsonl(args.input)
    items = [r for i, r in enumerate(items) if i % args.num_shards == args.shard_index]
    if args.max_items is not None:
        items = items[: args.max_items]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Group by K so the script also works on replication pools with varying K.
    by_k: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        by_k[len(item["choices"])].append(item)

    out = []
    for k in sorted(by_k):
        out.extend(
            score_items(
                model,
                tokenizer,
                by_k[k],
                batch_size=args.batch_size,
                template=args.prompt_template,
                device=device,
            )
        )
    out.sort(key=lambda x: x["id"])
    write_jsonl(args.output, out)
    print(f"wrote {len(out)} scored items -> {args.output}")


if __name__ == "__main__":
    main()
