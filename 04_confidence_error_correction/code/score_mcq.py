#!/usr/bin/env python3
"""Distribution-level permutation-debiased MCQ scoring for Topic 04 G-1v2.

G-1v1 averaged mapped probabilities arithmetically and required 8/10 top-wrong
argmax stability. That mixed semantic commitment with option-position
susceptibility and mechanically selected against truly diffuse wrong beliefs.

G-1v2 keeps every mapped permutation distribution, then defines the primary
semantic distribution by mean log-probability (normalized geometric mean)
across a balanced permutation family. Position susceptibility is retained as a
separate diagnostic, not used as the treatment itself.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcq_utils import (
    arithmetic_mean_distribution,
    build_prompt_records,
    geometric_mean_distribution,
    label_token_sequences,
    map_permuted_probs_to_semantic,
    permutation_susceptibility,
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


def score_items(
    model,
    tokenizer,
    items: list[dict],
    batch_size: int,
    template: str,
    permutation_scheme: str,
    device,
) -> list[dict]:
    if not items:
        return []
    ks = {len(x["choices"]) for x in items}
    if len(ks) != 1:
        raise ValueError(f"score_items requires fixed K; got {sorted(ks)}")
    k = next(iter(ks))
    records = build_prompt_records(
        tokenizer, items, template, permutation_scheme=permutation_scheme
    )
    label_seqs = label_token_sequences(tokenizer, k)
    single_token = all(len(x) == 1 for x in label_seqs)
    print(
        f"K={k} prompts={len(records)} scheme={permutation_scheme} "
        f"label_token_lengths={[len(x) for x in label_seqs]}"
    )

    if single_token:
        local_probs, label_mass, greedy_is_label = score_single_token_labels_batched(
            model, tokenizer, records, k, batch_size, device
        )
    else:
        print(
            "WARNING: labels are multi-token; exact candidate scoring is used, "
            "but label_mass/greedy_is_label diagnostics are unavailable."
        )
        local_probs = score_multi_token_labels_slow(model, tokenizer, records, k, device)
        label_mass = [None] * len(records)
        greedy_is_label = [None] * len(records)

    mapped_by_item: dict[int, list[tuple[int, list[float], float | None, int | None]]] = defaultdict(list)
    for rec, probs, mass, greedy_ok in zip(records, local_probs, label_mass, greedy_is_label):
        mapped = map_permuted_probs_to_semantic(probs, rec.permutation)
        mapped_by_item[rec.item_index].append(
            (rec.permutation_index, mapped, mass, greedy_ok)
        )

    out = []
    for i, item in enumerate(items):
        ordered = sorted(mapped_by_item[i], key=lambda z: z[0])
        mapped = [x[1] for x in ordered]
        masses = [x[2] for x in ordered if x[2] is not None]
        greedy_flags = [x[3] for x in ordered if x[3] is not None]
        if len(mapped) != k:
            raise RuntimeError(f"Incomplete balanced permutation set for {item['id']}")

        debiased = geometric_mean_distribution(mapped)
        arithmetic = arithmetic_mean_distribution(mapped)
        metrics = semantic_metrics(debiased, int(item["answer"]))
        arithmetic_metrics = semantic_metrics(arithmetic, int(item["answer"]))

        a = int(item["answer"])
        top_wrong_by_perm = [
            max((j for j in range(k) if j != a), key=lambda j: semantic[j])
            for semantic in mapped
        ]
        modal_wrong, count = Counter(top_wrong_by_perm).most_common(1)[0]

        qtok = len(tokenizer.encode(str(item["question"]), add_special_tokens=False))
        atok = len(
            tokenizer.encode(str(item["choices"][a]), add_special_tokens=False)
        )

        out.append(
            {
                **item,
                "choice_count": k,
                "semantic_probs": debiased,
                "semantic_probs_debiased": debiased,
                "semantic_probs_arithmetic": arithmetic,
                **metrics,
                "v1_arithmetic_p_correct": arithmetic_metrics["p_correct"],
                "v1_arithmetic_wrong_concentration": arithmetic_metrics["wrong_concentration"],
                "v1_arithmetic_top_wrong": arithmetic_metrics["top_wrong"],
                "position_susceptibility_js": permutation_susceptibility(mapped, debiased),
                "modal_top_wrong": int(modal_wrong),
                "top_wrong_stability": count / len(top_wrong_by_perm),
                "top_wrong_by_perm": top_wrong_by_perm,
                "permutation_probs": mapped,
                "mean_label_mass": (
                    sum(float(x) for x in masses) / len(masses) if masses else None
                ),
                "min_label_mass": min((float(x) for x in masses), default=None),
                "greedy_is_allowed_label_rate": (
                    sum(int(x) for x in greedy_flags) / len(greedy_flags)
                    if greedy_flags
                    else None
                ),
                "question_token_count": qtok,
                "correct_answer_token_count": atok,
                "prompt_template": template,
                "permutation_scheme": permutation_scheme,
                "measurement_version": "g1v2_logmean",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--prompt-template", choices=["primary", "alternate"], default="primary")
    ap.add_argument(
        "--permutation-scheme",
        choices=["cyclic", "hashed_cyclic"],
        default="cyclic",
        help="cyclic=primary family A; hashed_cyclic=independent balanced audit family B",
    )
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
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

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
                permutation_scheme=args.permutation_scheme,
                device=device,
            )
        )
    out.sort(key=lambda x: x["id"])
    write_jsonl(args.output, out)
    print(f"wrote {len(out)} scored items -> {args.output}")


if __name__ == "__main__":
    main()
