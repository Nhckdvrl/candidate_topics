#!/usr/bin/env python3
"""Score Topic-11 trajectories with the paper's LLaDA confidence protocol.

Primary paper-compatible score:
  one teacher-forced forward pass on prompt + prescribed trajectory, then mean
  same-position probability over output tokens.

Identification guardrail:
  also report mean probability on the *unchanged continuation tokens* only.
  If a consistency effect appears only on the manipulated announcement token but
  not on unchanged downstream tokens, the structural-consistency story is not
  considered established.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hamming(a: list[int], b: list[int]) -> int:
    if len(a) != len(b):
        return 10**9
    return sum(x != y for x, y in zip(a, b))


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def encode_sample(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    prompt_ids = encode_prompt(tokenizer, row["prompt"])
    announce_ids = tokenizer(row["announcement_text"], add_special_tokens=False)["input_ids"]
    tail_ids = tokenizer(row["continuation_text"], add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + announce_ids + tail_ids
    return {
        **row,
        "prompt_ids": prompt_ids,
        "announce_ids": announce_ids,
        "tail_ids": tail_ids,
        "input_ids": input_ids,
        "prompt_len": len(prompt_ids),
        "announce_len": len(announce_ids),
        "tail_len": len(tail_ids),
        "output_len": len(announce_ids) + len(tail_ids),
    }


def eligible_orientation(cells: dict[str, dict[str, Any]], max_intervention_tokens: int) -> tuple[bool, str]:
    if set(cells) != {"CC", "IC", "CW", "IW"}:
        return False, "missing_cells"
    cc, ic, cw, iw = (cells[k] for k in ("CC", "IC", "CW", "IW"))

    # Factorial minimality after tokenization.
    if cc["prompt_ids"] != ic["prompt_ids"] or cw["prompt_ids"] != iw["prompt_ids"]:
        return False, "prompt_not_fixed_within_correctness"
    if cc["tail_ids"] != ic["tail_ids"] or cw["tail_ids"] != iw["tail_ids"]:
        return False, "tail_not_fixed_within_consistency_contrast"
    if not (cc["tail_ids"] == cw["tail_ids"] == ic["tail_ids"] == iw["tail_ids"]):
        return False, "tail_not_common_across_factorial"
    if cc["announce_ids"] != cw["announce_ids"] or ic["announce_ids"] != iw["announce_ids"]:
        return False, "output_not_fixed_within_correctness"

    d_announce = hamming(cc["announce_ids"], ic["announce_ids"])
    d_prompt = hamming(cc["prompt_ids"], cw["prompt_ids"])
    if d_announce < 1 or d_announce > max_intervention_tokens:
        return False, f"announcement_token_diff={d_announce}"
    if d_prompt < 1 or d_prompt > max_intervention_tokens:
        return False, f"prompt_token_diff={d_prompt}"

    if len({r["output_len"] for r in cells.values()}) != 1:
        return False, "output_length_mismatch"
    if len({len(r["input_ids"]) for r in cells.values()}) != 1:
        return False, "total_length_mismatch"
    return True, "ok"


def select_eligible_pairs(rows: list[dict[str, Any]], max_intervention_tokens: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        grouped[(int(r["pair_id"]), int(r["orientation"]))][r["cell"]] = r

    orientation_status: dict[tuple[int, int], tuple[bool, str]] = {}
    reason_counts: dict[str, int] = defaultdict(int)
    for key, cells in grouped.items():
        status = eligible_orientation(cells, max_intervention_tokens)
        orientation_status[key] = status
        reason_counts[status[1]] += 1

    pair_ok: dict[int, bool] = {}
    pair_ids = sorted({p for p, _ in grouped})
    for pair_id in pair_ids:
        pair_ok[pair_id] = all(orientation_status.get((pair_id, o), (False, "missing"))[0] for o in (0, 1))

    kept = [r for r in rows if pair_ok.get(int(r["pair_id"]), False)]
    reason_counts["eligible_pairs"] = sum(pair_ok.values())
    reason_counts["total_pairs"] = len(pair_ok)
    return kept, dict(reason_counts)


def batch_records(records: list[dict[str, Any]], batch_size: int):
    # Sort by length to reduce padding while retaining pair ids in output.
    records = sorted(records, key=lambda r: len(r["input_ids"]))
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def score_batch(model, tokenizer, batch: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")
        pad_id = tokenizer.eos_token_id

    max_len = max(len(r["input_ids"]) for r in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    out_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
    announce_mask = torch.zeros_like(out_mask)
    tail_mask = torch.zeros_like(out_mask)

    # Right padding is sufficient because scoring is non-causal and attention_mask is supplied.
    for i, r in enumerate(batch):
        ids = torch.tensor(r["input_ids"], dtype=torch.long, device=device)
        n = len(r["input_ids"])
        input_ids[i, :n] = ids
        attention[i, :n] = 1
        p = r["prompt_len"]
        a = r["announce_len"]
        out_mask[i, p:n] = True
        announce_mask[i, p : p + a] = True
        tail_mask[i, p + a : n] = True

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        target_logits = torch.gather(logits, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        log_z = torch.logsumexp(logits, dim=-1)
        probs = torch.exp(target_logits - log_z).float()

    results: list[dict[str, Any]] = []
    for i, r in enumerate(batch):
        def mean_for(mask: torch.Tensor) -> float:
            vals = probs[i][mask[i]]
            return float(vals.mean().item())

        results.append(
            {
                "pair_id": int(r["pair_id"]),
                "orientation": int(r["orientation"]),
                "cell": r["cell"],
                "internal_consistent": bool(r["internal_consistent"]),
                "externally_correct": bool(r["externally_correct"]),
                "branch_anchor": int(r["branch_anchor"]),
                "alternate_anchor": int(r["alternate_anchor"]),
                "prompt_anchor": int(r["prompt_anchor"]),
                "announced_anchor": int(r["announced_anchor"]),
                "reported_final": int(r["reported_final"]),
                "true_final": int(r["true_final"]),
                "confidence_full": mean_for(out_mask),
                "confidence_announcement": mean_for(announce_mask),
                "confidence_tail": mean_for(tail_mask),
                "prompt_len": int(r["prompt_len"]),
                "output_len": int(r["output_len"]),
                "announce_len": int(r["announce_len"]),
                "tail_len": int(r["tail_len"]),
            }
        )

    del logits, target_logits, log_z, probs
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--max-intervention-tokens", type=int, default=1)
    p.add_argument("--min-pairs", type=int, default=128)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("Require 0 <= shard-id < num-shards")

    raw = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoded = [encode_sample(tokenizer, r) for r in raw]
    eligible, stats = select_eligible_pairs(encoded, args.max_intervention_tokens)
    n_pairs = stats["eligible_pairs"]
    print("tokenization audit:", json.dumps(stats, sort_keys=True))
    if n_pairs < args.min_pairs:
        raise SystemExit(f"Only {n_pairs} eligible mirrored pairs; need >= {args.min_pairs}. Expand design or anchor range.")

    # Keep complete mirrored pairs on the same worker.
    eligible = [r for r in eligible if int(r["pair_id"]) % args.num_shards == args.shard_id]
    if not eligible:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
        print("no records assigned to this shard")
        return

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype)
    model = model.to(args.device).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as f:
        for batch in batch_records(eligible, args.batch_size):
            for result in score_batch(model, tokenizer, batch, args.device):
                f.write(json.dumps(result, sort_keys=True) + "\n")
                n += 1
    print(f"scored {n} samples on shard {args.shard_id}/{args.num_shards} -> {args.output}")


if __name__ == "__main__":
    main()
