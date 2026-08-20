#!/usr/bin/env python3
"""Teacher-forced suffix NLL using the final model's actual chat template.

Unlike the initial implementation, this does not concatenate raw prompt text
without a chat template. We form the model's user-message prefix with
`apply_chat_template(..., add_generation_prompt=True)` and score only assistant
trace tokens after that boundary.
"""
from __future__ import annotations

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import read_jsonl, write_jsonl, split_reasoning_steps


def cut(trace: str, frac: float) -> tuple[str, str]:
    steps = split_reasoning_steps(trace)
    if not steps:
        return "", ""
    if frac <= 0:
        return "", "\n".join(steps)
    n = max(1, min(math.ceil(frac * len(steps)), len(steps) - 1))
    return "\n".join(steps[:n]), "\n".join(steps[n:])


def ids(tok, text: str) -> torch.Tensor:
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids


@torch.inference_mode()
def suffix_nll(model, tok, user_prompt: str, prefix: str, suffix: str) -> tuple[float, int]:
    base = tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prefix_ids = ids(tok, prefix + ("\n" if prefix else ""))
    suffix_ids = ids(tok, suffix)
    full = torch.cat([base, prefix_ids, suffix_ids], dim=1).to(model.device)
    suffix_start = base.shape[1] + prefix_ids.shape[1]

    logits = model(full).logits[:, :-1]
    target = full[:, 1:]
    logp = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    # logp index j predicts token full[j+1], so token `suffix_start` begins at j=suffix_start-1.
    vals = logp[:, max(0, suffix_start - 1) :]
    if vals.numel() == 0:
        return float("nan"), 0
    return float(-vals.mean().item()), int(vals.numel())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fractions", default="0,0.10,0.25,0.50")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    from common import stable_hash_int

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    fracs = [float(x) for x in args.fractions.split(",") if x.strip()]

    out = []
    for r in read_jsonl(args.input):
        pid = str(r["problem_id"])
        if stable_hash_int(pid) % args.num_shards != args.shard_index:
            continue
        trace = r.get("trace") or r.get("old_correct_trace") or r.get("verified_correct_trace")
        if not trace:
            continue
        # Use original problem prompt, not a re-entry prompt that already embeds prefix.
        user_prompt = str(r["prompt"])
        for frac in fracs:
            pref, suff = cut(str(trace), frac)
            if not suff:
                continue
            loss, n = suffix_nll(model, tok, user_prompt, pref, suff)
            out.append(
                {
                    "problem_id": pid,
                    "group": r.get("group"),
                    "source": r.get("source", "trace"),
                    "prefix_fraction": frac,
                    "prefix": pref,
                    "suffix_nll": loss,
                    "suffix_tokens": n,
                }
            )
    write_jsonl(args.output, out)
    print(f"rows={len(out)} shard={args.shard_index}/{args.num_shards}")


if __name__ == "__main__":
    main()
