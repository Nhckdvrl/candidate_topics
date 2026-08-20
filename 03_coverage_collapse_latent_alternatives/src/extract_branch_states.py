from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_utils import full_prompt

def parse_args():
    p = argparse.ArgumentParser(description="Extract first-fork hidden states and output-accessibility baselines.")
    p.add_argument("--forks", default="artifacts/forks.jsonl")
    p.add_argument("--model", default="unsloth/Qwen2.5-0.5B", help="Base model or an official SFT checkpoint directory.")
    p.add_argument("--tag", required=True, help="Short checkpoint tag, e.g. base, e01, e04, e16")
    p.add_argument("--batch-size", type=int, default=1, help="Currently 1; kept explicit to avoid accidental padding-state changes.")
    p.add_argument("--output-dir", default="artifacts/states")
    return p.parse_args()


def load_jsonl(path: str):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def candidate_token_ids(tokenizer, candidate: str) -> list[int]:
    ids = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Candidate {candidate!r} tokenized to nothing")
    return ids


def candidate_embedding(model, tokenizer, candidate: str) -> np.ndarray:
    ids = candidate_token_ids(tokenizer, candidate)
    emb = model.get_input_embeddings().weight[torch.tensor(ids, device=model.get_input_embeddings().weight.device)]
    return emb.float().mean(dim=0).detach().cpu().numpy()


@torch.inference_mode()
def continuation_logprob(model, tokenizer, prompt_ids: torch.Tensor, candidate: str, first_logits: torch.Tensor) -> float:
    """Teacher-forced log p(' '+candidate | prompt), robust to multi-token candidate strings."""
    ids = candidate_token_ids(tokenizer, candidate)
    total = float(F.log_softmax(first_logits.float(), dim=-1)[ids[0]].item())
    if len(ids) == 1:
        return total
    seq = prompt_ids
    prev = ids[0]
    for nxt in ids[1:]:
        seq = torch.cat([seq, torch.tensor([[prev]], device=seq.device, dtype=seq.dtype)], dim=1)
        logits = model(seq).logits[0, -1]
        total += float(F.log_softmax(logits.float(), dim=-1)[nxt].item())
        prev = nxt
    return total


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("G0 extractor currently requires --batch-size 1")
    rows = load_jsonl(args.forks)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device
    n_layers = int(model.config.num_hidden_layers)
    dim = int(model.config.hidden_size)

    hidden = np.empty((len(rows), n_layers, dim), dtype=np.float16)
    emb_diff = np.empty((len(rows), dim), dtype=np.float16)
    margin = np.empty(len(rows), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int8)
    problem_ids = np.empty(len(rows), dtype=np.int32)

    for i, r in enumerate(tqdm(rows, desc=args.tag)):
        text = full_prompt(r["question"])
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        prompt_ids = enc["input_ids"].to(device)
        out = model(prompt_ids, output_hidden_states=True)
        # hidden_states[0] is embedding output; block k is hidden_states[k+1].
        for li in range(n_layers):
            hidden[i, li] = out.hidden_states[li + 1][0, -1].float().cpu().numpy().astype(np.float16)

        a, b = r["candidate_a"], r["candidate_b"]
        ea = candidate_embedding(model, tokenizer, a)
        eb = candidate_embedding(model, tokenizer, b)
        emb_diff[i] = (ea - eb).astype(np.float16)
        first_logits = out.logits[0, -1]
        la = continuation_logprob(model, tokenizer, prompt_ids, a, first_logits)
        lb = continuation_logprob(model, tokenizer, prompt_ids, b, first_logits)
        margin[i] = la - lb
        labels[i] = int(r["label_a_viable"])
        problem_ids[i] = int(r["problem_id"])

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.tag}.npz"
    np.savez_compressed(
        path,
        tag=np.array(args.tag),
        model=np.array(args.model),
        problem_id=problem_ids,
        label_a_viable=labels,
        hidden=hidden,
        candidate_embedding_diff=emb_diff,
        output_logprob_margin_a_minus_b=margin,
        block_layers=np.arange(n_layers, dtype=np.int16),
    )
    print(f"saved {path}: {len(rows)} graphs, {n_layers} layers, hidden dim={dim}")


if __name__ == "__main__":
    main()
