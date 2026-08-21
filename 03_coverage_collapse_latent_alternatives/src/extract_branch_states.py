from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_utils import flip_query_target, full_prompt, mask_query_target


def parse_args():
    p = argparse.ArgumentParser(description="Extract first-fork hidden states and native candidate margins.")
    p.add_argument("--forks", default="artifacts/forks.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--condition", choices=("original", "target_flip", "target_blind"), default="original")
    p.add_argument("--exclude-problem-ids", default=None, help="CSV/text file containing problem_id values to exclude.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prefix-variant", type=int, default=0, choices=(0, 1, 2))
    p.add_argument("--batch-size", type=int, default=1, help="Currently fixed at 1 to avoid padding-position ambiguity.")
    p.add_argument("--output-dir", default="artifacts/states")
    return p.parse_args()


def load_jsonl(path: str):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def load_excluded(path: str | None) -> set[int]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "problem_id" not in df.columns:
            raise ValueError(f"{p} must contain problem_id")
        return set(df["problem_id"].astype(int).tolist())
    vals = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            vals.add(int(line))
    return vals


def candidate_token_ids(tokenizer, candidate: str) -> list[int]:
    ids = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Candidate {candidate!r} tokenized to nothing")
    return ids


def candidate_embedding(model, tokenizer, candidate: str) -> np.ndarray:
    ids = candidate_token_ids(tokenizer, candidate)
    weight = model.get_input_embeddings().weight
    emb = weight[torch.tensor(ids, device=weight.device)]
    return emb.float().mean(dim=0).detach().cpu().numpy()


@torch.inference_mode()
def continuation_logprob(model, tokenizer, prompt_ids: torch.Tensor, candidate: str, first_logits: torch.Tensor) -> float:
    """Teacher-forced log p(' '+candidate | prompt), robust to multi-token candidates."""
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


def true_viable_margin(a_minus_b: float, label_a_viable: int) -> float:
    return a_minus_b if label_a_viable else -a_minus_b


def conditioned_question_and_label(r: dict, condition: str) -> tuple[str, int, str]:
    y = int(r["label_a_viable"])
    if condition == "original":
        return r["question"], y, r["target"]
    if condition == "target_flip":
        q = flip_query_target(r["question"], r["target"], r["alternative_target"])
        return q, 1 - y, r["alternative_target"]
    if condition == "target_blind":
        q = mask_query_target(r["question"], r["target"], r["control_target"])
        return q, y, r["control_target"]
    raise ValueError(condition)


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Extractor currently requires --batch-size 1")

    excluded = load_excluded(args.exclude_problem_ids)
    rows = [r for r in load_jsonl(args.forks) if int(r["problem_id"]) not in excluded]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No fork examples selected after exclusions")

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
    true_margin = np.empty(len(rows), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int8)
    original_labels = np.empty(len(rows), dtype=np.int8)
    problem_ids = np.empty(len(rows), dtype=np.int32)
    query_targets = []

    for i, r in enumerate(tqdm(rows, desc=f"{args.tag}:{args.condition}")):
        question, y, query_target = conditioned_question_and_label(r, args.condition)
        text = full_prompt(question, prefix_variant=args.prefix_variant)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        prompt_ids = enc["input_ids"].to(device)
        with torch.inference_mode():
            out = model(prompt_ids, output_hidden_states=True, use_cache=False)

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
        labels[i] = y
        original_labels[i] = int(r["label_a_viable"])
        true_margin[i] = true_viable_margin(margin[i], y)
        problem_ids[i] = int(r["problem_id"])
        query_targets.append(query_target)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.tag}_{args.condition}.npz"
    np.savez_compressed(
        path,
        tag=np.array(args.tag),
        condition=np.array(args.condition),
        model=np.array(args.model),
        problem_id=problem_ids,
        label_a_viable=labels,
        original_label_a_viable=original_labels,
        query_target=np.asarray(query_targets),
        hidden=hidden,
        candidate_embedding_diff=emb_diff,
        output_logprob_margin_a_minus_b=margin,
        output_true_viable_margin=true_margin,
        block_layers=np.arange(n_layers, dtype=np.int16),
        prefix_variant=np.array(args.prefix_variant, dtype=np.int8),
        tie_word_embeddings=np.array(int(bool(getattr(model.config, "tie_word_embeddings", False))), dtype=np.int8),
    )
    print(
        f"saved {path}: {len(rows)} graphs, {n_layers} layers, hidden dim={dim}, "
        f"native-choice-acc={(true_margin > 0).mean():.3f}, "
        f"tied-embeddings={bool(getattr(model.config, 'tie_word_embeddings', False))}"
    )


if __name__ == "__main__":
    main()
