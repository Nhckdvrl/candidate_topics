from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Pythia log-likelihoods and hidden states for one checkpoint.")
    p.add_argument("--model", default="EleutherAI/pythia-410m")
    p.add_argument("--step", type=int, required=True, help="Pythia training step; loaded as revision=step{N}.")
    p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--split", default="train")
    p.add_argument("--num-examples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--positions-per-text", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="artifacts/checkpoints")
    return p.parse_args()


def choose_block_layers(n_layers: int) -> list[int]:
    # Transformer block indices, not including the embedding state in HF's hidden_states tuple.
    raw = [round((n_layers - 1) * f) for f in (0.25, 0.50, 0.75, 1.0)]
    return sorted(set(int(x) for x in raw))


def byte_truncate(text: str, max_bytes: int = 4096) -> str:
    # Keep a deterministic byte cap so a few very long rows do not dominate pilot runtime.
    b = text.encode("utf-8")[:max_bytes]
    return b.decode("utf-8", errors="ignore")


def load_fixed_texts(dataset_name: str, split: str, n: int, seed: int) -> tuple[list[int], list[str]]:
    ds = load_dataset(dataset_name, split=split)
    if "text" not in ds.column_names:
        raise ValueError(f"Dataset {dataset_name} must contain a 'text' column, got {ds.column_names}")
    rng = np.random.default_rng(seed)
    candidate = np.arange(len(ds))
    rng.shuffle(candidate)
    ids: list[int] = []
    texts: list[str] = []
    for idx in candidate:
        text = ds[int(idx)]["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        ids.append(int(idx))
        texts.append(byte_truncate(text))
        if len(texts) == n:
            break
    if len(texts) < n:
        raise RuntimeError(f"Only found {len(texts)} non-empty texts; requested {n}")
    return ids, texts


def quantile_positions(length: int, k: int) -> np.ndarray:
    if length <= 1:
        return np.zeros(k, dtype=np.int64)
    lo, hi = 1, length - 1
    if k == 1:
        return np.array([(lo + hi) // 2], dtype=np.int64)
    return np.rint(np.linspace(lo, hi, num=k)).astype(np.int64)


def torch_dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    revision = f"step{args.step}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step{args.step}.npz"
    meta_path = out_dir / f"step{args.step}.json"

    ids, texts = load_fixed_texts(args.dataset, args.split, args.num_examples, args.seed)
    byte_lengths = np.empty(args.num_examples, dtype=np.int32)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device if args.device != "cuda" else "auto",
    ).eval()

    n_layers = int(model.config.num_hidden_layers)
    block_layers = choose_block_layers(n_layers)
    hidden_dim = int(model.config.hidden_size)
    n_obs = args.num_examples * args.positions_per_text

    log_likelihood = np.empty(args.num_examples, dtype=np.float64)
    token_lengths = np.empty(args.num_examples, dtype=np.int32)
    sampled_positions = np.empty((args.num_examples, args.positions_per_text), dtype=np.int32)
    hidden = np.empty((n_obs, len(block_layers), hidden_dim), dtype=np.float16)
    obs_example_id = np.empty(n_obs, dtype=np.int32)
    obs_position = np.empty(n_obs, dtype=np.int32)

    write_obs = 0
    for start in tqdm(range(0, args.num_examples, args.batch_size), desc=f"step {args.step}"):
        stop = min(start + args.batch_size, args.num_examples)
        batch_texts = texts[start:stop]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_tokens,
        )
        device = next(model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = out.logits[:, :-1].float()
            targets = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:].bool()
            token_logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            seq_ll = (token_logp * target_mask).sum(dim=1)

        for bi in range(stop - start):
            ex = start + bi
            length = int(attention_mask[bi].sum().item())
            token_lengths[ex] = length
            effective_text = tokenizer.decode(input_ids[bi, :length].detach().cpu(), skip_special_tokens=True)
            byte_lengths[ex] = len(effective_text.encode("utf-8"))
            log_likelihood[ex] = float(seq_ll[bi].item())
            pos = quantile_positions(length, args.positions_per_text)
            sampled_positions[ex] = pos

            for p in pos:
                for li, block_idx in enumerate(block_layers):
                    # HF hidden_states[0] is embedding output; block k is at k+1.
                    v = out.hidden_states[block_idx + 1][bi, int(p)].detach().float().cpu().numpy()
                    hidden[write_obs, li] = v.astype(np.float16)
                obs_example_id[write_obs] = ex
                obs_position[write_obs] = int(p)
                write_obs += 1

        del out, logits, token_logp, seq_ll
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez_compressed(
        out_path,
        step=np.array(args.step, dtype=np.int64),
        dataset_row_ids=np.asarray(ids, dtype=np.int64),
        byte_lengths=byte_lengths,
        token_lengths=token_lengths,
        log_likelihood=log_likelihood,
        sampled_positions=sampled_positions,
        block_layers=np.asarray(block_layers, dtype=np.int32),
        hidden=hidden,
        obs_example_id=obs_example_id,
        obs_position=obs_position,
    )
    meta = {
        "model": args.model,
        "revision": revision,
        "dataset": args.dataset,
        "split": args.split,
        "num_examples": args.num_examples,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "positions_per_text": args.positions_per_text,
        "block_layers": block_layers,
        "hidden_dim": hidden_dim,
        "mean_input_bytes": float(byte_lengths.mean()),
        "output": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
