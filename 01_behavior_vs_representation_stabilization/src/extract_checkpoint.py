from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract behavior or one-layer residual representations for one Pythia checkpoint.")
    p.add_argument("--model", default="EleutherAI/pythia-410m")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--corpus", default="artifacts/corpus/pile_chunks_seed42.jsonl")
    p.add_argument("--mode", choices=["behavior", "representation"], required=True)
    p.add_argument("--layer", default="middle", help="Transformer block index or 'middle'.")
    p.add_argument("--positions-per-text", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--output-dir", default="artifacts/checkpoints")
    return p.parse_args()


def torch_dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_corpus(path: Path) -> tuple[list[int], list[str], np.ndarray, str]:
    payload = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    ids = [int(r["example_id"]) for r in rows]
    if ids != list(range(len(rows))):
        raise ValueError("Corpus example_id values must be contiguous and ordered")
    texts = [r["text"] for r in rows]
    byte_lengths = np.asarray([int(r["byte_length"]) for r in rows], dtype=np.int32)
    for text, n_bytes in zip(texts, byte_lengths):
        if len(text.encode("utf-8")) != int(n_bytes):
            raise ValueError("Corpus byte_length does not match UTF-8 payload")
    return ids, texts, byte_lengths, digest


def quantile_positions(length: int, k: int) -> np.ndarray:
    if length < 3:
        raise ValueError(f"Sequence too short for representation sampling: {length}")
    lo, hi = 1, length - 2
    if k == 1:
        return np.asarray([(lo + hi) // 2], dtype=np.int32)
    return np.rint(np.linspace(lo, hi, num=k)).astype(np.int32)


def resolve_layer(model, layer_arg: str) -> int:
    n_layers = int(model.config.num_hidden_layers)
    if layer_arg == "middle":
        return n_layers // 2
    idx = int(layer_arg)
    if idx < 0 or idx >= n_layers:
        raise ValueError(f"Layer {idx} outside [0, {n_layers - 1}]")
    return idx


def main() -> None:
    args = parse_args()
    revision = f"step{args.step}"
    corpus_path = Path(args.corpus)
    ids, texts, byte_lengths, corpus_sha = load_corpus(corpus_path)
    n_examples = len(texts)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        torch_dtype=torch_dtype(args.dtype),
    ).to(device).eval()
    layer_idx = resolve_layer(model, args.layer)
    hidden_dim = int(model.config.hidden_size)

    captured: dict[str, torch.Tensor] = {}
    hook = None
    if args.mode == "representation":
        # Pythia is GPT-NeoX. The input to a transformer block is the residual stream
        # immediately before that block (resid_pre), matching the hook convention used
        # by Crosscoding Through Time.
        block = model.gpt_neox.layers[layer_idx]

        def capture_resid_pre(_module, inputs):
            captured["resid_pre"] = inputs[0].detach()

        hook = block.register_forward_pre_hook(capture_resid_pre)

    log_likelihood = np.empty(n_examples, dtype=np.float64)
    token_lengths = np.empty(n_examples, dtype=np.int32)

    if args.mode == "representation":
        n_obs = n_examples * args.positions_per_text
        hidden = np.empty((n_obs, hidden_dim), dtype=np.float16)
        obs_example_id = np.empty(n_obs, dtype=np.int32)
        obs_position = np.empty(n_obs, dtype=np.int32)
        write_obs = 0

    try:
        for start in tqdm(range(0, n_examples, args.batch_size), desc=f"{args.mode} step {args.step}"):
            stop = min(start + args.batch_size, n_examples)
            enc = tokenizer(
                texts[start:stop],
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            lengths = enc["attention_mask"].sum(dim=1)
            if int(lengths.max()) > args.max_tokens:
                raise ValueError(
                    f"Corpus produced {int(lengths.max())} tokens, above --max-tokens={args.max_tokens}; "
                    "fix corpus construction rather than silently truncating."
                )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            captured.clear()

            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
                logits = out.logits[:, :-1].float()
                targets = input_ids[:, 1:]
                target_mask = attention_mask[:, 1:].bool()
                token_logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                seq_ll = (token_logp * target_mask).sum(dim=1)

            if args.mode == "representation" and "resid_pre" not in captured:
                raise RuntimeError("Residual hook did not fire")

            for bi in range(stop - start):
                ex = start + bi
                length = int(attention_mask[bi].sum().item())
                token_lengths[ex] = length
                log_likelihood[ex] = float(seq_ll[bi].item())

                if args.mode == "representation":
                    positions = quantile_positions(length, args.positions_per_text)
                    h = captured["resid_pre"][bi]
                    for pos in positions:
                        hidden[write_obs] = h[int(pos)].float().cpu().numpy().astype(np.float16)
                        obs_example_id[write_obs] = ex
                        obs_position[write_obs] = int(pos)
                        write_obs += 1

            del out, logits, token_logp, seq_ll
    finally:
        if hook is not None:
            hook.remove()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "behavior" if args.mode == "behavior" else f"repr_l{layer_idx}"
    out_path = out_dir / f"step{args.step}_{suffix}.npz"

    payload = {
        "step": np.asarray(args.step, dtype=np.int64),
        "example_id": np.asarray(ids, dtype=np.int32),
        "byte_lengths": byte_lengths,
        "token_lengths": token_lengths,
        "log_likelihood": log_likelihood,
    }
    if args.mode == "representation":
        payload.update(
            hidden=hidden,
            obs_example_id=obs_example_id,
            obs_position=obs_position,
            layer_idx=np.asarray(layer_idx, dtype=np.int32),
        )
    np.savez(out_path, **payload)

    meta = {
        "model": args.model,
        "revision": revision,
        "mode": args.mode,
        "layer_idx": layer_idx if args.mode == "representation" else None,
        "num_examples": n_examples,
        "positions_per_text": args.positions_per_text if args.mode == "representation" else 0,
        "mean_input_bytes": float(byte_lengths.mean()),
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_sha,
        "output": str(out_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
