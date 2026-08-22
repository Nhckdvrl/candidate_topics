#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topic12.ablation import get_decoder_layers, residual_scale_layer
from topic12.benchmarks import Example, make_prompt
from run_ablation import DEFAULT_MODEL_REVISION


def main():
    p = argparse.ArgumentParser(description="Real-model preflight for Topic 12 layer hook")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--layer", type=int, default=10)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = args.model_revision.strip() or None
    if Path(args.model).exists():
        revision = None
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    layers = get_decoder_layers(model)
    if len(layers) != 28:
        raise SystemExit(f"FAIL: expected Qwen3-1.7B 28 layers, got {len(layers)}")

    ex = Example(task="preflight", uid="preflight", problem="What is 17 + 25?", gold="42", metadata={})
    prompt = make_prompt(ex)
    enc = tok(prompt, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        baseline = model(**enc, use_cache=False).logits[:, -1, :].float()
    with residual_scale_layer(model, args.layer, scale=1.0):
        with torch.inference_mode():
            identity = model(**enc, use_cache=False).logits[:, -1, :].float()
    max_identity_diff = float((baseline - identity).abs().max().item())
    if max_identity_diff != 0.0:
        raise SystemExit(f"FAIL: scale=1 hook is not exact identity; max logit diff={max_identity_diff}")

    with residual_scale_layer(model, args.layer, scale=0.0):
        with torch.inference_mode():
            bypass = model(**enc, use_cache=False).logits[:, -1, :].float()
    bypass_diff = float((baseline - bypass).abs().max().item())
    if bypass_diff <= 1e-6:
        raise SystemExit("FAIL: scale=0 bypass did not change logits; hook may not be active")

    eos = getattr(model.generation_config, "eos_token_id", None) or tok.eos_token_id
    with residual_scale_layer(model, args.layer, scale=0.0):
        with torch.inference_mode():
            out = model.generate(
                **enc, do_sample=False, max_new_tokens=8, use_cache=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=eos,
            )
    if out.shape[1] <= enc["input_ids"].shape[1]:
        raise SystemExit("FAIL: cached generation produced no continuation")

    print("PASS: real-model layer intervention preflight")
    print(f"  model commit: {getattr(model.config, '_commit_hash', None)}")
    print(f"  layers: {len(layers)}")
    print(f"  scale=1 max logit diff: {max_identity_diff}")
    print(f"  scale=0 max logit diff: {bypass_diff}")
    print("  cached generation: OK")


if __name__ == "__main__":
    main()
