#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topic12.ablation import get_decoder_layers, parse_layer_spec, residual_scale_layer, shard_layers
from topic12.benchmarks import grade_math, load_tasks, make_prompt


def parse_args():
    p = argparse.ArgumentParser(description="Topic 12 layer-necessity generation sweep")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--tasks", default="math500,gsm8k")
    p.add_argument("--n-per-task", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--layers", default="all", help="all | none | 0,2,5 | 8-12")
    p.add_argument("--layer-shard-index", type=int, default=0)
    p.add_argument("--layer-shard-count", type=int, default=1)
    p.add_argument("--residual-scale", type=float, default=0.0)
    p.add_argument("--include-baseline", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=1536)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_version(name: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def complete_file(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip()) == expected_rows
    except Exception:
        return False


def generate_condition(
    *,
    model,
    tokenizer,
    task_examples,
    layer: int | None,
    residual_scale: float,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    ctx = (
        nullcontext()
        if layer is None
        else residual_scale_layer(model, layer_index=layer, scale=residual_scale)
    )

    with ctx:
        for task, examples in task_examples.items():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                prompts = [make_prompt(ex) for ex in batch]
                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_tokens,
                    add_special_tokens=True,
                )
                input_width = enc["input_ids"].shape[1]
                enc = {k: v.to(device) for k, v in enc.items()}

                t0 = time.time()
                with torch.inference_mode():
                    generated = model.generate(
                        **enc,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                elapsed = time.time() - t0

                for i, ex in enumerate(batch):
                    new_ids = generated[i, input_width:]
                    response = tokenizer.decode(new_ids, skip_special_tokens=True)
                    if tokenizer.pad_token_id is None:
                        n_new = int(new_ids.numel())
                    else:
                        n_new = int((new_ids != tokenizer.pad_token_id).sum().item())
                    truncated = n_new >= max_new_tokens
                    correct, parse_ok, grade_error = grade_math(response, ex.gold)
                    rows.append(
                        {
                            "task": task,
                            "uid": ex.uid,
                            "problem": ex.problem,
                            "gold": ex.gold,
                            "metadata": ex.metadata,
                            "layer": layer,
                            "residual_scale": 1.0 if layer is None else residual_scale,
                            "prompt": prompts[i],
                            "response": response,
                            "correct": bool(correct),
                            "parse_ok": bool(parse_ok),
                            "grade_error": grade_error,
                            "new_tokens": n_new,
                            "truncated": bool(truncated),
                            "batch_generation_seconds": elapsed,
                        }
                    )
    return rows


def main():
    args = parse_args()
    if not (0.0 <= args.residual_scale <= 1.0):
        raise SystemExit("--residual-scale must be in [0,1]")

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(args.device)
    model.eval()

    num_layers = len(get_decoder_layers(model))
    all_selected = parse_layer_spec(args.layers, num_layers)
    selected = shard_layers(
        all_selected,
        shard_index=args.layer_shard_index,
        shard_count=args.layer_shard_count,
    )

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    task_examples = load_tasks(tasks, args.n_per_task, args.seed)
    expected_rows = sum(len(v) for v in task_examples.values())

    ledger = [
        ex.to_dict()
        for task in sorted(task_examples)
        for ex in task_examples[task]
    ]
    write_jsonl_atomic(
        out_dir / f"ledger_worker_{args.layer_shard_index}.jsonl",
        ledger,
    )

    model_commit = getattr(model.config, "_commit_hash", None)
    first_example = next(iter(task_examples.values()))[0]
    manifest = {
        "model": args.model,
        "model_commit": model_commit,
        "num_layers": num_layers,
        "tasks": tasks,
        "n_per_task": args.n_per_task,
        "seed": args.seed,
        "selected_layers": selected,
        "all_requested_layers": all_selected,
        "layer_shard_index": args.layer_shard_index,
        "layer_shard_count": args.layer_shard_count,
        "residual_scale": args.residual_scale,
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "math_verify": package_version("math-verify"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "prompt_sha256": hashlib.sha256(
            make_prompt(first_example).encode("utf-8")
        ).hexdigest(),
    }
    manifest_name = (
        "manifest_baseline.json"
        if args.include_baseline and not selected
        else f"manifest_worker_{args.layer_shard_index}.json"
    )
    (out_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    conditions: list[tuple[int | None, Path]] = []
    if args.include_baseline:
        conditions.append((None, out_dir / "baseline.jsonl"))
    for layer in selected:
        conditions.append((layer, out_dir / f"layer_{layer:02d}.jsonl"))

    for layer, path in conditions:
        if not args.overwrite and complete_file(path, expected_rows):
            print(f"[skip] complete: {path}")
            continue
        label = "baseline" if layer is None else f"layer {layer}"
        print(f"[run] {label}; rows={expected_rows}; scale={args.residual_scale}")
        rows = generate_condition(
            model=model,
            tokenizer=tokenizer,
            task_examples=task_examples,
            layer=layer,
            residual_scale=args.residual_scale,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        write_jsonl_atomic(path, rows)
        acc = np.mean([r["correct"] for r in rows]) if rows else float("nan")
        trunc = np.mean([r["truncated"] for r in rows]) if rows else float("nan")
        print(f"[done] {label}: accuracy={acc:.4f}, truncation={trunc:.3%} -> {path}")

    print("Topic 12 sweep worker complete.")


if __name__ == "__main__":
    main()
