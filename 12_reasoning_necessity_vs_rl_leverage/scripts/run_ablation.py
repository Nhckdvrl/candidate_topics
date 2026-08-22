#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
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
from topic12.benchmarks import (
    DEFAULT_PROMPT_STYLE,
    GSM8K_REVISION,
    MATH500_REVISION,
    grade_math,
    load_tasks,
    make_prompt,
)

DEFAULT_MODEL_REVISION = "912d2727784ca0a6f718845aa14d4d9e5f48fe26"


def parse_args():
    p = argparse.ArgumentParser(description="Topic 12 layer-necessity generation sweep")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION,
                   help="Immutable HF revision. Set empty only for an explicitly staged local snapshot.")
    p.add_argument("--expect-layers", type=int, default=28)
    p.add_argument("--prompt-style", choices=["qwen_math_seed", "plain_math"], default=DEFAULT_PROMPT_STYLE)
    p.add_argument("--tasks", default="math500,gsm8k")
    p.add_argument("--n-per-task", type=int, default=256)
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


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_run_contract(out_dir: Path, contract: dict[str, Any]) -> str:
    """Prevent stale files from another scientific protocol being silently resumed."""
    contract_id = canonical_hash(contract)
    payload = {"contract_id": contract_id, **contract}
    path = out_dir / "run_contract.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("contract_id") != contract_id:
            raise RuntimeError(
                "Output directory already belongs to a different scientific contract. "
                "Use a new OUT directory; never mix conditions across protocols.\n"
                f"existing={old.get('contract_id')} requested={contract_id}"
            )
    else:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        try:
            tmp.replace(path)
        except FileNotFoundError:
            pass
        if json.loads(path.read_text(encoding="utf-8")).get("contract_id") != contract_id:
            raise RuntimeError("Concurrent worker created a different run contract")
    return contract_id


def complete_file(path: Path, expected_rows: int, contract_id: str) -> bool:
    if not path.exists():
        return False
    try:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return len(rows) == expected_rows and all(r.get("contract_id") == contract_id for r in rows)
    except Exception:
        return False


def generate_condition(
    *, model, tokenizer, task_examples, layer: int | None, residual_scale: float,
    batch_size: int, max_input_tokens: int, max_new_tokens: int, device: str,
    prompt_style: str, contract_id: str,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    ctx = nullcontext() if layer is None else residual_scale_layer(model, layer_index=layer, scale=residual_scale)
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id

    with ctx:
        for task, examples in task_examples.items():
            for start in range(0, len(examples), batch_size):
                batch = examples[start:start + batch_size]
                prompts = [make_prompt(ex, style=prompt_style) for ex in batch]
                raw_ids = tokenizer(prompts, padding=False, truncation=False, add_special_tokens=True)["input_ids"]
                input_lengths = [len(x) for x in raw_ids]
                enc = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_input_tokens, add_special_tokens=True,
                )
                input_width = enc["input_ids"].shape[1]
                enc = {k: v.to(device) for k, v in enc.items()}

                t0 = time.time()
                with torch.inference_mode():
                    generated = model.generate(
                        **enc, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
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
                    rows.append({
                        "contract_id": contract_id,
                        "task": task,
                        "uid": ex.uid,
                        "problem": ex.problem,
                        "gold": ex.gold,
                        "metadata": ex.metadata,
                        "layer": layer,
                        "residual_scale": 1.0 if layer is None else residual_scale,
                        "prompt_style": prompt_style,
                        "prompt": prompts[i],
                        "input_tokens": int(input_lengths[i]),
                        "input_truncated": bool(input_lengths[i] > max_input_tokens),
                        "response": response,
                        "correct": bool(correct),
                        "parse_ok": bool(parse_ok),
                        "grade_error": grade_error,
                        "new_tokens": n_new,
                        "truncated": bool(truncated),
                        "batch_generation_seconds": elapsed,
                    })
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

    revision = args.model_revision.strip() or None
    if Path(args.model).exists():
        revision = None
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation, trust_remote_code=args.trust_remote_code,
    )
    model.to(args.device)
    model.eval()

    num_layers = len(get_decoder_layers(model))
    if args.expect_layers and num_layers != args.expect_layers:
        raise RuntimeError(f"Expected {args.expect_layers} decoder layers, loaded {num_layers}")
    all_selected = parse_layer_spec(args.layers, num_layers)
    selected = shard_layers(all_selected, args.layer_shard_index, args.layer_shard_count)
    tasks = [x.strip().lower() for x in args.tasks.split(",") if x.strip()]
    task_examples = load_tasks(tasks, args.n_per_task, args.seed)
    expected_rows = sum(len(v) for v in task_examples.values())

    model_commit = getattr(model.config, "_commit_hash", None)
    contract = {
        "model": args.model,
        "model_source_is_local": Path(args.model).exists(),
        "requested_model_revision": args.model_revision,
        "resolved_model_commit": model_commit,
        "tasks": sorted(tasks),
        "dataset_revisions": {"math500": MATH500_REVISION, "gsm8k": GSM8K_REVISION},
        "n_per_task": args.n_per_task,
        "seed": args.seed,
        "prompt_style": args.prompt_style,
        "residual_scale": args.residual_scale,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "expected_layers": args.expect_layers,
    }
    contract_id = ensure_run_contract(out_dir, contract)

    ledger = [ex.to_dict() for task in sorted(task_examples) for ex in task_examples[task]]
    write_jsonl_atomic(out_dir / f"ledger_worker_{args.layer_shard_index}.jsonl", ledger)

    first_example = next(iter(task_examples.values()))[0]
    manifest = {
        "contract_id": contract_id,
        "model": args.model,
        "requested_model_revision": args.model_revision,
        "resolved_model_commit": model_commit,
        "num_layers": num_layers,
        "tasks": tasks,
        "n_per_task": args.n_per_task,
        "seed": args.seed,
        "selected_layers": selected,
        "all_requested_layers": all_selected,
        "layer_shard_index": args.layer_shard_index,
        "layer_shard_count": args.layer_shard_count,
        "residual_scale": args.residual_scale,
        "prompt_style": args.prompt_style,
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
        "prompt_sha256": hashlib.sha256(make_prompt(first_example, style=args.prompt_style).encode("utf-8")).hexdigest(),
    }
    manifest_name = "manifest_baseline.json" if args.include_baseline and not selected else f"manifest_worker_{args.layer_shard_index}.json"
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    conditions: list[tuple[int | None, Path]] = []
    if args.include_baseline:
        conditions.append((None, out_dir / "baseline.jsonl"))
    for layer in selected:
        conditions.append((layer, out_dir / f"layer_{layer:02d}.jsonl"))

    for layer, path in conditions:
        if not args.overwrite and complete_file(path, expected_rows, contract_id):
            print(f"[skip] complete under contract {contract_id[:12]}: {path}")
            continue
        label = "baseline" if layer is None else f"layer {layer}"
        print(f"[run] {label}; rows={expected_rows}; scale={args.residual_scale}; contract={contract_id[:12]}")
        rows = generate_condition(
            model=model, tokenizer=tokenizer, task_examples=task_examples, layer=layer,
            residual_scale=args.residual_scale, batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
            device=args.device, prompt_style=args.prompt_style, contract_id=contract_id,
        )
        write_jsonl_atomic(path, rows)
        acc = np.mean([r["correct"] for r in rows]) if rows else float("nan")
        trunc = np.mean([r["truncated"] for r in rows]) if rows else float("nan")
        print(f"[done] {label}: accuracy={acc:.4f}, truncation={trunc:.3%} -> {path}")

    print("Topic 12 sweep worker complete.")


if __name__ == "__main__":
    main()
