#!/usr/bin/env python3
"""Run a frozen Topic 18 design on local Hugging Face causal LMs.

Example:
  .venv_clean2/bin/python run_local_models.py --design design.jsonl \
    --model qwen=/models/Qwen3-8B --output predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_model(value: str) -> tuple[str, str, str]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError(
            "--model must be FAMILY:MODEL_ID=LOCAL_PATH"
        )
    identity, path = value.split("=", 1)
    family, model_id = identity.split(":", 1)
    if not family or not model_id or not path:
        raise argparse.ArgumentTypeError(
            "--model must be FAMILY:MODEL_ID=LOCAL_PATH"
        )
    return family, model_id, path


def load_design(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "item_id" not in row or "prompt" not in row:
                raise ValueError(f"{path}:{lineno}: requires item_id and prompt")
            rows.append(row)
    if not rows:
        raise ValueError("empty design")
    return rows


def render_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
            # Qwen3 otherwise spends the frozen short output budget inside a
            # <think> block.  Other templates ignore this Jinja kwarg.
            enable_thinking=False,
        )
    return prompt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", action="append", type=parse_model, required=True,
                   help="repeat FAMILY:MODEL_ID=LOCAL_PATH for each panel member")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--allow-download", action="store_true",
                   help="default is local-files-only for reproducibility")
    p.add_argument("--resume", action="store_true",
                   help="append only missing model/item rows from an interrupted run")
    args = p.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch size and max new tokens must be positive")
    if len({model_id for _, model_id, _ in args.model}) != len(args.model):
        raise ValueError("model IDs must be unique")
    if len({family for family, _, _ in args.model}) != len(args.model):
        raise ValueError("each panel member must come from a distinct model family")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "torch/transformers are required; use the repository local .venv_clean2 environment"
        ) from exc

    design = load_design(args.design)
    existing: set[tuple[str, str]] = set()
    if args.output.exists():
        if not args.resume:
            raise ValueError(f"{args.output} exists; use --resume or choose a new output")
        with args.output.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                row = json.loads(line)
                key = (str(row["model_id"]), str(row["item_id"]))
                if key in existing:
                    raise ValueError(f"duplicate existing prediction at line {lineno}: {key}")
                existing.add(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    output_file = args.output.open("a" if args.resume else "w", encoding="utf-8")
    for model_family, model_id, model_path in args.model:
        pending = [row for row in design if (model_id, str(row["item_id"])) not in existing]
        if not pending:
            continue
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=not args.allow_download, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=not args.allow_download,
            trust_remote_code=False, torch_dtype="auto", device_map="auto",
        )
        model.eval()
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            prompts = [render_prompt(tokenizer, str(row["prompt"])) for row in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_width = encoded["input_ids"].shape[1]
            texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            for row, text in zip(batch, texts, strict=True):
                output_file.write(json.dumps({
                    "model_family": model_family, "model_id": model_id,
                    "model_revision": Path(model_path).name,
                    "item_id": row["item_id"], "output": text.strip(),
                }, ensure_ascii=False) + "\n")
                written += 1
            output_file.flush()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_file.close()
    print(json.dumps({"output": str(args.output), "models": len(args.model),
                      "predictions_written": written,
                      "existing_predictions": len(existing)}, indent=2))


if __name__ == "__main__":
    main()
