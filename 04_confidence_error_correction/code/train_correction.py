#!/usr/bin/env python3
"""Full-parameter corrective SFT with exact exposure cycles.

Designed for independent single-GPU jobs (or ordinary Accelerate DDP within one
fast node). The optimizer persists across cycles; a checkpoint is saved after
every cycle.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TokenizedSFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.examples = []
        eos = tokenizer.eos_token_id
        for row in rows:
            messages = [{"role": "user", "content": row["prompt"]}]
            try:
                prompt_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
            except Exception:
                prompt_ids = tokenizer.encode(row["prompt"] + "\nAnswer:", add_special_tokens=True)
            if hasattr(prompt_ids, "get") and prompt_ids.get("input_ids") is not None:
                prompt_ids = prompt_ids["input_ids"]
            if isinstance(prompt_ids, str):
                prompt_ids = tokenizer.encode(prompt_ids, add_special_tokens=False)
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
            if eos is not None:
                response_ids = response_ids + [eos]
            input_ids = list(prompt_ids) + list(response_ids)
            labels = [-100] * len(prompt_ids) + list(response_ids)

            if len(input_ids) > max_length:
                overflow = len(input_ids) - max_length
                if overflow >= len(prompt_ids):
                    continue
                prompt_ids = prompt_ids[overflow:]
                input_ids = list(prompt_ids) + list(response_ids)
                labels = [-100] * len(prompt_ids) + list(response_ids)

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "id": row["id"],
                    "pair_id": row["pair_id"],
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Collator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        ids, labels, mask = [], [], []
        for x in batch:
            n = len(x["input_ids"])
            pad = max_len - n
            ids.append(x["input_ids"] + [self.pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            mask.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def save_checkpoint(accelerator: Accelerator, model, tokenizer, path: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped.save_pretrained(
        path,
        state_dict=state_dict,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(path)
    accelerator.wait_for_everyone()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--split", choices=["discovery", "confirmation"], required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--per-device-batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = ap.parse_args()

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    set_seed(args.seed + accelerator.process_index)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    rows = [r for r in read_jsonl(args.train_data) if r["split"] == args.split]
    by_cycle = defaultdict(list)
    for r in rows:
        by_cycle[int(r["cycle"])].append(r)

    outdir = Path(args.output_dir)
    if accelerator.is_main_process:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.wait_for_everyone()

    log_path = outdir / "train_log.jsonl"
    global_update = 0

    for cycle in range(1, args.cycles + 1):
        cycle_rows = list(by_cycle.get(cycle, []))
        if not cycle_rows:
            raise RuntimeError(f"No training rows for cycle {cycle}")
        random.Random(args.seed * 1000 + cycle).shuffle(cycle_rows)
        dataset = TokenizedSFTDataset(cycle_rows, tokenizer, args.max_length)
        loader = DataLoader(
            dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=Collator(tokenizer),
            drop_last=False,
        )
        loader = accelerator.prepare(loader)

        model.train()
        loss_sum = 0.0
        microsteps = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_update += 1
                loss_sum += float(loss.detach().float().item())
                microsteps += 1

        ckpt = outdir / f"cycle_{cycle:02d}"
        save_checkpoint(accelerator, model, tokenizer, ckpt)

        mean_loss_tensor = torch.tensor(
            [loss_sum, microsteps], device=accelerator.device, dtype=torch.float64
        )
        reduced = accelerator.reduce(mean_loss_tensor, reduction="sum")
        if accelerator.is_main_process:
            total_loss, total_steps = reduced.tolist()
            row = {
                "cycle": cycle,
                "mean_microbatch_loss": total_loss / max(total_steps, 1),
                "global_optimizer_updates": global_update,
                "semantic_training_items": len(cycle_rows),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row))

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
