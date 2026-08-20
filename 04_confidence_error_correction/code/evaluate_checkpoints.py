#!/usr/bin/env python3
"""Evaluate base + every correction-cycle checkpoint on matched items."""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from score_mcq import score_items


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def flatten_pairs(path: str, split: str) -> list[dict]:
    out = []
    for p in read_jsonl(path):
        if p["split"] != split:
            continue
        for group in ("high", "low"):
            src = p[group]
            r = dict(src)
            r["pair_id"] = p["pair_id"]
            r["group"] = group
            r["split"] = split
            r["frozen_old_wrong"] = int(src["top_wrong"])
            r["frozen_base_p_correct"] = float(src["p_correct"])
            r["frozen_wrong_concentration"] = float(src["wrong_concentration"])
            r["frozen_question_token_count"] = int(src.get("question_token_count", 0))
            r["frozen_correct_answer_token_count"] = int(src.get("correct_answer_token_count", 0))
            out.append(r)
    return out


def checkpoint_paths(base_model: str, run_dir: Path) -> list[tuple[int, str]]:
    paths = [(0, base_model)]
    found = []
    for p in run_dir.glob("cycle_*"):
        m = re.fullmatch(r"cycle_(\d+)", p.name)
        if m:
            found.append((int(m.group(1)), str(p)))
    return paths + sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--split", choices=["discovery", "confirmation"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is None:
        m = re.search(r"seed(\d+)", str(args.run_dir))
        args.seed = int(m.group(1)) if m else 0
        print(f"inferred evaluation seed={args.seed} from run-dir")

    items = flatten_pairs(args.pairs, args.split)
    device = torch.device(args.device)
    all_rows = []
    baseline_diffs = []

    for cycle, model_path in checkpoint_paths(args.base_model, Path(args.run_dir)):
        print(f"Evaluating cycle={cycle} model={model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device).eval()

        scored = score_items(
            model,
            tokenizer,
            items,
            batch_size=args.batch_size,
            template="primary",
            device=device,
        )
        for r in scored:
            old_wrong = int(r["frozen_old_wrong"])
            if cycle == 0:
                baseline_diffs.append(abs(float(r["p_correct"]) - float(r["frozen_base_p_correct"])))
            all_rows.append(
                {
                    "id": r["id"],
                    "pair_id": r["pair_id"],
                    "group": r["group"],
                    "split": r["split"],
                    "category": r.get("category", "unknown"),
                    "seed": args.seed,
                    "cycle": cycle,
                    "base_p_correct": float(r["frozen_base_p_correct"]),
                    "wrong_concentration": float(r["frozen_wrong_concentration"]),
                    "question_token_count": int(r["frozen_question_token_count"]),
                    "correct_answer_token_count": int(r["frozen_correct_answer_token_count"]),
                    "p_correct": float(r["p_correct"]),
                    "p_old_wrong": float(r["semantic_probs"][old_wrong]),
                    "top1_correct": int(r["top1_correct"]),
                    "correct_rank": int(r["target_rank"]),
                    "answer_entropy": float(r["answer_entropy"]),
                }
            )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if baseline_diffs:
        print(
            "baseline reproduction: "
            f"mean_abs_diff={np.mean(baseline_diffs):.6g} "
            f"max_abs_diff={np.max(baseline_diffs):.6g}"
        )
        if np.max(baseline_diffs) > 1e-4:
            print("WARNING: cycle-0 scorer does not exactly reproduce frozen G-1 base scores")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
