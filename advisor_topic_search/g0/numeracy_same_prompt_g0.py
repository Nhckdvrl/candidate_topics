#!/usr/bin/env python3
"""Frozen same-prompt G0 for the Numeracy representation-vs-access candidate.

Scientific prerequisite
-----------------------
On the exact same five-shot prompt and model, does the last-input-token residual
stream contain linearly decodable correct ranking information while the model's
greedy generation is wrong?

This script is deliberately narrow:
- primary model: Qwen/Qwen3-8B
- official seed-0 int_sci_compare + dec_sci_compare JSONL files
- exact 5-shot demonstrations from VCY019/Numeracy-Probing/src/verbalization.py
- hidden-state position: last input token only
- one logistic classifier per layer
- layer chosen by validation accuracy, earliest-layer tie break
- test evaluated once after layer selection
- hard regime fixed by seed paper: |log2(a/b)| < 0.1

It does NOT perform activation steering or patching.  If this prerequisite does
not pass, do not start mechanism work.

Expected usage
--------------
First generate the official data with the upstream repository, then run:

python advisor_topic_search/g0/numeracy_same_prompt_g0.py \
  --data-root /path/to/Numeracy-Probing/data \
  --model Qwen/Qwen3-8B \
  --out-dir numeracy_same_prompt_qwen3_8b

Dependencies: torch, transformers, numpy, scikit-learn, tqdm.
No paid API or human annotation is used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[×x*]\s*10\^?-?\d+)?")
DATASETS = ("int_sci_compare", "dec_sci_compare")

FEW_SHOT = {
    "int_sci_compare": [
        ("9.9 × 10^2", "100", 0),
        ("161230", "7.182 × 10^5", 1),
        ("713", "4.78 × 10^2", 0),
        ("1.354 × 10^6", "4906723", 1),
        ("20834", "6.5 × 10^3", 0),
    ],
    "dec_sci_compare": [
        ("9.9 × 10^2", "899.9", 0),
        ("161230.51", "7.182 × 10^5", 1),
        ("712.34", "4.78 × 10^2", 0),
        ("1.354 × 10^6", "4906723.2", 1),
        ("20834.17033", "6.5 × 10^3", 0),
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out-dir", type=Path, default=Path("numeracy_same_prompt_g0"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--train-limit", type=int, default=None, help="Smoke-test only; leave unset for frozen G0.")
    p.add_argument("--val-limit", type=int, default=None, help="Smoke-test only; leave unset for frozen G0.")
    p.add_argument("--test-limit", type=int, default=None, help="Smoke-test only; leave unset for frozen G0.")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_value(s: str) -> float:
    return float(eval(str(s).replace("×", "*").replace("x", "*").replace("^", "**").replace(",", "")))


def load_jsonl(path: Path, limit: int | None = None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def make_prompt(sample: dict, dataset_name: str) -> str:
    demos = []
    for a, b, ans_id in FEW_SHOT[dataset_name]:
        ans = (a, b)[ans_id]
        demos.append(f"Q: Which is larger, {a} or {b}? A: {ans}")
    return "\n".join(demos) + f"\nQ: Which is larger, {sample['a']} or {sample['b']}? A:"


def label(sample: dict) -> int:
    return int(parse_value(sample["a"]) > parse_value(sample["b"]))


def hard(sample: dict) -> bool:
    a, b = parse_value(sample["a"]), parse_value(sample["b"])
    return abs(math.log2(a / b)) < 0.1


def answer_position(sample: dict) -> str:
    return "a" if label(sample) == 1 else "b"


def parse_generated_number(text: str):
    m = NUMBER_RE.search(text)
    if not m:
        return None
    try:
        return parse_value(m.group(0))
    except Exception:
        return None


def generation_correct(sample: dict, completion: str) -> tuple[bool, bool]:
    pred = parse_generated_number(completion)
    if pred is None:
        return False, False
    a, b = parse_value(sample["a"]), parse_value(sample["b"])
    target = max(a, b)
    # Mirror upstream spirit while allowing tiny floating-point noise.
    ok = abs(pred - target) <= max(1e-3, 1e-8 * abs(target))
    return bool(ok), True


def first_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def extract_hidden(model, tokenizer, prompts, batch_size: int):
    """Return [N, L, D] float16 CPU array for last non-padding input token."""
    chunks = []
    device = first_device(model)
    for start in tqdm(range(0, len(prompts), batch_size), desc="hidden states"):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # hidden_states[0] is embedding output; use transformer layer outputs only.
        hs = out.hidden_states[1:]
        last_idx = attention_mask.sum(dim=1) - 1
        per_layer = []
        arange = torch.arange(input_ids.shape[0], device=device)
        for layer in hs:
            per_layer.append(layer[arange, last_idx, :].detach().to("cpu", dtype=torch.float16))
        # [B,L,D]
        chunks.append(torch.stack(per_layer, dim=1).numpy())
        del out, hs, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def generate(model, tokenizer, prompts, samples, batch_size: int, max_new_tokens: int):
    device = first_device(model)
    records = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="generation"):
        batch_prompts = prompts[start : start + batch_size]
        batch_samples = samples[start : start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            seq = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = tokenizer.batch_decode(seq[:, input_ids.shape[1] :], skip_special_tokens=True)
        for sample, completion in zip(batch_samples, completions):
            ok, parseable = generation_correct(sample, completion)
            records.append({
                "generation_correct": ok,
                "parseable": parseable,
                "completion": completion,
            })
        del seq, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def fit_probes(train_x, train_y, val_x, val_y):
    n_layers = train_x.shape[1]
    models, val_accs = [], []
    for layer in tqdm(range(n_layers), desc="fit probes"):
        clf = LogisticRegression(max_iter=10000, random_state=0)
        clf.fit(train_x[:, layer, :].astype(np.float32), train_y)
        pred = clf.predict(val_x[:, layer, :].astype(np.float32))
        models.append(clf)
        val_accs.append(float(accuracy_score(val_y, pred)))
    best = max(val_accs)
    # Earliest-layer tie break (within numerical precision).
    selected = next(i for i, a in enumerate(val_accs) if abs(a - best) < 1e-12)
    return models, val_accs, selected


def evaluate_dataset(name, data_root, model, tokenizer, args):
    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    splits = {}
    prompts = {}
    labels = {}
    hidden = {}

    for split in ("train", "val", "test"):
        rows = load_jsonl(data_root / name / f"{split}.jsonl", limits[split])
        splits[split] = rows
        prompts[split] = [make_prompt(x, name) for x in rows]
        labels[split] = np.asarray([label(x) for x in rows], dtype=np.int64)
        hidden[split] = extract_hidden(model, tokenizer, prompts[split], args.batch_size)

    probe_models, val_accs, selected = fit_probes(
        hidden["train"], labels["train"], hidden["val"], labels["val"]
    )
    clf = probe_models[selected]
    test_probe = clf.predict(hidden["test"][:, selected, :].astype(np.float32))
    test_probe_correct = test_probe == labels["test"]

    gen = generate(
        model, tokenizer, prompts["test"], splits["test"], args.batch_size, args.max_new_tokens
    )
    gen_correct = np.asarray([r["generation_correct"] for r in gen], dtype=bool)
    parseable = np.asarray([r["parseable"] for r in gen], dtype=bool)
    hard_mask = np.asarray([hard(x) for x in splits["test"]], dtype=bool)

    records = []
    for i, sample in enumerate(splits["test"]):
        records.append({
            "index": i,
            "a": sample["a"],
            "b": sample["b"],
            "digit": sample.get("digit"),
            "hard": bool(hard_mask[i]),
            "gold_position": answer_position(sample),
            "probe_prediction_position": "a" if int(test_probe[i]) == 1 else "b",
            "probe_correct": bool(test_probe_correct[i]),
            "generation_correct": bool(gen_correct[i]),
            "parseable": bool(parseable[i]),
            "critical": bool(test_probe_correct[i] and not gen_correct[i]),
            "completion": gen[i]["completion"],
        })

    def metrics(mask):
        n = int(mask.sum())
        if n == 0:
            return {}
        p = test_probe_correct[mask]
        g = gen_correct[mask]
        pa = parseable[mask]
        critical = p & ~g
        errors = ~g
        return {
            "n": n,
            "probe_accuracy": float(p.mean()),
            "generation_accuracy": float(g.mean()),
            "gap": float(p.mean() - g.mean()),
            "n_critical": int(critical.sum()),
            "critical_rate": float(critical.mean()),
            "error_coverage_by_probe_correct": float(critical.sum() / errors.sum()) if errors.sum() else None,
            "invalid_rate": float((~pa).mean()),
            "n11_probe_ok_gen_ok": int((p & g).sum()),
            "n10_probe_ok_gen_wrong": int((p & ~g).sum()),
            "n01_probe_wrong_gen_ok": int((~p & g).sum()),
            "n00_probe_wrong_gen_wrong": int((~p & ~g).sum()),
        }

    return {
        "dataset": name,
        "selected_layer_zero_based": int(selected),
        "selected_layer_one_based": int(selected + 1),
        "validation_probe_accuracy_by_layer": val_accs,
        "full_test": metrics(np.ones(len(splits["test"]), dtype=bool)),
        "hard_test": metrics(hard_mask),
        "records": records,
    }


def survival_gate(results):
    hard = [r["hard_test"] for r in results]
    total_n = sum(x["n"] for x in hard)
    pooled_probe_correct = sum(
        x["n11_probe_ok_gen_ok"] + x["n10_probe_ok_gen_wrong"] for x in hard
    )
    pooled_gen_correct = sum(
        x["n11_probe_ok_gen_ok"] + x["n01_probe_wrong_gen_ok"] for x in hard
    )
    pooled_critical = sum(x["n10_probe_ok_gen_wrong"] for x in hard)
    pooled_invalid = sum(round(x["invalid_rate"] * x["n"]) for x in hard)
    probe_acc = pooled_probe_correct / total_n
    gen_acc = pooled_gen_correct / total_n
    gap = probe_acc - gen_acc

    conditions = {
        "pooled_probe_accuracy_ge_0p80": probe_acc >= 0.80,
        "pooled_gap_ge_0p15": gap >= 0.15,
        "pooled_n_critical_ge_60": pooled_critical >= 60,
        "each_dataset_n_critical_ge_20": all(x["n10_probe_ok_gen_wrong"] >= 20 for x in hard),
        "positive_gap_both_datasets": all(x["gap"] > 0 for x in hard),
        "pooled_invalid_rate_lt_0p05": pooled_invalid / total_n < 0.05,
    }
    return {
        "verdict": "GO_CAUSAL_G1" if all(conditions.values()) else "KILL_OR_DOWNGRADE_ACCESS_PROJECT",
        "conditions": conditions,
        "pooled_hard": {
            "n": total_n,
            "probe_accuracy": probe_acc,
            "generation_accuracy": gen_acc,
            "gap": gap,
            "n_critical": pooled_critical,
            "invalid_rate": pooled_invalid / total_n,
        },
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()

    results = []
    for name in DATASETS:
        result = evaluate_dataset(name, args.data_root, model, tokenizer, args)
        results.append(result)
        # Keep item-level records separate from compact summary.
        with (args.out_dir / f"{name}_records.jsonl").open("w", encoding="utf-8") as f:
            for row in result.pop("records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    gate = survival_gate(results)
    payload = {
        "model": args.model,
        "prompt": "official balanced 5-shot",
        "hard_regime": "abs(log2(a/b)) < 0.1",
        "seed": args.seed,
        "smoke_limits": {
            "train": args.train_limit,
            "val": args.val_limit,
            "test": args.test_limit,
        },
        "datasets": results,
        "survival_gate": gate,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
