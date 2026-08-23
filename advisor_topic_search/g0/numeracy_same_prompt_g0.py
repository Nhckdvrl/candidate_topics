#!/usr/bin/env python3
"""Frozen same-prompt G0 for the Numeracy representation-vs-access candidate.

Scientific prerequisite
-----------------------
On the exact same five-shot prompt and Qwen3-8B, does the last-input-token
residual stream contain linearly decodable correct ranking information while
the model's greedy generation is wrong?

This script is deliberately seed-exact and narrow:
- primary model: Qwen/Qwen3-8B
- primary dataset: official seed-0 int_sci_compare only
- exact 5-shot int-sci demonstrations from the EACL-2026 paper / official code
- hidden-state position: last input token only
- one logistic classifier per layer
- layer chosen by validation accuracy, earliest-layer tie break
- test evaluated once after layer selection
- hard regime fixed by seed paper: |log2(a/b)| < 0.1

Why int-sci only?
-----------------
The paper states that int-sci is used for the headline Table 1 and the explicit
k=1..5 few-shot experiment. dec-sci is reserved for post-G0 confirmation rather
than being added as a second prerequisite gamble.

This script does NOT perform activation steering or patching. If this
prerequisite does not pass, do not start mechanism work.
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

PRIMARY_DATASET = "int_sci_compare"
NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[×x*]\s*10\^?-?\d+)?")
FEW_SHOT = [
    ("9.9 × 10^2", "100", 0),
    ("161230", "7.182 × 10^5", 1),
    ("713", "4.78 × 10^2", 0),
    ("1.354 × 10^6", "4906723", 1),
    ("20834", "6.5 × 10^3", 0),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out-dir", type=Path, default=Path("numeracy_same_prompt_qwen3_8b"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--train-limit", type=int, default=None, help="Smoke only; unset for frozen G0.")
    p.add_argument("--val-limit", type=int, default=None, help="Smoke only; unset for frozen G0.")
    p.add_argument("--test-limit", type=int, default=None, help="Smoke only; unset for frozen G0.")
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


def make_prompt(sample: dict) -> str:
    demos = []
    for a, b, ans_id in FEW_SHOT:
        ans = (a, b)[ans_id]
        demos.append(f"Q: Which is larger, {a} or {b}? A: {ans}")
    return "\n".join(demos) + f"\nQ: Which is larger, {sample['a']} or {sample['b']}? A:"


def label(sample: dict) -> int:
    return int(parse_value(sample["a"]) > parse_value(sample["b"]))


def is_hard(sample: dict) -> bool:
    a, b = parse_value(sample["a"]), parse_value(sample["b"])
    return abs(math.log2(a / b)) < 0.1


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
    ok = abs(pred - target) <= max(1e-3, 1e-8 * abs(target))
    return bool(ok), True


def input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def extract_hidden(model, tokenizer, prompts, batch_size: int):
    """Return [N, L, D] float16 CPU array for final prompt token.

    Padding is LEFT, so the final non-padding token is always position -1.
    Direct layer[:, -1, :] also avoids cross-device fancy indexing when
    device_map=auto distributes layers across GPUs.
    """
    device = input_device(model)
    chunks = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="hidden"):
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
        layers = out.hidden_states[1:]  # skip embedding output
        per_layer = [x[:, -1, :].detach().to("cpu", dtype=torch.float16) for x in layers]
        chunks.append(torch.stack(per_layer, dim=1).numpy())
        del out, layers, input_ids, attention_mask
    return np.concatenate(chunks, axis=0)


def run_generation(model, tokenizer, prompts, samples, batch_size: int, max_new_tokens: int):
    device = input_device(model)
    rows = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="generation"):
        ps = prompts[start : start + batch_size]
        ss = samples[start : start + batch_size]
        enc = tokenizer(ps, return_tensors="pt", padding=True, add_special_tokens=True)
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
        for sample, completion in zip(ss, completions):
            correct, parseable = generation_correct(sample, completion)
            rows.append({"correct": correct, "parseable": parseable, "completion": completion})
        del seq, input_ids, attention_mask
    return rows


def fit_layerwise_probes(train_x, train_y, val_x, val_y):
    models, val_accs = [], []
    for layer in tqdm(range(train_x.shape[1]), desc="probes"):
        clf = LogisticRegression(max_iter=10000, random_state=0)
        clf.fit(train_x[:, layer, :].astype(np.float32), train_y)
        pred = clf.predict(val_x[:, layer, :].astype(np.float32))
        models.append(clf)
        val_accs.append(float(accuracy_score(val_y, pred)))
    best = max(val_accs)
    selected = next(i for i, x in enumerate(val_accs) if abs(x - best) < 1e-12)
    return models, val_accs, selected


def summarize(mask, probe_ok, gen_ok, parseable):
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    p = probe_ok[mask]
    g = gen_ok[mask]
    q = parseable[mask]
    critical = p & ~g
    return {
        "n": n,
        "probe_accuracy": float(p.mean()),
        "generation_accuracy": float(g.mean()),
        "gap": float(p.mean() - g.mean()),
        "n_critical": int(critical.sum()),
        "critical_rate": float(critical.mean()),
        "n_invalid": int((~q).sum()),
        "invalid_rate": float((~q).mean()),
        "n11_probe_ok_gen_ok": int((p & g).sum()),
        "n10_probe_ok_gen_wrong": int((p & ~g).sum()),
        "n01_probe_wrong_gen_ok": int((~p & g).sum()),
        "n00_probe_wrong_gen_wrong": int((~p & ~g).sum()),
        "error_coverage_by_probe_correct": float(critical.sum() / (~g).sum()) if (~g).sum() else None,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    smoke = any(x is not None for x in (args.train_limit, args.val_limit, args.test_limit))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=dtype)
    model.eval()

    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    data, prompts, labels, hidden = {}, {}, {}, {}
    for split in ("train", "val", "test"):
        path = args.data_root / PRIMARY_DATASET / f"{split}.jsonl"
        data[split] = load_jsonl(path, limits[split])
        prompts[split] = [make_prompt(x) for x in data[split]]
        labels[split] = np.asarray([label(x) for x in data[split]], dtype=np.int64)
        hidden[split] = extract_hidden(model, tokenizer, prompts[split], args.batch_size)

    probes, val_accs, selected = fit_layerwise_probes(
        hidden["train"], labels["train"], hidden["val"], labels["val"]
    )
    test_pred = probes[selected].predict(hidden["test"][:, selected, :].astype(np.float32))
    probe_ok = test_pred == labels["test"]

    generated = run_generation(
        model, tokenizer, prompts["test"], data["test"], args.batch_size, args.max_new_tokens
    )
    gen_ok = np.asarray([x["correct"] for x in generated], dtype=bool)
    parseable = np.asarray([x["parseable"] for x in generated], dtype=bool)
    hard_mask = np.asarray([is_hard(x) for x in data["test"]], dtype=bool)
    full_mask = np.ones(len(data["test"]), dtype=bool)

    full = summarize(full_mask, probe_ok, gen_ok, parseable)
    hard = summarize(hard_mask, probe_ok, gen_ok, parseable)

    conditions = {
        "full_probe_accuracy_ge_0p90": full.get("probe_accuracy", 0) >= 0.90,
        "hard_probe_accuracy_ge_0p80": hard.get("probe_accuracy", 0) >= 0.80,
        "hard_gap_ge_0p15": hard.get("gap", -1) >= 0.15,
        "hard_n_critical_ge_30": hard.get("n_critical", 0) >= 30,
        "hard_gap_positive": hard.get("gap", -1) > 0,
        "hard_invalid_rate_lt_0p05": hard.get("invalid_rate", 1) < 0.05,
    }
    if smoke:
        verdict = "SMOKE_ONLY_NO_PROJECT_DECISION"
    else:
        verdict = "GO_CAUSAL_G1" if all(conditions.values()) else "KILL_OR_DOWNGRADE_ACCESS_PROJECT"

    with (args.out_dir / "test_records.jsonl").open("w", encoding="utf-8") as f:
        for i, sample in enumerate(data["test"]):
            row = {
                "index": i,
                "a": sample["a"],
                "b": sample["b"],
                "digit": sample.get("digit"),
                "hard": bool(hard_mask[i]),
                "gold_position": "a" if labels["test"][i] == 1 else "b",
                "probe_position": "a" if test_pred[i] == 1 else "b",
                "probe_correct": bool(probe_ok[i]),
                "generation_correct": bool(gen_ok[i]),
                "parseable": bool(parseable[i]),
                "critical": bool(probe_ok[i] and not gen_ok[i]),
                "completion": generated[i]["completion"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "model": args.model,
        "dataset": PRIMARY_DATASET,
        "prompt": "official int-sci balanced 5-shot",
        "hard_regime": "abs(log2(a/b)) < 0.1",
        "selected_layer_zero_based": int(selected),
        "selected_layer_one_based": int(selected + 1),
        "validation_probe_accuracy_by_layer": val_accs,
        "full_test": full,
        "hard_test": hard,
        "smoke_limits": {
            "train": args.train_limit,
            "val": args.val_limit,
            "test": args.test_limit,
        },
        "survival_gate": {"verdict": verdict, "conditions": conditions},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
