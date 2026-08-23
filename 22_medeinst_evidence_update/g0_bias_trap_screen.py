#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


FINAL_RE = re.compile(r"FINAL_DIAGNOSIS\s*:\s*(.+)", flags=re.IGNORECASE)


def normalize(text: str) -> str:
    text = (text or "").lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9+\-/ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_final_diagnosis(text: str, strict_marker: bool = True) -> str | None:
    matches = FINAL_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    if strict_marker:
        return None
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def resolve_diagnosis(candidate: str | None, labels: list[str]) -> str | None:
    ans = normalize(candidate or "")
    if not ans:
        return None
    exact = [label for label in labels if normalize(label) == ans]
    if len(exact) == 1:
        return exact[0]
    hits = [label for label in labels if normalize(label) and normalize(label) in ans]
    if hits:
        longest_len = max(len(normalize(label)) for label in hits)
        longest = [label for label in hits if len(normalize(label)) == longest_len]
        if len(longest) == 1:
            return longest[0]
    return None


def make_pairs(ds) -> list[tuple[dict, dict]]:
    grouped = defaultdict(list)
    for row in ds:
        grouped[row["case_id"]].append(dict(row))
    pairs = []
    for rows in grouped.values():
        by = defaultdict(list)
        for row in rows:
            by[row["case_type"]].append(row)
        if len(by["control"]) == 1 and len(by["trap"]) == 1:
            pairs.append((by["control"][0], by["trap"][0]))
    return pairs


def fixed_random_sample(pairs: list[tuple[dict, dict]], n: int, seed: int) -> list[tuple[dict, dict]]:
    rng = random.Random(seed)
    if n >= len(pairs):
        out = list(pairs)
        rng.shuffle(out)
        return out
    return rng.sample(pairs, n)


def prompt(tok, row: dict, mode: str) -> str:
    if mode == "cot":
        instruction = (
            "Analyze the clinical case step by step and determine the single most likely diagnosis. "
            "At the very end, on a new line, write exactly: FINAL_DIAGNOSIS: <diagnosis>."
        )
    elif mode == "direct":
        instruction = (
            "Determine the single most likely diagnosis from the clinical case. Do not explain. "
            "Return exactly one line: FINAL_DIAGNOSIS: <diagnosis>."
        )
    else:
        raise ValueError(mode)

    user = instruction + "\n\nClinical narrative:\n" + row.get("narrative", "")
    messages = [
        {"role": "system", "content": "Controlled diagnostic benchmark. Use only the case information provided."},
        {"role": "user", "content": user},
    ]
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(mode == "cot"),
            )
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] + "\n\n" + user + "\n"


@torch.inference_mode()
def generate(model, tok, text: str, max_new_tokens: int) -> str:
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    out = model.generate(
        input_ids=ids,
        attention_mask=mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="zhui711/MedEinst")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--mode", choices=["cot", "direct"], default="cot")
    ap.add_argument("--cot-max-new-tokens", type=int, default=1024)
    ap.add_argument("--direct-max-new-tokens", type=int, default=64)
    ap.add_argument("--outdir", default="artifacts/g0_behavior_cot")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "records.jsonl"
    record_path.unlink(missing_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    all_pairs = make_pairs(ds)
    pairs = fixed_random_sample(all_pairs, args.n_pairs, args.seed)
    labels = sorted({row["ground_truth"] for row in ds})

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    max_new_tokens = args.cot_max_new_tokens if args.mode == "cot" else args.direct_max_new_tokens
    recs = []
    for control, trap in tqdm(pairs, desc=f"MedEinst G0 {args.mode}"):
        control_text = generate(model, tok, prompt(tok, control, args.mode), max_new_tokens)
        trap_text = generate(model, tok, prompt(tok, trap, args.mode), max_new_tokens)
        control_final = extract_final_diagnosis(control_text, strict_marker=True)
        trap_final = extract_final_diagnosis(trap_text, strict_marker=True)
        control_pred = resolve_diagnosis(control_final, labels)
        trap_pred = resolve_diagnosis(trap_final, labels)
        control_gt = control["ground_truth"]
        trap_gt = trap["ground_truth"]
        row = {
            "case_id": control["case_id"],
            "mode": args.mode,
            "control_gt": control_gt,
            "trap_gt": trap_gt,
            "control_output": control_text,
            "trap_output": trap_text,
            "control_final": control_final,
            "trap_final": trap_final,
            "control_pred": control_pred,
            "trap_pred": trap_pred,
            "control_correct": control_pred == control_gt,
            "trap_correct": trap_pred == trap_gt,
            "bias_trap": bool(control_pred == control_gt and trap_gt != control_gt and trap_pred == control_gt),
            "invalid": control_pred is None or trap_pred is None,
        }
        recs.append(row)
        with record_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(recs)
    control_correct = [row for row in recs if row["control_correct"]]
    bias_traps = [row for row in recs if row["bias_trap"]]
    control_acc = len(control_correct) / max(1, n)
    trap_acc = sum(row["trap_correct"] for row in recs) / max(1, n)
    invalid_rate = sum(row["invalid"] for row in recs) / max(1, n)
    btr = len(bias_traps) / max(1, len(control_correct))
    btr_lo, btr_hi = wilson_interval(len(bias_traps), len(control_correct))
    transition_count = len({(row["control_gt"], row["trap_gt"]) for row in bias_traps})

    if args.mode == "cot":
        gate = {
            "control_accuracy_ge_0.35": control_acc >= 0.35,
            "control_correct_count_ge_50": len(control_correct) >= 50,
            "bias_trap_count_ge_20": len(bias_traps) >= 20,
            "bias_trap_rate_ge_0.30": btr >= 0.30,
            "bias_trap_wilson_lower_ge_0.20": btr_lo >= 0.20,
            "bias_trap_transition_count_ge_8": transition_count >= 8,
            "invalid_rate_le_0.10": invalid_rate <= 0.10,
        }
        go_verdict = "SEED_PHENOMENON_REPRODUCED"
        stop_verdict = "SEED_PHENOMENON_NOT_REPRODUCED"
    else:
        gate = {
            "control_accuracy_ge_0.30": control_acc >= 0.30,
            "control_correct_count_ge_40": len(control_correct) >= 40,
            "bias_trap_count_ge_16": len(bias_traps) >= 16,
            "bias_trap_rate_ge_0.20": btr >= 0.20,
            "bias_trap_wilson_lower_ge_0.10": btr_lo >= 0.10,
            "bias_trap_transition_count_ge_6": transition_count >= 6,
            "invalid_rate_le_0.10": invalid_rate <= 0.10,
        }
        go_verdict = "DIRECT_MODE_MECHANISM_OBJECT_READY"
        stop_verdict = "DIRECT_MODE_MECHANISM_OBJECT_TOO_WEAK"

    summary = {
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "sampled_pairs": n,
        "max_new_tokens": max_new_tokens,
        "sample_case_ids": [row["case_id"] for row in recs],
        "control_accuracy": control_acc,
        "trap_accuracy": trap_acc,
        "control_correct_count": len(control_correct),
        "bias_trap_count": len(bias_traps),
        "bias_trap_rate_among_control_correct": btr,
        "bias_trap_rate_wilson_95": [btr_lo, btr_hi],
        "bias_trap_transition_count": transition_count,
        "invalid_rate": invalid_rate,
        "paper_reference_for_qwen3_14b": {"baseline_accuracy": 0.4412, "bias_trap_rate": 0.5419},
        "gate": gate,
        "verdict": go_verdict if all(gate.values()) else stop_verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
