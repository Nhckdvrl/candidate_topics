#!/usr/bin/env python3
"""Frozen G1 causal-access test for Topic 20.

This implements G1-P0..P3 from G1_CAUSAL_ACCESS.md:

- fit the seed-0 ranking probe at the predeclared saturation layer L_sat=20
  (zero-based transformer block 19);
- evaluate the frozen probe + baseline generation on a fresh int-sci test seed;
- deduplicate exact displayed (a,b) pairs for inferential counts;
- on fresh hard examples that are originally probe-correct and generation-correct,
  minimally reflect the residual state across the ranking probe hyperplane;
- compare exact opposite-operand flips against eight equal-norm random directions
  orthogonal to the ranking direction.

No layer/token/strength/model/prompt search is performed. The conditional
notation intervention from G1-P4 is intentionally NOT implemented here; it is
allowed only after its fresh-seed descriptive prerequisite is confirmed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = "int_sci_compare"
L_SAT_BLOCK_ZERO_BASED = 19
L_SAT_ONE_BASED = 20
EXPECTED_SEED0_VAL_ACC = 0.990625
RANDOM_SEEDS = list(range(20260831, 20260839))
NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[×x*]\s*10\^?-?\d+)?")
SCI_RE = re.compile(r"[×x*]\s*10\^")

FEW_SHOT = [
    ("9.9 × 10^2", "100", 0),
    ("161230", "7.182 × 10^5", 1),
    ("713", "4.78 × 10^2", 0),
    ("1.354 × 10^6", "4906723", 1),
    ("20834", "6.5 × 10^3", 0),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed0-data-root", type=Path, required=True,
                   help="Root containing official seed-0 int_sci_compare train/val.")
    p.add_argument("--fresh-data-root", type=Path, required=True,
                   help="Root containing fresh-seed int_sci_compare test.")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out-dir", type=Path,
                   default=Path("20_numeracy_representation_access/artifacts/g1"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--train-limit", type=int, default=None, help="Smoke only.")
    p.add_argument("--val-limit", type=int, default=None, help="Smoke only.")
    p.add_argument("--test-limit", type=int, default=None, help="Smoke only.")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_value(s: str) -> float:
    return float(eval(str(s).replace("×", "*").replace("x", "*")
                      .replace("^", "**").replace(",", "")))


def is_sci(s: str) -> bool:
    return SCI_RE.search(str(s)) is not None


def make_prompt(sample: dict) -> str:
    demos = []
    for a, b, ans_id in FEW_SHOT:
        demos.append(f"Q: Which is larger, {a} or {b}? A: {(a, b)[ans_id]}")
    return "\n".join(demos) + f"\nQ: Which is larger, {sample['a']} or {sample['b']}? A:"


def label(sample: dict) -> int:
    return int(parse_value(sample["a"]) > parse_value(sample["b"]))


def gold_side(sample: dict) -> str:
    return "a" if label(sample) == 1 else "b"


def is_tie(sample: dict) -> bool:
    a, b = parse_value(sample["a"]), parse_value(sample["b"])
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)


def is_hard(sample: dict) -> bool:
    a, b = parse_value(sample["a"]), parse_value(sample["b"])
    return (not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)) and abs(math.log2(a / b)) < 0.1


def load_jsonl(path: Path, limit: int | None = None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_generated_number(text: str):
    m = NUMBER_RE.search(str(text))
    if not m:
        return None
    try:
        return parse_value(m.group(0))
    except Exception:
        return None


def same_num(x, y) -> bool:
    if x is None or y is None:
        return False
    return math.isclose(float(x), float(y), rel_tol=1e-10, abs_tol=max(1e-6, 1e-10 * abs(float(y))))


def classify_completion(sample: dict, completion: str):
    pred = parse_generated_number(completion)
    a, b = parse_value(sample["a"]), parse_value(sample["b"])
    if pred is None:
        return {"parseable": False, "choice": "invalid", "correct": False,
                "scientific_choice": None, "pred_value": None}

    a_match, b_match = same_num(pred, a), same_num(pred, b)
    if a_match and not b_match:
        side = "a"
    elif b_match and not a_match:
        side = "b"
    else:
        side = "neither_or_ambiguous"

    correct = side == gold_side(sample)
    sci_choice = is_sci(sample[side]) if side in {"a", "b"} else None
    return {"parseable": True, "choice": side, "correct": bool(correct),
            "scientific_choice": sci_choice, "pred_value": float(pred)}


def input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def decoder_layers(model):
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise RuntimeError("Could not locate decoder layer list for intervention.")


def extract_layer_hidden(model, tokenizer, prompts, block_idx: int, batch_size: int):
    """Return [N,D] output of block_idx at the final prompt token."""
    device = input_device(model)
    chunks = []
    hidden_index = block_idx + 1  # hidden_states[0] is embedding output
    for start in tqdm(range(0, len(prompts), batch_size), desc=f"hidden L{block_idx+1}"):
        ps = prompts[start:start + batch_size]
        enc = tokenizer(ps, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False, return_dict=True)
        h = out.hidden_states[hidden_index][:, -1, :].detach().to("cpu", dtype=torch.float32)
        chunks.append(h.numpy())
        del out, input_ids, attention_mask
    return np.concatenate(chunks, axis=0)


def restore_output(output: Any, new_hidden: torch.Tensor):
    if torch.is_tensor(output):
        return new_hidden
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    if isinstance(output, list):
        return [new_hidden] + list(output[1:])
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


class PrefillIntervention:
    """Modify only the final prompt token on the generation prefill pass."""

    def __init__(self, mode: str, w: np.ndarray, b: float, random_direction: np.ndarray | None = None):
        self.mode = mode
        w = np.asarray(w, dtype=np.float32)
        self.w = w
        self.b = float(b)
        self.w_norm2 = float(np.dot(w, w))
        self.w_norm = math.sqrt(self.w_norm2)
        if self.w_norm2 <= 0:
            raise ValueError("Degenerate ranking direction")
        self.random_direction = None if random_direction is None else np.asarray(random_direction, dtype=np.float32)
        self.n_prefill_calls = 0
        self.n_modified_rows = 0

    def hook(self, module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        if hidden.ndim != 3:
            return output
        # generate() with cache uses full prompt on prefill and sequence length 1 afterwards.
        if hidden.shape[1] <= 1:
            return output

        h = hidden[:, -1, :].float()
        w = torch.as_tensor(self.w, device=h.device, dtype=h.dtype)
        margin = h @ w + self.b
        coeff = -2.0 * margin / self.w_norm2
        rank_delta = coeff[:, None] * w[None, :]

        if self.mode == "rank_reflection":
            delta = rank_delta
        elif self.mode == "random_equal_norm":
            if self.random_direction is None:
                raise RuntimeError("random direction missing")
            r = torch.as_tensor(self.random_direction, device=h.device, dtype=h.dtype)
            # r is unit norm and orthogonal to w by construction.
            delta_norm = rank_delta.norm(dim=-1)
            delta = delta_norm[:, None] * r[None, :]
        else:
            raise ValueError(self.mode)

        changed = hidden.clone()
        changed[:, -1, :] = (h + delta).to(hidden.dtype)
        self.n_prefill_calls += 1
        self.n_modified_rows += int(hidden.shape[0])
        return restore_output(output, changed)


def run_generation(model, tokenizer, prompts, samples, batch_size: int,
                   max_new_tokens: int, intervention: PrefillIntervention | None = None):
    device = input_device(model)
    rows = []
    handle = None
    if intervention is not None:
        layers = decoder_layers(model)
        handle = layers[L_SAT_BLOCK_ZERO_BASED].register_forward_hook(intervention.hook)
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc=(intervention.mode if intervention else "baseline")):
            ps = prompts[start:start + batch_size]
            ss = samples[start:start + batch_size]
            enc = tokenizer(ps, return_tensors="pt", padding=True, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.inference_mode():
                seq = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completions = tokenizer.batch_decode(seq[:, input_ids.shape[1]:], skip_special_tokens=True)
            for sample, completion in zip(ss, completions):
                c = classify_completion(sample, completion)
                rows.append({**c, "completion": completion})
            del seq, input_ids, attention_mask
    finally:
        if handle is not None:
            handle.remove()
    return rows


def orthogonal_random_direction(dim: int, w: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(dim).astype(np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    r = r - np.dot(r, w64) / np.dot(w64, w64) * w64
    n = np.linalg.norm(r)
    if n < 1e-12:
        raise RuntimeError("Random direction collapsed during orthogonalization")
    return (r / n).astype(np.float32)


def unique_test_rows(rows):
    kept, raw_to_kept = [], []
    seen = set()
    n_ties = 0
    n_dups = 0
    for i, row in enumerate(rows):
        if is_tie(row):
            n_ties += 1
            continue
        key = (str(row["a"]), str(row["b"]))
        if key in seen:
            n_dups += 1
            continue
        seen.add(key)
        copied = dict(row)
        copied["raw_index"] = i
        kept.append(copied)
        raw_to_kept.append(i)
    return kept, {"raw_n": len(rows), "unique_n": len(kept),
                  "excluded_ties": n_ties, "excluded_exact_displayed_duplicates": n_dups}


def baseline_summary(samples, probe_pred, generated):
    hard = np.asarray([is_hard(x) for x in samples], dtype=bool)
    y = np.asarray([label(x) for x in samples], dtype=int)
    probe_ok = probe_pred == y
    gen_ok = np.asarray([x["correct"] for x in generated], dtype=bool)
    invalid = np.asarray([not x["parseable"] for x in generated], dtype=bool)
    critical = probe_ok & ~gen_ok

    h = hard
    hard_errors = [generated[i] for i in range(len(samples)) if hard[i] and not gen_ok[i]]
    operand_errors = [x for x in hard_errors if x["choice"] in {"a", "b"}]
    sci_choices = sum(bool(x["scientific_choice"]) for x in operand_errors)

    out = {
        "n": len(samples),
        "n_hard": int(h.sum()),
        "probe_accuracy_full": float(probe_ok.mean()),
        "probe_accuracy_hard": float(probe_ok[h].mean()) if h.any() else None,
        "generation_accuracy_full": float(gen_ok.mean()),
        "generation_accuracy_hard": float(gen_ok[h].mean()) if h.any() else None,
        "n_hard_critical": int((h & critical).sum()),
        "hard_critical_rate": float(critical[h].mean()) if h.any() else None,
        "hard_invalid_rate": float(invalid[h].mean()) if h.any() else None,
        "n_hard_generation_errors": len(hard_errors),
        "n_hard_errors_exact_operand": len(operand_errors),
        "n_hard_errors_choose_scientific_operand": sci_choices,
        "hard_error_scientific_choice_rate": (sci_choices / len(operand_errors) if operand_errors else None),
    }
    conditions = {
        "n_hard_ge_100": out["n_hard"] >= 100,
        "hard_probe_ge_0p90": (out["probe_accuracy_hard"] or 0) >= 0.90,
        "n_hard_critical_ge_25": out["n_hard_critical"] >= 25,
        "hard_critical_rate_ge_0p20": (out["hard_critical_rate"] or 0) >= 0.20,
        "hard_invalid_lt_0p05": (out["hard_invalid_rate"] if out["hard_invalid_rate"] is not None else 1) < 0.05,
    }
    out["fresh_object_gate"] = {"pass": all(conditions.values()), "conditions": conditions}
    out["notation_followup_eligible"] = (
        out["hard_error_scientific_choice_rate"] is not None and
        out["hard_error_scientific_choice_rate"] >= 0.80
    )
    return out, hard, probe_ok, gen_ok


def bootstrap_delta(rank_flip: np.ndarray, null_flip_matrix: np.ndarray, n_boot: int, seed: int):
    """Bootstrap examples; null seeds stay nested within each example."""
    per_example_null = null_flip_matrix.mean(axis=1)
    diff = rank_flip.astype(float) - per_example_null
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(diff)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    return float(diff.mean()), [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


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

    # Seed-0 probe fitting at the already-frozen L_sat.
    train = load_jsonl(args.seed0_data_root / DATASET / "train.jsonl", args.train_limit)
    val = load_jsonl(args.seed0_data_root / DATASET / "val.jsonl", args.val_limit)
    train_prompts = [make_prompt(x) for x in train]
    val_prompts = [make_prompt(x) for x in val]
    x_train = extract_layer_hidden(model, tokenizer, train_prompts, L_SAT_BLOCK_ZERO_BASED, args.batch_size)
    x_val = extract_layer_hidden(model, tokenizer, val_prompts, L_SAT_BLOCK_ZERO_BASED, args.batch_size)
    y_train = np.asarray([label(x) for x in train], dtype=int)
    y_val = np.asarray([label(x) for x in val], dtype=int)
    probe = LogisticRegression(max_iter=10000, random_state=0)
    probe.fit(x_train, y_train)
    val_pred = probe.predict(x_val)
    val_acc = float(accuracy_score(y_val, val_pred))
    w = probe.coef_[0].astype(np.float32)
    b = float(probe.intercept_[0])
    np.savez(args.out_dir / "rank_probe_lsat.npz", w=w, b=b,
             block_zero_based=L_SAT_BLOCK_ZERO_BASED,
             layer_one_based=L_SAT_ONE_BASED,
             seed0_val_accuracy=val_acc)

    # Fresh test: raw integrity + unique inferential set.
    fresh_raw = load_jsonl(args.fresh_data_root / DATASET / "test.jsonl", args.test_limit)
    fresh, fresh_audit = unique_test_rows(fresh_raw)
    fresh_prompts = [make_prompt(x) for x in fresh]
    x_fresh = extract_layer_hidden(model, tokenizer, fresh_prompts, L_SAT_BLOCK_ZERO_BASED, args.batch_size)
    fresh_probe_pred = probe.predict(x_fresh)
    baseline = run_generation(model, tokenizer, fresh_prompts, fresh, args.batch_size, args.max_new_tokens)
    base_summary, hard_mask, probe_ok, gen_ok = baseline_summary(fresh, fresh_probe_pred, baseline)

    with (args.out_dir / "fresh_baseline_records.jsonl").open("w", encoding="utf-8") as f:
        for i, sample in enumerate(fresh):
            row = {
                "unique_index": i,
                "raw_index": sample["raw_index"],
                "a": sample["a"], "b": sample["b"], "digit": sample.get("digit"),
                "hard": bool(hard_mask[i]),
                "gold_side": gold_side(sample),
                "probe_side": "a" if int(fresh_probe_pred[i]) == 1 else "b",
                "probe_correct": bool(probe_ok[i]),
                **baseline[i],
                "critical": bool(probe_ok[i] and not gen_ok[i]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.out_dir / "fresh_data_audit.json").write_text(
        json.dumps(fresh_audit, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "fresh_baseline_summary.json").write_text(
        json.dumps({"seed0_probe_val_accuracy": val_acc, **base_summary}, indent=2) + "\n",
        encoding="utf-8")

    if smoke:
        payload = {
            "verdict": "SMOKE_ONLY_NO_G1_DECISION",
            "seed0_probe_val_accuracy": val_acc,
            "fresh_data_audit": fresh_audit,
            "fresh_baseline": base_summary,
        }
        (args.out_dir / "rank_causal_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    if abs(val_acc - EXPECTED_SEED0_VAL_ACC) > 1e-12:
        raise RuntimeError(
            f"L_sat seed-0 validation accuracy drifted: got {val_acc}, expected {EXPECTED_SEED0_VAL_ACC}. "
            "Stop and audit environment/data before G1."
        )

    if not base_summary["fresh_object_gate"]["pass"]:
        payload = {
            "verdict": "STOP_G1_NONREPLICATION",
            "seed0_probe_val_accuracy": val_acc,
            "fresh_data_audit": fresh_audit,
            "fresh_baseline": base_summary,
        }
        (args.out_dir / "rank_causal_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    # Primary causal population: unique hard, probe-correct, generation-correct.
    population_idx = np.flatnonzero(hard_mask & probe_ok & gen_ok)
    pop_samples = [fresh[i] for i in population_idx]
    pop_prompts = [fresh_prompts[i] for i in population_idx]
    if not pop_samples:
        raise RuntimeError("No eligible hard correct population for G1 intervention")

    rank_int = PrefillIntervention("rank_reflection", w, b)
    rank_after = run_generation(model, tokenizer, pop_prompts, pop_samples,
                                args.batch_size, args.max_new_tokens, rank_int)

    rank_flip = np.zeros(len(pop_samples), dtype=bool)
    rank_changed = np.zeros(len(pop_samples), dtype=bool)
    rank_changed_exact_operand = np.zeros(len(pop_samples), dtype=bool)
    for i, (sample, after) in enumerate(zip(pop_samples, rank_after)):
        g = gold_side(sample)
        opposite = "b" if g == "a" else "a"
        rank_flip[i] = after["choice"] == opposite
        rank_changed[i] = after["choice"] != g
        rank_changed_exact_operand[i] = rank_changed[i] and after["choice"] in {"a", "b"}

    null_flip_matrix = np.zeros((len(pop_samples), len(RANDOM_SEEDS)), dtype=bool)
    null_outputs = []
    for j, seed in enumerate(RANDOM_SEEDS):
        r = orthogonal_random_direction(len(w), w, seed)
        null_int = PrefillIntervention("random_equal_norm", w, b, random_direction=r)
        after = run_generation(model, tokenizer, pop_prompts, pop_samples,
                               args.batch_size, args.max_new_tokens, null_int)
        per_seed_rows = []
        for i, (sample, row) in enumerate(zip(pop_samples, after)):
            g = gold_side(sample)
            opposite = "b" if g == "a" else "a"
            flip = row["choice"] == opposite
            null_flip_matrix[i, j] = flip
            per_seed_rows.append({
                "population_index": i,
                "source_unique_index": int(population_idx[i]),
                "random_seed": seed,
                "gold_side": g,
                "opposite_flip": bool(flip),
                **row,
            })
        null_outputs.extend(per_seed_rows)

    delta_f, ci = bootstrap_delta(rank_flip, null_flip_matrix, args.bootstrap, seed=20260840)
    f_rank = float(rank_flip.mean())
    null_rates = [float(null_flip_matrix[:, j].mean()) for j in range(null_flip_matrix.shape[1])]
    f_null = float(null_flip_matrix.mean())
    changed_n = int(rank_changed.sum())
    exact_among_changed = (
        float(rank_changed_exact_operand.sum() / changed_n) if changed_n else None
    )

    # Reflection analytically maps m -> -m except exact zero margins.
    fresh_pop_h = x_fresh[population_idx]
    margins = fresh_pop_h @ w + b
    probe_flip_fraction = float((np.sign(margins) == -np.sign(-margins)).mean())

    if (
        probe_flip_fraction >= 0.99 and
        delta_f >= 0.20 and
        ci[0] > 0 and
        exact_among_changed is not None and exact_among_changed >= 0.80
    ):
        verdict = "RANK_DIRECTION_CAUSAL"
    elif delta_f <= 0.05 and ci[1] <= 0.10:
        verdict = "READABLE_BUT_NOT_CAUSALLY_USED_AT_LSAT"
    else:
        verdict = "INCONCLUSIVE_DO_NOT_TUNE"

    with (args.out_dir / "rank_reflection_records.jsonl").open("w", encoding="utf-8") as f:
        for i, (sample, row) in enumerate(zip(pop_samples, rank_after)):
            out = {
                "population_index": i,
                "source_unique_index": int(population_idx[i]),
                "raw_index": sample["raw_index"],
                "a": sample["a"], "b": sample["b"], "digit": sample.get("digit"),
                "gold_side": gold_side(sample),
                "baseline_completion": baseline[population_idx[i]]["completion"],
                "rank_margin_before": float(margins[i]),
                "rank_margin_after_analytic": float(-margins[i]),
                "opposite_flip": bool(rank_flip[i]),
                **row,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    with (args.out_dir / "random_null_records.jsonl").open("w", encoding="utf-8") as f:
        for row in null_outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "verdict": verdict,
        "fresh_seed": 20260824,
        "model": args.model,
        "L_sat": {"block_zero_based": L_SAT_BLOCK_ZERO_BASED,
                   "layer_one_based": L_SAT_ONE_BASED,
                   "seed0_validation_probe_accuracy": val_acc},
        "fresh_data_audit": fresh_audit,
        "fresh_baseline": base_summary,
        "population_n": len(pop_samples),
        "probe_flip_fraction_analytic": probe_flip_fraction,
        "F_rank": f_rank,
        "F_null_by_seed": dict(zip(map(str, RANDOM_SEEDS), null_rates)),
        "F_null_mean": f_null,
        "DeltaF": delta_f,
        "DeltaF_bootstrap_95ci": ci,
        "rank_changed_n": changed_n,
        "exact_operand_fraction_among_changed": exact_among_changed,
        "notation_followup_eligible": base_summary["notation_followup_eligible"],
    }
    (args.out_dir / "rank_causal_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
