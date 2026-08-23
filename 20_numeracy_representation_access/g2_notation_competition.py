#!/usr/bin/env python3
"""Frozen G2 notation-competition test for Topic 20.

Scientific target
-----------------
After seed-0 discovery and seed-20260824 independent confirmation that hard
mixed-notation errors overwhelmingly choose the scientific-notation operand,
test on untouched seed 20260825 whether a linearly identified notation-side
coordinate causally competes with the already-correct ranking signal.

Protocol source: G2_NOTATION_COMPETITION.md

No layer/token/strength/model/prompt/seed search is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reuse the already-audited Topic-20 utilities.  This script lives beside the
# module, so direct execution adds this directory to sys.path.
import g1_rank_reflection as g1


FRESH_SEED = 20260825
RANDOM_SEEDS = list(range(20260901, 20260909))
EXPECTED_SEED0_TRAIN_SHA256 = "8a995020ecd21dc23f3a3ac97880652c78c85573fa95b53305a1f89004092914"
EXPECTED_SEED0_VAL_SHA256 = "73f0a6703283d186243b4f4db4238712e0e6b523757693553e3b33b202d33d2e"
EXPECTED_RANK_VAL_ACC = 0.990625
COS_TOL = 1e-5
LOGIT_TOL = 5e-4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed0-data-root", type=Path, required=True)
    p.add_argument("--fresh-data-root", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out-dir", type=Path,
                   default=Path("20_numeracy_representation_access/artifacts/g2"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=FRESH_SEED)
    p.add_argument("--train-limit", type=int, default=None, help="Smoke only.")
    p.add_argument("--val-limit", type=int, default=None, help="Smoke only.")
    p.add_argument("--test-limit", type=int, default=None, help="Smoke only.")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def notation_side(row: dict) -> int:
    """1 iff A is scientific notation, else 0 iff B is scientific notation."""
    a_sci = g1.is_sci(row["a"])
    b_sci = g1.is_sci(row["b"])
    if a_sci == b_sci:
        raise ValueError(f"Expected exactly one scientific operand: {row}")
    return int(a_sci)


def projection_remove(v: np.ndarray, against: np.ndarray) -> np.ndarray:
    v64 = np.asarray(v, dtype=np.float64)
    a64 = np.asarray(against, dtype=np.float64)
    return v64 - np.dot(v64, a64) / np.dot(a64, a64) * a64


def fit_coordinates(x_train, y_rank_train, y_not_train,
                    x_val, y_rank_val, y_not_val):
    rank = LogisticRegression(max_iter=10000, random_state=0)
    rank.fit(x_train, y_rank_train)
    rank_val_acc = float(accuracy_score(y_rank_val, rank.predict(x_val)))
    w_rank = rank.coef_[0].astype(np.float64)
    b_rank = float(rank.intercept_[0])

    notation = LogisticRegression(max_iter=10000, random_state=0)
    notation.fit(x_train, y_not_train)
    w_not = notation.coef_[0].astype(np.float64)

    w_not_orth = projection_remove(w_not, w_rank)
    n = float(np.linalg.norm(w_not_orth))
    if n < 1e-12:
        raise RuntimeError("Notation direction collapses after ranking orthogonalization")
    u_not = w_not_orth / n

    cosine = float(np.dot(u_not, w_rank) / np.linalg.norm(w_rank))

    # Frozen 1D threshold: fit only on seed-0 TRAIN projections, then evaluate
    # validation.  The decision boundary of the 1D logistic model is tau.
    z_train = (x_train @ u_not).reshape(-1, 1)
    z_val = (x_val @ u_not).reshape(-1, 1)
    scalar = LogisticRegression(max_iter=10000, random_state=0)
    scalar.fit(z_train, y_not_train)
    coef = float(scalar.coef_[0, 0])
    intercept = float(scalar.intercept_[0])
    if abs(coef) < 1e-12:
        raise RuntimeError("Degenerate 1D notation threshold model")
    tau = -intercept / coef
    not_val_acc = float(accuracy_score(y_not_val, scalar.predict(z_val)))

    return {
        "rank_model": rank,
        "w_rank": w_rank.astype(np.float32),
        "b_rank": b_rank,
        "rank_val_acc": rank_val_acc,
        "u_not": u_not.astype(np.float32),
        "tau_not": float(tau),
        "notation_val_acc": not_val_acc,
        "notation_rank_cosine": cosine,
        "raw_notation_probe_val_acc": float(accuracy_score(y_not_val, notation.predict(x_val))),
    }


class NotationNeutralization:
    def __init__(self, u_not: np.ndarray, tau: float, w_rank: np.ndarray, b_rank: float):
        self.mode = "notation_neutralization"
        self.u = np.asarray(u_not, dtype=np.float32)
        self.tau = float(tau)
        self.w_rank = np.asarray(w_rank, dtype=np.float32)
        self.b_rank = float(b_rank)
        self.n_prefill_calls = 0
        self.n_modified_rows = 0
        self.max_rank_logit_change = 0.0
        self.max_notation_residual = 0.0

    def hook(self, module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        if hidden.ndim != 3 or hidden.shape[1] <= 1:
            return output
        h = hidden[:, -1, :].float()
        u = torch.as_tensor(self.u, device=h.device, dtype=h.dtype)
        w = torch.as_tensor(self.w_rank, device=h.device, dtype=h.dtype)
        z = h @ u
        delta = (self.tau - z)[:, None] * u[None, :]
        before_rank = h @ w + self.b_rank
        h2 = h + delta
        after_rank = h2 @ w + self.b_rank
        after_z = h2 @ u

        self.max_rank_logit_change = max(
            self.max_rank_logit_change,
            float((after_rank - before_rank).abs().max().detach().cpu())
        )
        self.max_notation_residual = max(
            self.max_notation_residual,
            float((after_z - self.tau).abs().max().detach().cpu())
        )
        changed = hidden.clone()
        changed[:, -1, :] = h2.to(hidden.dtype)
        self.n_prefill_calls += 1
        self.n_modified_rows += int(hidden.shape[0])
        return g1.restore_output(output, changed)


class RandomMatchedIntervention:
    def __init__(self, u_not: np.ndarray, tau: float, direction: np.ndarray,
                 w_rank: np.ndarray, b_rank: float):
        self.mode = "random_matched"
        self.u = np.asarray(u_not, dtype=np.float32)
        self.tau = float(tau)
        self.r = np.asarray(direction, dtype=np.float32)
        self.w_rank = np.asarray(w_rank, dtype=np.float32)
        self.b_rank = float(b_rank)
        self.n_prefill_calls = 0
        self.n_modified_rows = 0
        self.max_rank_logit_change = 0.0

    def hook(self, module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        if hidden.ndim != 3 or hidden.shape[1] <= 1:
            return output
        h = hidden[:, -1, :].float()
        u = torch.as_tensor(self.u, device=h.device, dtype=h.dtype)
        r = torch.as_tensor(self.r, device=h.device, dtype=h.dtype)
        w = torch.as_tensor(self.w_rank, device=h.device, dtype=h.dtype)
        z = h @ u
        norm = (self.tau - z).abs()
        delta = norm[:, None] * r[None, :]
        before_rank = h @ w + self.b_rank
        h2 = h + delta
        after_rank = h2 @ w + self.b_rank
        self.max_rank_logit_change = max(
            self.max_rank_logit_change,
            float((after_rank - before_rank).abs().max().detach().cpu())
        )
        changed = hidden.clone()
        changed[:, -1, :] = h2.to(hidden.dtype)
        self.n_prefill_calls += 1
        self.n_modified_rows += int(hidden.shape[0])
        return g1.restore_output(output, changed)


def random_orthogonal_to_two(dim: int, a: np.ndarray, b: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(dim).astype(np.float64)
    # Modified Gram-Schmidt against ranking then notation direction.
    r = projection_remove(r, a)
    r = projection_remove(r, b)
    # Numerical cleanup against ranking again because b is only finite-precision orthogonal.
    r = projection_remove(r, a)
    n = float(np.linalg.norm(r))
    if n < 1e-12:
        raise RuntimeError("Random null collapsed")
    r /= n
    return r.astype(np.float32)


def count_expected_prefills(n_examples: int, batch_size: int) -> int:
    return math.ceil(n_examples / batch_size)


def bootstrap_delta(rescue: np.ndarray, null_matrix: np.ndarray, n_boot: int, seed: int):
    per_example_null = null_matrix.mean(axis=1)
    diff = rescue.astype(float) - per_example_null
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(diff)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    return float(diff.mean()), [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    args = parse_args()
    g1.set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    smoke = any(x is not None for x in (args.train_limit, args.val_limit, args.test_limit))

    train_path = args.seed0_data_root / g1.DATASET / "train.jsonl"
    val_path = args.seed0_data_root / g1.DATASET / "val.jsonl"
    if not smoke:
        got_train, got_val = sha256(train_path), sha256(val_path)
        if got_train != EXPECTED_SEED0_TRAIN_SHA256 or got_val != EXPECTED_SEED0_VAL_SHA256:
            raise RuntimeError(f"Seed-0 data checksum mismatch: train={got_train}, val={got_val}")

    tokenizer = g1.AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = g1.AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=dtype)
    model.eval()

    train = g1.load_jsonl(train_path, args.train_limit)
    val = g1.load_jsonl(val_path, args.val_limit)
    x_train = g1.extract_layer_hidden(model, tokenizer, [g1.make_prompt(x) for x in train],
                                      g1.L_SAT_BLOCK_ZERO_BASED, args.batch_size)
    x_val = g1.extract_layer_hidden(model, tokenizer, [g1.make_prompt(x) for x in val],
                                    g1.L_SAT_BLOCK_ZERO_BASED, args.batch_size)
    y_rank_train = np.asarray([g1.label(x) for x in train], dtype=int)
    y_rank_val = np.asarray([g1.label(x) for x in val], dtype=int)
    y_not_train = np.asarray([notation_side(x) for x in train], dtype=int)
    y_not_val = np.asarray([notation_side(x) for x in val], dtype=int)

    c = fit_coordinates(x_train, y_rank_train, y_not_train, x_val, y_rank_val, y_not_val)
    if not smoke and abs(c["rank_val_acc"] - EXPECTED_RANK_VAL_ACC) > 1e-12:
        raise RuntimeError(f"Frozen rank probe drift: {c['rank_val_acc']} != {EXPECTED_RANK_VAL_ACC}")

    rep_checks = {
        "rank_val_accuracy": c["rank_val_acc"],
        "notation_val_accuracy": c["notation_val_acc"],
        "raw_notation_probe_val_accuracy": c["raw_notation_probe_val_acc"],
        "notation_rank_cosine": c["notation_rank_cosine"],
        "notation_accuracy_ge_0p95": c["notation_val_acc"] >= 0.95,
        "abs_cosine_le_1e_5": abs(c["notation_rank_cosine"]) <= COS_TOL,
    }
    rep_checks["pass"] = rep_checks["notation_accuracy_ge_0p95"] and rep_checks["abs_cosine_le_1e_5"]
    (args.out_dir / "notation_representation_checks.json").write_text(
        json.dumps(rep_checks, indent=2) + "\n", encoding="utf-8")
    np.savez(args.out_dir / "rank_probe_lsat.npz", w=c["w_rank"], b=c["b_rank"])
    np.savez(args.out_dir / "notation_probe_lsat.npz", u_not=c["u_not"], tau_not=c["tau_not"])

    if not rep_checks["pass"] and not smoke:
        payload = {"verdict": "STOP_G2_NOTATION_REPRESENTATION_NOT_CLEAN", "representation_checks": rep_checks}
        (args.out_dir / "notation_causal_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    fresh_path = args.fresh_data_root / g1.DATASET / "test.jsonl"
    fresh_raw = g1.load_jsonl(fresh_path, args.test_limit)
    fresh, audit = g1.unique_test_rows(fresh_raw)
    audit["test_sha256"] = sha256(fresh_path)
    audit["fresh_seed"] = FRESH_SEED
    (args.out_dir / "fresh_data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    prompts = [g1.make_prompt(x) for x in fresh]
    x_fresh = g1.extract_layer_hidden(model, tokenizer, prompts,
                                      g1.L_SAT_BLOCK_ZERO_BASED, args.batch_size)
    rank_pred = c["rank_model"].predict(x_fresh)
    baseline = g1.run_generation(model, tokenizer, prompts, fresh,
                                 args.batch_size, args.max_new_tokens)

    hard = np.asarray([g1.is_hard(x) for x in fresh], dtype=bool)
    rank_ok = rank_pred == np.asarray([g1.label(x) for x in fresh], dtype=int)
    gen_ok = np.asarray([x["correct"] for x in baseline], dtype=bool)
    exact_operand_error = np.asarray([
        (not r["correct"]) and r["choice"] in {"a", "b"} for r in baseline
    ], dtype=bool)
    sci_error = np.asarray([
        bool(r["scientific_choice"]) if ((not r["correct"]) and r["choice"] in {"a", "b"}) else False
        for r in baseline
    ], dtype=bool)

    hard_operand_errors = hard & exact_operand_error
    hard_sci_errors = hard & sci_error
    n_hard = int(hard.sum())
    n_operand_errors = int(hard_operand_errors.sum())
    n_sci_errors = int(hard_sci_errors.sum())
    sci_rate = n_sci_errors / n_operand_errors if n_operand_errors else None

    object_gate = {
        "n_unique_hard": n_hard,
        "n_hard_exact_operand_errors": n_operand_errors,
        "n_hard_scientific_operand_errors": n_sci_errors,
        "scientific_operand_choice_rate": sci_rate,
        "n_unique_hard_ge_100": n_hard >= 100,
        "n_hard_exact_operand_errors_ge_30": n_operand_errors >= 30,
        "scientific_operand_choice_rate_ge_0p80": sci_rate is not None and sci_rate >= 0.80,
    }
    object_gate["pass"] = all([
        object_gate["n_unique_hard_ge_100"],
        object_gate["n_hard_exact_operand_errors_ge_30"],
        object_gate["scientific_operand_choice_rate_ge_0p80"],
    ])

    with (args.out_dir / "fresh_baseline_records.jsonl").open("w", encoding="utf-8") as f:
        for i, (sample, row) in enumerate(zip(fresh, baseline)):
            f.write(json.dumps({
                "unique_index": i,
                "raw_index": sample["raw_index"],
                "a": sample["a"], "b": sample["b"], "digit": sample.get("digit"),
                "hard": bool(hard[i]),
                "rank_probe_correct": bool(rank_ok[i]),
                "gold_side": g1.gold_side(sample),
                **row,
            }, ensure_ascii=False) + "\n")

    base_summary = {
        "fresh_seed": FRESH_SEED,
        "representation_checks": rep_checks,
        "object_gate": object_gate,
        "full_rank_probe_accuracy": float(rank_ok.mean()),
        "hard_rank_probe_accuracy": float(rank_ok[hard].mean()) if hard.any() else None,
        "full_generation_accuracy": float(gen_ok.mean()),
        "hard_generation_accuracy": float(gen_ok[hard].mean()) if hard.any() else None,
    }
    (args.out_dir / "fresh_baseline_summary.json").write_text(
        json.dumps(base_summary, indent=2) + "\n", encoding="utf-8")

    if smoke:
        payload = {"verdict": "SMOKE_ONLY_NO_G2_DECISION", **base_summary}
        (args.out_dir / "notation_causal_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    if not object_gate["pass"]:
        payload = {"verdict": "STOP_G2_NOTATION_NONREPLICATION", **base_summary}
        (args.out_dir / "notation_causal_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    primary_mask = hard & rank_ok & (~gen_ok) & exact_operand_error & sci_error
    primary_idx = np.flatnonzero(primary_mask)
    if len(primary_idx) < 25:
        payload = {
            "verdict": "STOP_G2_INSUFFICIENT_CAUSAL_SUPPORT",
            **base_summary,
            "primary_n": int(len(primary_idx)),
        }
        (args.out_dir / "notation_causal_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    pop_samples = [fresh[i] for i in primary_idx]
    pop_prompts = [prompts[i] for i in primary_idx]
    pop_h = x_fresh[primary_idx]
    rank_margin_before = pop_h @ c["w_rank"] + c["b_rank"]
    notation_before = pop_h @ c["u_not"]
    neutral_norm = np.abs(c["tau_not"] - notation_before)

    intervention = NotationNeutralization(c["u_not"], c["tau_not"], c["w_rank"], c["b_rank"])
    after = g1.run_generation(model, tokenizer, pop_prompts, pop_samples,
                              args.batch_size, args.max_new_tokens, intervention)
    expected_prefills = count_expected_prefills(len(pop_samples), args.batch_size)
    if intervention.n_prefill_calls != expected_prefills:
        raise RuntimeError(f"Neutralization hook call mismatch: {intervention.n_prefill_calls} != {expected_prefills}")

    rescue = np.asarray([r["correct"] for r in after], dtype=bool)
    changed = np.asarray([r["choice"] != baseline[primary_idx[i]]["choice"] for i, r in enumerate(after)], dtype=bool)
    valid_changed = np.asarray([changed[i] and r["choice"] in {"a", "b"} for i, r in enumerate(after)], dtype=bool)
    changed_to_correct = np.asarray([valid_changed[i] and r["correct"] for i, r in enumerate(after)], dtype=bool)
    invalid_or_neither = np.asarray([r["choice"] not in {"a", "b"} for r in after], dtype=bool)

    null_matrix = np.zeros((len(pop_samples), len(RANDOM_SEEDS)), dtype=bool)
    null_records = []
    null_rank_drift = []
    for j, seed in enumerate(RANDOM_SEEDS):
        r = random_orthogonal_to_two(len(c["w_rank"]), c["w_rank"], c["u_not"], seed)
        null_int = RandomMatchedIntervention(c["u_not"], c["tau_not"], r, c["w_rank"], c["b_rank"])
        out = g1.run_generation(model, tokenizer, pop_prompts, pop_samples,
                                args.batch_size, args.max_new_tokens, null_int)
        if null_int.n_prefill_calls != expected_prefills:
            raise RuntimeError(f"Null hook call mismatch seed={seed}")
        null_rank_drift.append(null_int.max_rank_logit_change)
        for i, row in enumerate(out):
            null_matrix[i, j] = bool(row["correct"])
            null_records.append({
                "population_index": i,
                "source_unique_index": int(primary_idx[i]),
                "random_seed": seed,
                "rescue": bool(row["correct"]),
                **row,
            })

    delta_r, ci = bootstrap_delta(rescue, null_matrix, args.bootstrap, seed=20260910)
    r_not = float(rescue.mean())
    null_rates = [float(null_matrix[:, j].mean()) for j in range(null_matrix.shape[1])]
    r_null = float(null_matrix.mean())
    invalid_rate = float(invalid_or_neither.mean())
    n_changed_valid = int(valid_changed.sum())
    correct_among_changed = float(changed_to_correct.sum() / n_changed_valid) if n_changed_valid else None

    manip = {
        "neutralization_prefill_calls": intervention.n_prefill_calls,
        "expected_prefill_calls": expected_prefills,
        "max_rank_logit_change_neutralization": intervention.max_rank_logit_change,
        "max_notation_threshold_residual": intervention.max_notation_residual,
        "max_rank_logit_change_random_null": max(null_rank_drift) if null_rank_drift else None,
        "notation_neutralized_ge_0p99": intervention.max_notation_residual <= LOGIT_TOL,
        "ranking_preserved_ge_0p99": intervention.max_rank_logit_change <= LOGIT_TOL,
    }

    if (
        manip["notation_neutralized_ge_0p99"] and
        manip["ranking_preserved_ge_0p99"] and
        delta_r >= 0.20 and ci[0] > 0 and
        invalid_rate < 0.10 and
        correct_among_changed is not None and correct_among_changed >= 0.80
    ):
        verdict = "NOTATION_COMPETITION_CAUSAL"
    elif (
        manip["notation_neutralized_ge_0p99"] and
        manip["ranking_preserved_ge_0p99"] and
        delta_r <= 0.05 and ci[1] <= 0.10
    ):
        verdict = "NOTATION_READABLE_BUT_NOT_CAUSAL_AT_LSAT"
    else:
        verdict = "INCONCLUSIVE_DO_NOT_TUNE"

    with (args.out_dir / "notation_neutralization_records.jsonl").open("w", encoding="utf-8") as f:
        for i, (sample, row) in enumerate(zip(pop_samples, after)):
            f.write(json.dumps({
                "population_index": i,
                "source_unique_index": int(primary_idx[i]),
                "raw_index": sample["raw_index"],
                "a": sample["a"], "b": sample["b"], "digit": sample.get("digit"),
                "gold_side": g1.gold_side(sample),
                "baseline_completion": baseline[primary_idx[i]]["completion"],
                "rank_margin_before": float(rank_margin_before[i]),
                "notation_coordinate_before": float(notation_before[i]),
                "notation_threshold": float(c["tau_not"]),
                "intervention_l2": float(neutral_norm[i]),
                "rescue": bool(row["correct"]),
                **row,
            }, ensure_ascii=False) + "\n")

    with (args.out_dir / "random_null_records.jsonl").open("w", encoding="utf-8") as f:
        for row in null_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "verdict": verdict,
        **base_summary,
        "primary_n": int(len(primary_idx)),
        "manipulation_checks": manip,
        "R_not": r_not,
        "R_null_by_seed": dict(zip(map(str, RANDOM_SEEDS), null_rates)),
        "R_null_mean": r_null,
        "DeltaR": delta_r,
        "DeltaR_bootstrap_95ci": ci,
        "invalid_or_neither_rate": invalid_rate,
        "changed_valid_n": n_changed_valid,
        "correct_fraction_among_changed_valid": correct_among_changed,
    }
    (args.out_dir / "notation_causal_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
