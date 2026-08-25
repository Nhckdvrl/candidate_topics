#!/usr/bin/env python3
"""Frozen Topic 28 G1: adjacent truthful-clue order intervention."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import analyze_reversal_structure as analysis
import g0_progressive_reversal as g0


QUESTION_REVISION = "3dae05a66d3e0fd8c6b23ef8656ff6f4437bb1d4"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SEED = 20260825
BOOTSTRAP_REPS = 2000
MAX_NEW_TOKENS = 24
EXPECTED_PANEL = 498
EXPECTED_QIDS = 415

MIN_VALID_FRACTION = 0.98
MIN_FIRST_CORRECT = 100
MIN_COMMON_CORRECT = 75
MIN_ORIGINAL_REVERSALS = 20
MIN_DELTA_ORDER = 0.02
MIN_DELTA_FINAL_ERROR = 0.01

SYSTEM_PROMPT = (
    "You answer Quiz Bowl questions. Return only the short answer, "
    "with no explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/g1"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=None,
        help="Engineering smoke test only; debug runs cannot receive a verdict.",
    )
    parser.add_argument(
        "--panel-only",
        action="store_true",
        help="Write the outcome-blind panel without loading the model.",
    )
    return parser.parse_args()


def build_user_prompt(clues: Iterable[str]) -> str:
    numbered = "\n".join(f"{i}. {clue}" for i, clue in enumerate(clues, start=1))
    return (
        "Identify the answer described by these clues.\n\n"
        f"Clues:\n{numbered}\n\nAnswer:"
    )


def build_panel(questions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    q = questions.copy()
    idf = analysis.build_atomic_clue_idf(q)
    q["atomic_clue"] = [
        g0.extract_added_clue(full, spans, int(idx))
        for full, spans, idx in zip(q.full_quiz_question, q.clue_spans, q.n_clues)
    ]
    if q["atomic_clue"].isna().any() or q["atomic_clue"].str.strip().eq("").any():
        raise RuntimeError("Question artifact contains an empty atomic clue")
    q["specificity"] = q["atomic_clue"].map(
        lambda clue: analysis.clue_specificity(clue, idf)
    )
    threshold = float(q["specificity"].median())
    q["qid"] = q["orig_qid"].astype(str)
    q["clue_idx"] = pd.to_numeric(q["n_clues"], errors="raise").astype(int)
    lookup = {(r.qid, int(r.clue_idx)): r for r in q.itertuples(index=False)}

    rows: list[dict] = []
    for (qid, trigger_idx), trigger in sorted(lookup.items()):
        t = trigger_idx - 1
        if t < 2:
            continue
        required = [(qid, idx) for idx in range(1, trigger_idx + 1)]
        if any(key not in lookup for key in required):
            continue
        metadata = trigger.metadata
        if not isinstance(metadata, dict) or not metadata.get("category"):
            continue
        if float(trigger.specificity) < threshold:
            continue
        aliases = tuple(trigger.alias_norms)
        if not aliases:
            continue

        atomic = [str(lookup[(qid, idx)].atomic_clue).strip() for idx in range(1, trigger_idx + 1)]
        if any(not clue for clue in atomic):
            continue
        prefix = atomic[: t - 1]
        clue_t = atomic[t - 1]
        clue_trigger = atomic[t]
        o1 = prefix + [clue_t]
        o2 = prefix + [clue_t, clue_trigger]
        s1 = prefix + [clue_trigger]
        s2 = prefix + [clue_trigger, clue_t]
        if sorted(o2) != sorted(s2) or len(o2) != len(s2):
            raise AssertionError("Final original/swap clue multisets differ")

        rows.append(
            {
                "boundary_id": f"{qid}_{t}_{trigger_idx}",
                "qid": qid,
                "t": t,
                "trigger_idx": trigger_idx,
                "category": metadata.get("category"),
                "subcategory": metadata.get("subcategory"),
                "answer": trigger.orig_answer_string,
                "aliases": aliases,
                "clue_t": clue_t,
                "trigger_clue": clue_trigger,
                "clue_t_specificity": float(lookup[(qid, t)].specificity),
                "trigger_specificity": float(trigger.specificity),
                "specificity_half": "Q4" if float(trigger.specificity) >= float(q["specificity"].quantile(0.75)) else "Q3",
                "o1_user": build_user_prompt(o1),
                "o2_user": build_user_prompt(o2),
                "s1_user": build_user_prompt(s1),
                "s2_user": build_user_prompt(s2),
                "o1_clue_count": len(o1),
                "o2_clue_count": len(o2),
                "s1_clue_count": len(s1),
                "s2_clue_count": len(s2),
            }
        )

    panel = pd.DataFrame(rows).sort_values(["qid", "t"]).reset_index(drop=True)
    receipt = {
        "question_rows": int(len(q)),
        "question_qids": int(q["qid"].nunique()),
        "specificity_median": threshold,
        "specificity_q75": float(q["specificity"].quantile(0.75)),
        "panel_boundaries": int(len(panel)),
        "panel_qids": int(panel["qid"].nunique()),
        "uses_response_outcomes": False,
    }
    return panel, receipt


def expand_states(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in panel.itertuples(index=False):
        for state in ("o1", "o2", "s1", "s2"):
            rows.append(
                {
                    "boundary_id": row.boundary_id,
                    "qid": row.qid,
                    "state": state,
                    "user_prompt": getattr(row, f"{state}_user"),
                    "aliases": row.aliases,
                }
            )
    return pd.DataFrame(rows)


def run_inference(states: pd.DataFrame, batch_size: int, device: str) -> pd.DataFrame:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for user in states["user_prompt"]
    ]
    raw_outputs: list[str] = []
    output_tokens: list[int] = []
    input_tokens: list[int] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=False
        ).to(device)
        in_len = encoded["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        raw_outputs.extend(x.strip() for x in decoded)
        output_tokens.extend(int((row != tokenizer.pad_token_id).sum().item()) for row in continuation)
        input_tokens.extend(int(x) for x in in_len)
        print(f"[generation] {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)

    out = states.copy()
    out["prediction"] = raw_outputs
    out["input_tokens"] = input_tokens
    out["output_tokens"] = output_tokens
    out["nonempty"] = out["prediction"].str.strip().ne("")
    out["not_truncated"] = out["output_tokens"] < MAX_NEW_TOKENS
    out["single_line"] = ~out["prediction"].str.contains(r"[\r\n]", regex=True)
    out["valid"] = out["nonempty"] & out["not_truncated"]
    out["correct"] = [
        g0.alias_exact(pred, aliases) if valid else False
        for pred, aliases, valid in zip(out.prediction, out.aliases, out.valid)
    ]
    out["prediction_norm"] = out["prediction"].map(g0.normalize_answer)
    out["torch_version"] = torch.__version__
    out["transformers_version"] = transformers.__version__
    return out


def pair_outputs(panel: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    pred = states.pivot(index="boundary_id", columns="state", values="prediction")
    norm = states.pivot(index="boundary_id", columns="state", values="prediction_norm")
    correct = states.pivot(index="boundary_id", columns="state", values="correct")
    valid = states.pivot(index="boundary_id", columns="state", values="valid")
    paired = panel.copy().set_index("boundary_id")
    for state in ("o1", "o2", "s1", "s2"):
        paired[f"{state}_prediction"] = pred[state]
        paired[f"{state}_prediction_norm"] = norm[state]
        paired[f"{state}_correct"] = correct[state].astype(bool)
        paired[f"{state}_valid"] = valid[state].astype(bool)
    paired["all_valid"] = paired[[f"{s}_valid" for s in ("o1", "o2", "s1", "s2")]].all(axis=1)
    paired["original_reversal"] = paired.o1_correct & ~paired.o2_correct
    paired["swap_reversal"] = paired.s1_correct & ~paired.s2_correct
    paired["original_final_error"] = ~paired.o2_correct
    paired["swap_final_error"] = ~paired.s2_correct
    paired["common_belief"] = paired.o1_correct & paired.s1_correct
    paired["original_only_final_harm"] = ~paired.o2_correct & paired.s2_correct
    paired["swap_only_final_harm"] = paired.o2_correct & ~paired.s2_correct
    paired["order_independent_final_error"] = ~paired.o2_correct & ~paired.s2_correct
    paired["four_state_pattern"] = [
        "".join("1" if bool(x) else "0" for x in vals)
        for vals in zip(paired.o1_correct, paired.o2_correct, paired.s1_correct, paired.s2_correct)
    ]
    return paired.reset_index()


def cluster_bootstrap_paired(
    df: pd.DataFrame,
    left: str,
    right: str,
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> tuple[float, float, float]:
    work = df[["qid", left, right]].copy()
    work["difference"] = work[left].astype(float) - work[right].astype(float)
    by_qid = work.groupby("qid")["difference"].agg(["sum", "count"])
    sums = by_qid["sum"].to_numpy(float)
    counts = by_qid["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(by_qid), size=(reps, len(by_qid)), replace=True)
    values = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(work["difference"].mean()), float(lo), float(hi)


def summarize(paired: pd.DataFrame, states: pd.DataFrame, debug: bool) -> dict:
    delta_order, order_lo, order_hi = cluster_bootstrap_paired(
        paired, "original_reversal", "swap_reversal"
    )
    delta_final, final_lo, final_hi = cluster_bootstrap_paired(
        paired, "original_final_error", "swap_final_error", seed=SEED + 1
    )
    n = len(paired)
    artifact_gates = {
        "panel_boundaries_exact": n == EXPECTED_PANEL,
        "unique_qids_exact": int(paired.qid.nunique()) == EXPECTED_QIDS,
        "valid_fraction": float(paired.all_valid.mean()) >= MIN_VALID_FRACTION,
        "o1_correct_support": int(paired.o1_correct.sum()) >= MIN_FIRST_CORRECT,
        "s1_correct_support": int(paired.s1_correct.sum()) >= MIN_FIRST_CORRECT,
        "common_belief_support": int(paired.common_belief.sum()) >= MIN_COMMON_CORRECT,
    }
    scientific_gates = {
        "original_reversal_support": int(paired.original_reversal.sum()) >= MIN_ORIGINAL_REVERSALS,
        "delta_order_magnitude": delta_order >= MIN_DELTA_ORDER,
        "delta_order_ci_positive": order_lo > 0,
        "delta_final_error_magnitude": delta_final >= MIN_DELTA_FINAL_ERROR,
        "delta_final_error_ci_positive": final_lo > 0,
    }
    if debug:
        verdict = "DEBUG_NO_VERDICT"
    elif not all(artifact_gates.values()):
        verdict = "STOP_G1_MEASUREMENT"
    elif all(scientific_gates.values()):
        verdict = "GO_PATH_DEPENDENT_REVERSAL"
    elif scientific_gates["delta_final_error_magnitude"] and scientific_gates["delta_final_error_ci_positive"]:
        verdict = "GO_ORDER_EFFECT_ONLY"
    else:
        verdict = "STOP_ORDER_DEPENDENCE"

    return {
        "verdict": verdict,
        "panel_boundaries": n,
        "unique_qids": int(paired.qid.nunique()),
        "valid_boundaries": int(paired.all_valid.sum()),
        "valid_fraction": float(paired.all_valid.mean()),
        "state_outputs": int(len(states)),
        "single_line_output_fraction": float(states.single_line.mean()),
        "o1_correct": int(paired.o1_correct.sum()),
        "o2_correct": int(paired.o2_correct.sum()),
        "s1_correct": int(paired.s1_correct.sum()),
        "s2_correct": int(paired.s2_correct.sum()),
        "common_belief": int(paired.common_belief.sum()),
        "original_reversals": int(paired.original_reversal.sum()),
        "swap_reversals": int(paired.swap_reversal.sum()),
        "delta_order": delta_order,
        "delta_order_ci95": [order_lo, order_hi],
        "original_final_errors": int(paired.original_final_error.sum()),
        "swap_final_errors": int(paired.swap_final_error.sum()),
        "delta_final_error": delta_final,
        "delta_final_error_ci95": [final_lo, final_hi],
        "original_only_final_harm": int(paired.original_only_final_harm.sum()),
        "swap_only_final_harm": int(paired.swap_only_final_harm.sum()),
        "order_independent_final_error": int(paired.order_independent_final_error.sum()),
        "artifact_gates": artifact_gates,
        "scientific_gates": scientific_gates,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    questions = g0.load_questions(QUESTION_REVISION, args.cache_dir)
    panel, panel_receipt = build_panel(questions)
    full_run = args.debug_limit is None
    if full_run and (len(panel) != EXPECTED_PANEL or panel.qid.nunique() != EXPECTED_QIDS):
        raise RuntimeError(f"Frozen panel contract failed: {len(panel)} boundaries, {panel.qid.nunique()} qids")
    if args.debug_limit is not None:
        panel = panel.head(args.debug_limit).copy()

    panel_to_write = panel.drop(columns=[c for c in panel if c.endswith("_user")]).copy()
    panel_to_write["aliases"] = panel_to_write["aliases"].map(json.dumps)
    panel_to_write.to_csv(args.out_dir / "panel.csv", index=False)
    if args.panel_only:
        receipt = {
            **panel_receipt,
            "panel_only": True,
            "debug": not full_run,
            "question_revision": QUESTION_REVISION,
        }
        (args.out_dir / "g1_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return

    states = expand_states(panel)
    states = run_inference(states, args.batch_size, args.device)
    paired = pair_outputs(panel, states)
    summary = summarize(paired, states, debug=not full_run)

    state_write = states.copy()
    state_write["aliases"] = state_write["aliases"].map(json.dumps)
    state_write.to_csv(args.out_dir / "state_outputs.csv", index=False)
    paired_write = paired.drop(columns=[c for c in paired if c.endswith("_user")]).copy()
    paired_write["aliases"] = paired_write["aliases"].map(json.dumps)
    paired_write.to_csv(args.out_dir / "paired_results.csv", index=False)
    paired["four_state_pattern"].value_counts().rename_axis(
        "four_state_pattern"
    ).reset_index(name="n").sort_values("four_state_pattern").to_csv(
        args.out_dir / "four_state_patterns.csv", index=False
    )
    for column in ("specificity_half", "category", "t"):
        grouped = (
            paired.groupby(column, dropna=False)
            .agg(
                n=("boundary_id", "size"),
                original_reversals=("original_reversal", "sum"),
                swap_reversals=("swap_reversal", "sum"),
                original_final_errors=("original_final_error", "sum"),
                swap_final_errors=("swap_final_error", "sum"),
            )
            .reset_index()
        )
        grouped.to_csv(args.out_dir / f"summary_by_{column}.csv", index=False)

    receipt = {
        **panel_receipt,
        "panel_only": False,
        "debug": not full_run,
        "question_revision": QUESTION_REVISION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
        "python": platform.python_version(),
        "device": args.device,
    }
    (args.out_dir / "g1_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.out_dir / "g1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
