#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from g0_bias_trap_screen import generate, wilson_interval


MAPPER_SALT = "medeinst-g0b-v3-dual-order-canonicalizer"


def mapper_label_orders(labels: list[str]) -> tuple[list[str], list[str]]:
    """Return two deterministic permutations of the closed benchmark label space."""
    first = list(labels)
    second = sorted(
        labels,
        key=lambda label: hashlib.sha256(f"{MAPPER_SALT}:{label}".encode("utf-8")).hexdigest(),
    )
    if len(labels) > 1 and second == first:
        second = list(reversed(second))
    return first, second


def build_mapper_user_text(final_text: str, ordered_labels: list[str]) -> str:
    menu = "\n".join(f"{i}. {label}" for i, label in enumerate(ordered_labels, start=1))
    return (
        "Map the diagnosis phrase below to the benchmark's CLOSED label vocabulary.\n"
        "You are a semantic label canonicalizer, NOT a clinician. Do not diagnose a patient and do not infer from missing case information.\n"
        "Choose a label only when the diagnosis text is semantically equivalent to that label.\n"
        "If the text is ambiguous, gives multiple alternative diagnoses without selecting one, or has no equivalent label, return 0.\n"
        "Return ONLY the integer ID (0-49), with no explanation.\n\n"
        f"DIAGNOSIS TEXT:\n{final_text.strip()}\n\n"
        f"CLOSED LABELS:\n{menu}\n\n"
        "ID:"
    )


def mapper_prompt(tok, final_text: str, ordered_labels: list[str]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Deterministic benchmark label canonicalization only. "
                "You will never receive a patient case or ground truth."
            ),
        },
        {"role": "user", "content": build_mapper_user_text(final_text, ordered_labels)},
    ]
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] + "\n\n" + messages[1]["content"] + "\n"


def parse_mapper_choice(raw_text: str, ordered_labels: list[str]) -> str | None:
    """Strictly accept one numeric ID; 0 is an explicit abstention."""
    text = (raw_text or "").strip()
    match = re.fullmatch(r"(\d{1,2})\s*[.)]?", text)
    if match is None:
        return None
    idx = int(match.group(1))
    if idx == 0:
        return None
    if idx < 1 or idx > len(ordered_labels):
        return None
    return ordered_labels[idx - 1]


def map_once(model, tok, final_text: str, ordered_labels: list[str], max_new_tokens: int) -> tuple[str | None, str]:
    result = generate(
        model,
        tok,
        mapper_prompt(tok, final_text, ordered_labels),
        max_new_tokens=max_new_tokens,
        mode="direct",
        generation_seed=0,
    )
    return parse_mapper_choice(result.final_text, ordered_labels), result.final_text


def dual_order_canonicalize(
    model,
    tok,
    final_text: str,
    labels: list[str],
    max_new_tokens: int,
) -> tuple[str | None, dict]:
    """
    Canonicalize using two fixed label orders and accept only order-invariant agreement.

    The mapper sees only post-thinking final-answer text and the closed label vocabulary.
    It never sees the clinical narrative, case type, paired branch, or ground truth.
    """
    if not (final_text or "").strip():
        return None, {"accepted": False, "reason": "empty_final", "raw": []}

    orders = mapper_label_orders(labels)
    mapped: list[str | None] = []
    raws: list[str] = []
    for order in orders:
        pred, raw = map_once(model, tok, final_text, order, max_new_tokens)
        mapped.append(pred)
        raws.append(raw)

    accepted = mapped[0] is not None and mapped[0] == mapped[1]
    return (
        mapped[0] if accepted else None,
        {
            "accepted": accepted,
            "mapped": mapped,
            "raw": raws,
            "reason": "dual_order_agreement" if accepted else "dual_order_disagreement_or_abstention",
        },
    )


def run_mapper_preflight(model, tok, labels: list[str], max_new_tokens: int) -> dict:
    """Require exact self-mapping for every canonical label under both fixed orders."""
    failures = []
    for label in labels:
        pred, audit = dual_order_canonicalize(
            model,
            tok,
            f"Final diagnosis: {label}",
            labels,
            max_new_tokens,
        )
        if pred != label:
            failures.append({"label": label, "prediction": pred, "audit": audit})
    return {
        "n_labels": len(labels),
        "passed": len(failures) == 0,
        "failure_count": len(failures),
        "failures": failures,
    }


def recompute_invalid_reasons(row: dict, branch: str, pred: str | None, mode: str) -> list[str]:
    reasons = []
    if bool(row.get(f"{branch}_hit_max_tokens", False)):
        reasons.append(f"{branch}:hit_max_tokens")
    if mode == "cot" and not bool(row.get(f"{branch}_thinking_closed", False)):
        reasons.append(f"{branch}:thinking_not_closed")
    if pred is None:
        reasons.append(f"{branch}:unresolved_final")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-records", required=True)
    ap.add_argument("--dataset", default="zhui711/MedEinst")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--mode", choices=["cot", "direct"], required=True)
    ap.add_argument("--mapper-max-new-tokens", type=int, default=8)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    input_path = Path(args.input_records)
    if not input_path.exists():
        raise SystemExit(f"missing frozen input records: {input_path}")

    original = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not original:
        raise SystemExit("no records found")
    modes = {str(row.get("mode")) for row in original}
    if modes != {args.mode}:
        raise SystemExit(f"record mode mismatch: expected {args.mode}, saw {sorted(modes)}")

    ds = load_dataset(args.dataset, split=args.split)
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

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "records.jsonl"
    invalid_path = out / "invalid_examples.jsonl"
    record_path.unlink(missing_ok=True)
    invalid_path.unlink(missing_ok=True)

    preflight = run_mapper_preflight(model, tok, labels, args.mapper_max_new_tokens)
    (out / "canonicalizer_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if not preflight["passed"]:
        summary = {
            "repair_version": "g0_measurement_repair_v3_dual_order_canonicalizer",
            "mode": args.mode,
            "model": args.model,
            "sampled_pairs": len(original),
            "canonicalizer_preflight": preflight,
            "verdict": "CANONICALIZER_PREFLIGHT_FAILURE",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise SystemExit("canonicalizer preflight failed; no benchmark rows rescored")

    recs = []
    mapper_attempts = Counter()
    mapper_accepts = Counter()

    for frozen in original:
        row = dict(frozen)
        row["measurement_repair"] = "v3_dual_order_semantic_canonicalizer"

        new_preds: dict[str, str | None] = {}
        new_methods: dict[str, str] = {}
        new_candidates: dict[str, str | None] = {}
        mapper_audits: dict[str, dict | None] = {}

        for branch in ("control", "trap"):
            pred = row.get(f"{branch}_pred")
            method = str(row.get(f"{branch}_extract_method") or "unknown")
            candidate = row.get(f"{branch}_candidate")
            audit = None

            nonsemantic_invalid = bool(row.get(f"{branch}_hit_max_tokens", False)) or (
                args.mode == "cot" and not bool(row.get(f"{branch}_thinking_closed", False))
            )
            if pred is None and not nonsemantic_invalid:
                mapper_attempts[branch] += 1
                mapped, audit = dual_order_canonicalize(
                    model,
                    tok,
                    str(row.get(f"{branch}_final_text") or ""),
                    labels,
                    args.mapper_max_new_tokens,
                )
                if mapped is not None:
                    pred = mapped
                    method = "semantic_canonicalizer_v3"
                    candidate = str(row.get(f"{branch}_final_text") or "").strip()
                    mapper_accepts[branch] += 1

            new_preds[branch] = pred
            new_methods[branch] = method
            new_candidates[branch] = candidate
            mapper_audits[branch] = audit

        control_pred = new_preds["control"]
        trap_pred = new_preds["trap"]
        control_gt = row["control_gt"]
        trap_gt = row["trap_gt"]

        control_invalid = recompute_invalid_reasons(row, "control", control_pred, args.mode)
        trap_invalid = recompute_invalid_reasons(row, "trap", trap_pred, args.mode)
        invalid_reasons = control_invalid + trap_invalid
        control_valid = not control_invalid
        trap_valid = not trap_invalid
        control_correct = bool(control_valid and control_pred == control_gt)
        trap_correct = bool(trap_valid and trap_pred == trap_gt)
        bias_trap = bool(control_correct and trap_valid and trap_gt != control_gt and trap_pred == control_gt)

        row.update(
            {
                "control_pred": control_pred,
                "trap_pred": trap_pred,
                "control_candidate": new_candidates["control"],
                "trap_candidate": new_candidates["trap"],
                "control_extract_method": new_methods["control"],
                "trap_extract_method": new_methods["trap"],
                "control_mapper_audit": mapper_audits["control"],
                "trap_mapper_audit": mapper_audits["trap"],
                "control_valid": control_valid,
                "trap_valid": trap_valid,
                "control_correct": control_correct,
                "trap_correct": trap_correct,
                "bias_trap": bias_trap,
                "invalid": bool(invalid_reasons),
                "invalid_reasons": invalid_reasons,
            }
        )
        recs.append(row)
        with record_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if invalid_reasons:
            with invalid_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(recs)
    control_correct_rows = [row for row in recs if row["control_correct"]]
    bias_traps = [row for row in recs if row["bias_trap"]]
    control_acc = len(control_correct_rows) / max(1, n)
    trap_acc = sum(row["trap_correct"] for row in recs) / max(1, n)
    invalid_rate = sum(row["invalid"] for row in recs) / max(1, n)
    btr = len(bias_traps) / max(1, len(control_correct_rows))
    btr_lo, btr_hi = wilson_interval(len(bias_traps), len(control_correct_rows))
    transition_count = len({(row["control_gt"], row["trap_gt"]) for row in bias_traps})

    invalid_reason_counts = Counter(reason for row in recs for reason in row["invalid_reasons"])
    extraction_method_counts = Counter()
    for row in recs:
        extraction_method_counts[f"control:{row['control_extract_method']}"] += 1
        extraction_method_counts[f"trap:{row['trap_extract_method']}"] += 1

    if args.mode == "cot":
        gate = {
            "control_accuracy_ge_0.35": control_acc >= 0.35,
            "control_correct_count_ge_50": len(control_correct_rows) >= 50,
            "bias_trap_count_ge_20": len(bias_traps) >= 20,
            "bias_trap_rate_ge_0.30": btr >= 0.30,
            "bias_trap_wilson_lower_ge_0.20": btr_lo >= 0.20,
            "bias_trap_transition_count_ge_8": transition_count >= 8,
            "invalid_rate_le_0.10": invalid_rate <= 0.10,
        }
        go_verdict = "SEED_PHENOMENON_REPRODUCED"
        scientific_stop = "SEED_PHENOMENON_NOT_REPRODUCED"
        measurement_stop = "MEASUREMENT_CANONICALIZATION_FAILURE"
    else:
        gate = {
            "control_accuracy_ge_0.30": control_acc >= 0.30,
            "control_correct_count_ge_40": len(control_correct_rows) >= 40,
            "bias_trap_count_ge_16": len(bias_traps) >= 16,
            "bias_trap_rate_ge_0.20": btr >= 0.20,
            "bias_trap_wilson_lower_ge_0.10": btr_lo >= 0.10,
            "bias_trap_transition_count_ge_6": transition_count >= 6,
            "invalid_rate_le_0.10": invalid_rate <= 0.10,
        }
        go_verdict = "DIRECT_MODE_MECHANISM_OBJECT_READY"
        scientific_stop = "DIRECT_MODE_MECHANISM_OBJECT_TOO_WEAK"
        measurement_stop = "DIRECT_MODE_CANONICALIZATION_FAILURE"

    if invalid_rate > 0.10:
        verdict = measurement_stop
    elif all(gate.values()):
        verdict = go_verdict
    else:
        verdict = scientific_stop

    order_a, order_b = mapper_label_orders(labels)
    summary = {
        "repair_version": "g0_measurement_repair_v3_dual_order_canonicalizer",
        "source_records": str(input_path),
        "source_records_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "model": args.model,
        "mode": args.mode,
        "sampled_pairs": n,
        "sample_case_ids": [row["case_id"] for row in recs],
        "canonicalizer": {
            "fallback_only": True,
            "input_scope": "post-thinking final-answer text only; no narrative, case_type, pair branch, or ground truth",
            "abstention_id": 0,
            "dual_order_agreement_required": True,
            "mapper_max_new_tokens": args.mapper_max_new_tokens,
            "order_a_sha256": hashlib.sha256("\n".join(order_a).encode("utf-8")).hexdigest(),
            "order_b_sha256": hashlib.sha256("\n".join(order_b).encode("utf-8")).hexdigest(),
            "preflight": preflight,
            "attempts": dict(mapper_attempts),
            "accepted": dict(mapper_accepts),
        },
        "control_accuracy": control_acc,
        "trap_accuracy": trap_acc,
        "control_correct_count": len(control_correct_rows),
        "bias_trap_count": len(bias_traps),
        "bias_trap_rate_among_control_correct": btr,
        "bias_trap_rate_wilson_95": [btr_lo, btr_hi],
        "bias_trap_transition_count": transition_count,
        "invalid_rate": invalid_rate,
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "extraction_method_counts": dict(sorted(extraction_method_counts.items())),
        "paper_reference_for_qwen3_14b": {"baseline_accuracy": 0.4412, "bias_trap_rate": 0.5419},
        "gate": gate,
        "verdict": verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
