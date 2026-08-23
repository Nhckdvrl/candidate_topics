#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


FINAL_RE = re.compile(
    r"(?:FINAL_DIAGNOSIS|FINAL\s+DIAGNOSIS|FINAL\s+ANSWER|DIAGNOSIS)\s*:\s*(.+)",
    flags=re.IGNORECASE,
)
DIAGNOSIS_IS_RE = re.compile(
    r"(?:most\s+likely\s+diagnosis|final\s+diagnosis|diagnosis)\s+(?:is|would\s+be)\s+(.+)",
    flags=re.IGNORECASE,
)


@dataclass
class GenerationResult:
    full_text: str
    final_text: str
    new_token_count: int
    thinking_closed: bool
    hit_max_tokens: bool
    stopped_on_eos: bool


def normalize(text: str) -> str:
    text = (text or "").lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9+\-/ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_answer_line(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^[\s#>*\-]+", "", text).strip()
    text = re.sub(r"\*\*|__|`", "", text)
    return text.strip().rstrip(". ;")


def resolve_diagnosis(candidate: str | None, labels: list[str]) -> str | None:
    """Conservatively map one candidate string to exactly one canonical dataset label."""
    ans = normalize(candidate or "")
    if not ans:
        return None

    exact = [label for label in labels if normalize(label) == ans]
    if len(exact) == 1:
        return exact[0]

    # Canonical-label containment is intentionally conservative. Prefer the
    # longest canonical label so e.g. a longer diagnosis is not shadowed by a
    # shorter substring label.
    hits = [label for label in labels if normalize(label) and normalize(label) in ans]
    if hits:
        longest_len = max(len(normalize(label)) for label in hits)
        longest = [label for label in hits if len(normalize(label)) == longest_len]
        if len(longest) == 1:
            return longest[0]
    return None


def extract_prediction(final_text: str, labels: list[str]) -> tuple[str | None, str | None, str]:
    """Extract only from the post-thinking final answer, never from reasoning.

    Returns (canonical_prediction, extracted_candidate, extraction_method).
    No external judge and no semantic fuzzy matching are used.
    """
    text = (final_text or "").strip()
    if not text:
        return None, None, "empty_final"

    marker_matches = FINAL_RE.findall(text)
    for candidate in reversed(marker_matches):
        candidate = _strip_answer_line(candidate.splitlines()[0])
        pred = resolve_diagnosis(candidate, labels)
        if pred is not None:
            return pred, candidate, "explicit_marker"

    diag_matches = DIAGNOSIS_IS_RE.findall(text)
    for candidate in reversed(diag_matches):
        candidate = _strip_answer_line(candidate.splitlines()[0])
        pred = resolve_diagnosis(candidate, labels)
        if pred is not None:
            return pred, candidate, "diagnosis_is"

    # Prefer later lines because benchmark prompts ask for the final diagnosis
    # after reasoning. This is applied only to post-</think> content.
    lines = [_strip_answer_line(x) for x in text.splitlines() if _strip_answer_line(x)]
    for line in reversed(lines):
        pred = resolve_diagnosis(line, labels)
        if pred is not None:
            return pred, line, "resolved_final_line"

    # Last-resort canonical mention in final-answer content only. If several
    # different canonical labels occur, use the last textual mention. This
    # avoids reading diagnoses mentioned inside the hidden thinking block.
    norm_text = normalize(text)
    mentions: list[tuple[int, int, str]] = []
    for label in labels:
        nl = normalize(label)
        if not nl:
            continue
        start = 0
        while True:
            pos = norm_text.find(nl, start)
            if pos < 0:
                break
            mentions.append((pos, len(nl), label))
            start = pos + max(1, len(nl))
    if mentions:
        mentions.sort(key=lambda x: (x[0], x[1]))
        return mentions[-1][2], mentions[-1][2], "last_canonical_mention"

    return None, None, "unresolved_final"


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


def stable_pair_seed(global_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompt(tok, row: dict, mode: str) -> str:
    if mode == "cot":
        instruction = (
            "Analyze the clinical case step by step and determine the single most likely diagnosis. "
            "After your reasoning, end with a concise final diagnosis. For easy benchmark parsing, "
            "prefer the form: FINAL_DIAGNOSIS: <diagnosis>."
        )
    elif mode == "direct":
        instruction = (
            "Determine the single most likely diagnosis from the clinical case. Do not explain. "
            "Return one concise diagnosis; prefer the form: FINAL_DIAGNOSIS: <diagnosis>."
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


def _eos_ids(tok) -> set[int]:
    value = tok.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(x) for x in value}


@torch.inference_mode()
def generate(model, tok, text: str, max_new_tokens: int, mode: str, generation_seed: int) -> GenerationResult:
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)

    set_generation_seed(generation_seed)
    kwargs = dict(
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if mode == "cot":
        # Qwen3 official thinking-mode recommendation. Greedy decoding is
        # explicitly discouraged by the model card.
        kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    else:
        # Direct mode is deliberately deterministic for later fixed-position
        # mechanism experiments.
        kwargs.update(do_sample=False)

    out = model.generate(**kwargs)
    new_ids = out[0, ids.shape[1]:].tolist()
    eos_ids = _eos_ids(tok)
    stopped_on_eos = bool(new_ids and new_ids[-1] in eos_ids)
    hit_max_tokens = len(new_ids) >= max_new_tokens and not stopped_on_eos

    thinking_closed = mode != "cot"
    final_ids = new_ids
    if mode == "cot":
        think_end_id = tok.convert_tokens_to_ids("</think>")
        if isinstance(think_end_id, int) and think_end_id >= 0 and think_end_id in new_ids:
            # Use the LAST close marker defensively and score only what follows.
            idx = len(new_ids) - 1 - new_ids[::-1].index(think_end_id)
            final_ids = new_ids[idx + 1:]
            thinking_closed = True
        else:
            thinking_closed = False
            final_ids = [] if hit_max_tokens else new_ids

    full_text = tok.decode(new_ids, skip_special_tokens=True).strip()
    final_text = tok.decode(final_ids, skip_special_tokens=True).strip()
    return GenerationResult(
        full_text=full_text,
        final_text=final_text,
        new_token_count=len(new_ids),
        thinking_closed=thinking_closed,
        hit_max_tokens=hit_max_tokens,
        stopped_on_eos=stopped_on_eos,
    )


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
    ap.add_argument("--cot-max-new-tokens", type=int, default=32768)
    ap.add_argument("--direct-max-new-tokens", type=int, default=64)
    ap.add_argument("--outdir", default="artifacts/g0_behavior_cot")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "records.jsonl"
    invalid_path = out / "invalid_examples.jsonl"
    record_path.unlink(missing_ok=True)
    invalid_path.unlink(missing_ok=True)

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
        case_id = str(control["case_id"])
        pair_seed = stable_pair_seed(args.seed, case_id)

        # Common-random-number sampling: control and trap receive the same
        # deterministic sampling stream in thinking mode.
        control_gen = generate(model, tok, prompt(tok, control, args.mode), max_new_tokens, args.mode, pair_seed)
        trap_gen = generate(model, tok, prompt(tok, trap, args.mode), max_new_tokens, args.mode, pair_seed)

        control_pred, control_candidate, control_method = extract_prediction(control_gen.final_text, labels)
        trap_pred, trap_candidate, trap_method = extract_prediction(trap_gen.final_text, labels)

        control_gt = control["ground_truth"]
        trap_gt = trap["ground_truth"]

        invalid_reasons = []
        for branch, gen, pred in (
            ("control", control_gen, control_pred),
            ("trap", trap_gen, trap_pred),
        ):
            if gen.hit_max_tokens:
                invalid_reasons.append(f"{branch}:hit_max_tokens")
            if args.mode == "cot" and not gen.thinking_closed:
                invalid_reasons.append(f"{branch}:thinking_not_closed")
            if pred is None:
                invalid_reasons.append(f"{branch}:unresolved_final")

        row = {
            "case_id": case_id,
            "mode": args.mode,
            "pair_generation_seed": pair_seed,
            "control_gt": control_gt,
            "trap_gt": trap_gt,
            "control_output": control_gen.full_text,
            "trap_output": trap_gen.full_text,
            "control_final_text": control_gen.final_text,
            "trap_final_text": trap_gen.final_text,
            "control_candidate": control_candidate,
            "trap_candidate": trap_candidate,
            "control_extract_method": control_method,
            "trap_extract_method": trap_method,
            "control_pred": control_pred,
            "trap_pred": trap_pred,
            "control_new_tokens": control_gen.new_token_count,
            "trap_new_tokens": trap_gen.new_token_count,
            "control_thinking_closed": control_gen.thinking_closed,
            "trap_thinking_closed": trap_gen.thinking_closed,
            "control_hit_max_tokens": control_gen.hit_max_tokens,
            "trap_hit_max_tokens": trap_gen.hit_max_tokens,
            "control_correct": control_pred == control_gt,
            "trap_correct": trap_pred == trap_gt,
            "bias_trap": bool(control_pred == control_gt and trap_gt != control_gt and trap_pred == control_gt),
            "invalid": bool(invalid_reasons),
            "invalid_reasons": invalid_reasons,
        }
        recs.append(row)
        with record_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if invalid_reasons:
            with invalid_path.open("a", encoding="utf-8") as f:
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

    invalid_reason_counts = Counter(reason for row in recs for reason in row["invalid_reasons"])
    extraction_method_counts = Counter()
    for row in recs:
        extraction_method_counts[f"control:{row['control_extract_method']}"] += 1
        extraction_method_counts[f"trap:{row['trap_extract_method']}"] += 1

    control_hit_cap = sum(row["control_hit_max_tokens"] for row in recs)
    trap_hit_cap = sum(row["trap_hit_max_tokens"] for row in recs)
    control_closed = sum(row["control_thinking_closed"] for row in recs)
    trap_closed = sum(row["trap_thinking_closed"] for row in recs)

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
        decoding = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
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
        decoding = {"do_sample": False}

    summary = {
        "repair_version": "g0b_measurement_repair_v2",
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "sampled_pairs": n,
        "max_new_tokens": max_new_tokens,
        "decoding": decoding,
        "sample_case_ids": [row["case_id"] for row in recs],
        "control_accuracy": control_acc,
        "trap_accuracy": trap_acc,
        "control_correct_count": len(control_correct),
        "bias_trap_count": len(bias_traps),
        "bias_trap_rate_among_control_correct": btr,
        "bias_trap_rate_wilson_95": [btr_lo, btr_hi],
        "bias_trap_transition_count": transition_count,
        "invalid_rate": invalid_rate,
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "extraction_method_counts": dict(sorted(extraction_method_counts.items())),
        "thinking_diagnostics": {
            "control_thinking_closed_count": control_closed,
            "trap_thinking_closed_count": trap_closed,
            "control_hit_max_tokens_count": control_hit_cap,
            "trap_hit_max_tokens_count": trap_hit_cap,
        },
        "paper_reference_for_qwen3_14b": {"baseline_accuracy": 0.4412, "bias_trap_rate": 0.5419},
        "gate": gate,
        "verdict": go_verdict if all(gate.values()) else stop_verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
