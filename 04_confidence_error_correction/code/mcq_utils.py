"""Shared MCQ scoring utilities for Topic 04."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import torch


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


PRIMARY_INSTRUCTION = "Choose the single best option. Reply with only its letter."
ALTERNATE_INSTRUCTION = "Which option is correct? Return only the option letter."


def cyclic_permutations(k: int) -> list[list[int]]:
    base = list(range(k))
    return [base[s:] + base[:s] for s in range(k)]


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def format_user_message(question: str, choices: list[str], template: str = "primary") -> str:
    if len(choices) > len(LABELS):
        raise ValueError(f"Too many choices ({len(choices)} > {len(LABELS)})")
    instruction = PRIMARY_INSTRUCTION if template == "primary" else ALTERNATE_INSTRUCTION
    return "\n".join(
        [
            question.strip(),
            "",
            "Options:",
            *[f"{LABELS[i]}. {choice}" for i, choice in enumerate(choices)],
            "",
            instruction,
        ]
    )



def format_training_user_message(question: str, choices: list[str]) -> str:
    if len(choices) > len(LABELS):
        raise ValueError(f"Too many choices ({len(choices)} > {len(LABELS)})")
    return "\n".join(
        [
            question.strip(),
            "",
            "Options:",
            *[f"{LABELS[i]}. {choice}" for i, choice in enumerate(choices)],
            "",
            "Give the correct answer. Respond in the form: Answer: <letter>. <answer text>",
        ]
    )


def chat_prompt_ids(tokenizer, question: str, choices: list[str], template: str = "primary") -> list[int]:
    user = format_user_message(question, choices, template)
    messages = [{"role": "user", "content": user}]
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
    except Exception:
        # Explicit fallback for tokenizers without a chat template.
        text = user + "\nAnswer:"
        ids = tokenizer.encode(text, add_special_tokens=True)
    # Newer Transformers/tokenizer combinations may return a BatchEncoding or
    # rendered text even when tokenize=True and return_tensors=None. The
    # scientific prompt and scoring boundary are unchanged; normalize those
    # representations here.
    if hasattr(ids, "get") and ids.get("input_ids") is not None:
        ids = ids["input_ids"]
    if isinstance(ids, str):
        ids = tokenizer.encode(ids, add_special_tokens=False)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def semantic_metrics(probs: list[float], answer: int) -> dict:
    if not (0 <= answer < len(probs)):
        raise ValueError("answer out of range")
    eps = 1e-12
    p_correct = float(probs[answer])
    wrong_ids = [i for i in range(len(probs)) if i != answer]
    wrong_mass = max(sum(float(probs[i]) for i in wrong_ids), eps)
    q = [float(probs[i]) / wrong_mass for i in wrong_ids]
    top_wrong = max(wrong_ids, key=lambda i: probs[i])
    sorted_wrong = sorted((float(probs[i]) for i in wrong_ids), reverse=True)
    entropy_q = -sum(v * math.log(max(v, eps)) for v in q)
    entropy_norm = entropy_q / math.log(len(q)) if len(q) > 1 else 0.0
    answer_entropy = -sum(float(p) * math.log(max(float(p), eps)) for p in probs)
    target_rank = 1 + sum(float(p) > p_correct for p in probs)
    return {
        "p_correct": p_correct,
        "wrong_concentration": max(q),
        "wrong_entropy_norm": entropy_norm,
        "wrong_concentration_entropy": 1.0 - entropy_norm,
        "top_wrong": int(top_wrong),
        "top_wrong_probability": float(probs[top_wrong]),
        "wrong_top12_margin": sorted_wrong[0] - sorted_wrong[1] if len(sorted_wrong) > 1 else sorted_wrong[0],
        "answer_entropy": answer_entropy,
        "target_rank": int(target_rank),
        "top1_correct": int(max(range(len(probs)), key=lambda i: probs[i]) == answer),
    }


@dataclass
class PromptRecord:
    item_index: int
    permutation_index: int
    permutation: list[int]
    prompt_ids: list[int]


def build_prompt_records(tokenizer, items: list[dict], template: str) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for item_index, item in enumerate(items):
        choices = list(item["choices"])
        for permutation_index, perm in enumerate(cyclic_permutations(len(choices))):
            permuted = [choices[i] for i in perm]
            records.append(
                PromptRecord(
                    item_index=item_index,
                    permutation_index=permutation_index,
                    permutation=perm,
                    prompt_ids=chat_prompt_ids(tokenizer, item["question"], permuted, template),
                )
            )
    return records


def label_token_sequences(tokenizer, k: int) -> list[list[int]]:
    seqs = []
    for i in range(k):
        ids = tokenizer.encode(LABELS[i], add_special_tokens=False)
        if not ids:
            raise ValueError(f"Label {LABELS[i]} tokenized to empty sequence")
        seqs.append(list(ids))
    return seqs


@torch.inference_mode()
def score_single_token_labels_batched(
    model,
    tokenizer,
    records: list[PromptRecord],
    k: int,
    batch_size: int,
    device: torch.device,
) -> list[list[float]]:
    """Fast path: all labels are one token; one forward pass per prompt."""
    label_seqs = label_token_sequences(tokenizer, k)
    if not all(len(x) == 1 for x in label_seqs):
        raise ValueError("single-token fast path requested for multi-token labels")
    label_ids = torch.tensor([x[0] for x in label_seqs], device=device, dtype=torch.long)
    out: list[list[float]] = []

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            encoded = tokenizer.pad(
                [{"input_ids": r.prompt_ids} for r in batch],
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
            candidate_logits = logits.index_select(-1, label_ids)
            probs = torch.softmax(candidate_logits.float(), dim=-1)
            out.extend(probs.cpu().tolist())
    finally:
        tokenizer.padding_side = old_padding_side
    return out


@torch.inference_mode()
def exact_candidate_sequence_logprob(
    model,
    prompt_ids: list[int],
    candidate_ids: list[int],
    device: torch.device,
) -> float:
    """Exact log p(candidate token sequence | fixed prompt token sequence)."""
    full = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device=device)
    logits = model(full).logits[:, :-1].log_softmax(-1)
    target = full[:, 1:]
    token_lp = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    start = max(len(prompt_ids) - 1, 0)
    return float(token_lp[:, start:].sum().item())


def score_multi_token_labels_slow(
    model,
    tokenizer,
    records: list[PromptRecord],
    k: int,
    device: torch.device,
) -> list[list[float]]:
    """Portable fallback for tokenizers where A/B/... are multi-token."""
    seqs = label_token_sequences(tokenizer, k)
    out = []
    for rec in records:
        scores = [
            exact_candidate_sequence_logprob(model, rec.prompt_ids, seq, device)
            for seq in seqs
        ]
        probs = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0).tolist()
        out.append(probs)
    return out


def map_permuted_probs_to_semantic(local_probs: list[float], permutation: list[int]) -> list[float]:
    semantic = [0.0] * len(permutation)
    for local_idx, original_idx in enumerate(permutation):
        semantic[original_idx] = float(local_probs[local_idx])
    return semantic
