"""Shared MCQ scoring utilities for Topic 04 G-1v2."""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

import torch


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PRIMARY_INSTRUCTION = "Choose the single best option. Reply with only its letter."
ALTERNATE_INSTRUCTION = "Which option is correct? Return only the option letter."


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def balanced_permutations(k: int, scheme: str = "cyclic", item_id: str | None = None) -> list[list[int]]:
    """Return K balanced permutations: every semantic choice occupies every label once.

    `cyclic` reproduces G-1v1.
    `hashed_cyclic` deterministically shuffles the base order per item and then
    takes all cyclic shifts. It is an independent balanced family used only for
    measurement-reliability auditing.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if scheme == "cyclic":
        base = list(range(k))
    elif scheme == "hashed_cyclic":
        if item_id is None:
            raise ValueError("hashed_cyclic requires item_id")
        base = list(range(k))
        random.Random(stable_int(f"topic04:{item_id}:hashed_cyclic")).shuffle(base)
        if base == list(range(k)) and k > 1:
            base = base[1:] + base[:1]
    else:
        raise ValueError(f"unknown permutation scheme: {scheme}")
    return [base[s:] + base[:s] for s in range(k)]


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
        text = user + "\nAnswer:"
        ids = tokenizer.encode(text, add_special_tokens=True)
    if hasattr(ids, "get") and ids.get("input_ids") is not None:
        ids = ids["input_ids"]
    if isinstance(ids, str):
        ids = tokenizer.encode(ids, add_special_tokens=False)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def normalize_distribution(values: list[float], eps: float = 1e-12) -> list[float]:
    xs = [max(float(x), eps) for x in values]
    s = sum(xs)
    if not math.isfinite(s) or s <= 0:
        raise ValueError("invalid probability vector")
    return [x / s for x in xs]


def geometric_mean_distribution(permutation_probs: list[list[float]], eps: float = 1e-12) -> list[float]:
    """Debias a complete balanced permutation set in log-probability space.

    Under z_{r,j} = semantic_j + position_{r,j}, every semantic choice visits
    every position once, so the mean position term is a choice-independent
    constant and vanishes after the final softmax.
    """
    if not permutation_probs:
        raise ValueError("empty permutation_probs")
    k = len(permutation_probs[0])
    if any(len(row) != k for row in permutation_probs):
        raise ValueError("ragged permutation_probs")
    mean_logs = []
    for j in range(k):
        vals = [max(float(row[j]), eps) for row in permutation_probs]
        mean_logs.append(sum(math.log(v) for v in vals) / len(vals))
    m = max(mean_logs)
    exps = [math.exp(x - m) for x in mean_logs]
    return normalize_distribution(exps, eps=eps)


def arithmetic_mean_distribution(permutation_probs: list[list[float]]) -> list[float]:
    if not permutation_probs:
        raise ValueError("empty permutation_probs")
    k = len(permutation_probs[0])
    return normalize_distribution(
        [sum(float(row[j]) for row in permutation_probs) / len(permutation_probs) for j in range(k)]
    )


def kl_divergence(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    p = normalize_distribution(p, eps)
    q = normalize_distribution(q, eps)
    return sum(pi * math.log(max(pi, eps) / max(qi, eps)) for pi, qi in zip(p, q))


def js_divergence(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    p = normalize_distribution(p, eps)
    q = normalize_distribution(q, eps)
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def permutation_susceptibility(
    permutation_probs: list[list[float]],
    reference_probs: list[float] | None = None,
) -> float:
    """Mean JS divergence from each mapped permutation to the debiased reference."""
    ref = reference_probs or geometric_mean_distribution(permutation_probs)
    return sum(js_divergence(row, ref) for row in permutation_probs) / len(permutation_probs)


def semantic_metrics(probs: list[float], answer: int) -> dict:
    probs = normalize_distribution(probs)
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


def build_prompt_records(
    tokenizer,
    items: list[dict],
    template: str,
    permutation_scheme: str = "cyclic",
) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for item_index, item in enumerate(items):
        choices = list(item["choices"])
        perms = balanced_permutations(
            len(choices), scheme=permutation_scheme, item_id=str(item.get("id", item_index))
        )
        for permutation_index, perm in enumerate(perms):
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
) -> tuple[list[list[float]], list[float], list[int]]:
    """Score K answer labels and retain response-channel diagnostics.

    Returns:
      conditional_probs: p(label | next token is one of allowed labels)
      label_mass: sum full-vocabulary next-token probability over allowed labels
      greedy_is_label: whether the unconstrained greedy next token is an allowed label
    """
    label_seqs = label_token_sequences(tokenizer, k)
    if not all(len(x) == 1 for x in label_seqs):
        raise ValueError("single-token fast path requested for multi-token labels")
    label_ids = torch.tensor([x[0] for x in label_seqs], device=device, dtype=torch.long)
    out: list[list[float]] = []
    masses: list[float] = []
    greedy_is_label: list[int] = []

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
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].float()
            log_z = torch.logsumexp(logits, dim=-1)
            candidate_logits = logits.index_select(-1, label_ids)
            candidate_log_z = torch.logsumexp(candidate_logits, dim=-1)
            probs = torch.softmax(candidate_logits, dim=-1)
            mass = torch.exp(candidate_log_z - log_z)
            greedy = logits.argmax(dim=-1)
            is_label = (greedy[:, None] == label_ids[None, :]).any(dim=-1)

            out.extend(probs.cpu().tolist())
            masses.extend(mass.cpu().tolist())
            greedy_is_label.extend(is_label.int().cpu().tolist())
    finally:
        tokenizer.padding_side = old_padding_side
    return out, masses, greedy_is_label


@torch.inference_mode()
def exact_candidate_sequence_logprob(
    model,
    prompt_ids: list[int],
    candidate_ids: list[int],
    device: torch.device,
) -> float:
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
