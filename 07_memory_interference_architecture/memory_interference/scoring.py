from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F


@dataclass
class CandidateScore:
    candidate: str
    sum_logprob: float
    mean_logprob: float
    token_count: int
    boundary_shift: int


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError("model output has no logits")


def score_candidates(
    model,
    tokenizer,
    prompt: str,
    candidates: Iterable[str],
    *,
    device: torch.device,
    batch_size: int = 16,
    max_boundary_shift: int = 0,
) -> List[CandidateScore]:
    """Teacher-force candidate continuations and return token-normalized log probabilities.

    Full prompt+candidate tokenization is used so SentencePiece/BPE boundary behavior is
    respected. Prompts end in a newline; candidates whose boundary shift exceeds the configured
    maximum are rejected rather than silently compared under inconsistent prefixes. The
    frozen pilot and confirmation configs require a zero-token boundary shift.
    """
    candidates = list(candidates)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    encoded = []
    for candidate in candidates:
        full_ids = tokenizer(prompt + candidate, add_special_tokens=True)["input_ids"]
        lcp = longest_common_prefix(prompt_ids, full_ids)
        shift = len(prompt_ids) - lcp
        if shift > max_boundary_shift:
            raise ValueError(
                f"tokenization boundary changed by {shift} tokens for candidate={candidate!r}; "
                "change the prompt delimiter or audit tokenizer compatibility"
            )
        if lcp >= len(full_ids):
            raise ValueError(f"candidate produced no continuation tokens: {candidate!r}")
        encoded.append((candidate, full_ids, lcp, shift))

    scores: List[CandidateScore] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        max_len = max(len(ids) for _, ids, _, _ in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for i, (_, ids, _, _) in enumerate(chunk):
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            input_ids[i, : len(ids)] = ids_t
            attention_mask[i, : len(ids)] = 1

        with torch.inference_mode():
            try:
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            except TypeError:
                output = model(input_ids)
            logits = _extract_logits(output)
            log_probs = F.log_softmax(logits.float(), dim=-1)

        for i, (candidate, ids, boundary, shift) in enumerate(chunk):
            token_lps = []
            for p in range(boundary, len(ids)):
                if p == 0:
                    continue
                token_lps.append(log_probs[i, p - 1, ids[p]])
            if not token_lps:
                raise ValueError(f"no scoreable tokens for candidate={candidate!r}")
            stacked = torch.stack(token_lps)
            scores.append(
                CandidateScore(
                    candidate=candidate,
                    sum_logprob=float(stacked.sum().item()),
                    mean_logprob=float(stacked.mean().item()),
                    token_count=len(token_lps),
                    boundary_shift=shift,
                )
            )
    return scores
