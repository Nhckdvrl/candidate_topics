from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class DecodeResult:
    predicted_digits: list[int]
    finalization_step: dict[int, int]
    confidence_at_finalization: dict[int, float]


def _exact_digit_token_ids(tokenizer) -> dict[int, int]:
    out: dict[int, int] = {}
    for d in range(1, 10):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != str(d):
            raise RuntimeError(
                f"tokenizer does not represent digit {d} as one exact token; "
                "the fixed-slot protocol would not be identifiable"
            )
        out[d] = int(ids[0])
    if len(set(out.values())) != 9:
        raise RuntimeError("digit token IDs are not unique")
    return out


def make_prompt(tokenizer, puzzle_text: str) -> list[int]:
    instruction = (
        "Solve this Sudoku. The 81-cell answer grid is already allocated after the prompt. "
        "Given cells are fixed and blank cells are masked. Fill each blank with the correct digit 1-9.\n\n"
        f"Puzzle:\n{puzzle_text}\n"
    )
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return tokenizer.encode(rendered, add_special_tokens=False)
    return tokenizer.encode(instruction, add_special_tokens=True)


def build_sequence(tokenizer, prompt_ids: Sequence[int], puzzle: Sequence[int], mask_id: int):
    """Build prompt + 81 fixed cell slots; givens are clamped, blanks are masks."""
    digit_ids = _exact_digit_token_ids(tokenizer)
    suffix = [mask_id if int(v) == 0 else digit_ids[int(v)] for v in puzzle]
    return list(prompt_ids) + suffix, digit_ids


def _load_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise RuntimeError("torch is required for model execution") from e
    return torch, F


def load_model_and_tokenizer(model_id: str, device: str = "cuda", dtype: str = "bfloat16"):
    torch, _ = _load_torch()
    from transformers import AutoModel, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch_dtype).to(device).eval()
    return model, tokenizer


def decode_fixed_slots(
    model,
    tokenizer,
    puzzle: Sequence[int],
    puzzle_text: str,
    *,
    mask_id: int,
    remasking: str = "low_confidence",
    temperature: float = 0.0,
    seed: int = 0,
) -> DecodeResult:
    """Instrumented LLaDA-style irreversible masked decoding.

    The transfer logic mirrors the public LLaDA generator: predict all masked
    positions, compute confidence of each proposed token, and reveal top-k
    positions. Here k=1 and only the 81 Sudoku cell slots can be revealed.
    Digit logits are grammar-constrained to {1,...,9}; this removes tokenization
    ambiguity while position selection remains driven by model confidence.
    Random remasking is available as a negative control.
    """
    torch, F = _load_torch()
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if remasking not in {"low_confidence", "random"}:
        raise ValueError("remasking must be low_confidence or random")

    prompt_ids = make_prompt(tokenizer, puzzle_text)
    sequence, digit_ids = build_sequence(tokenizer, prompt_ids, puzzle, mask_id)
    slot_start = len(prompt_ids)
    blank_slots = [slot_start + i for i, v in enumerate(puzzle) if int(v) == 0]
    if not blank_slots:
        raise ValueError("puzzle has no blanks")

    x = torch.tensor([sequence], dtype=torch.long, device=model.device)
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    allowed = torch.tensor(list(digit_ids.values()), dtype=torch.long, device=model.device)
    id_to_digit = {tok: d for d, tok in digit_ids.items()}

    finalization_step: dict[int, int] = {}
    finalization_conf: dict[int, float] = {}

    with torch.no_grad():
        for step in range(1, len(blank_slots) + 1):
            mask_index = x.eq(mask_id)
            logits = model(x).logits
            slot_logits = logits[:, blank_slots, :]

            constrained = torch.full_like(slot_logits, float("-inf"))
            constrained[:, :, allowed] = slot_logits[:, :, allowed]
            if temperature > 0:
                u = torch.rand(constrained.shape, generator=generator, device=constrained.device, dtype=torch.float64)
                gumbel = -torch.log(-torch.log(u.clamp_(1e-12, 1 - 1e-12)))
                score_logits = constrained.to(torch.float64) / max(temperature, 1e-8) + gumbel
                proposed_ids = score_logits.argmax(dim=-1)
            else:
                proposed_ids = constrained.argmax(dim=-1)

            probs = F.softmax(constrained.to(torch.float64), dim=-1)
            conf = probs.gather(-1, proposed_ids.unsqueeze(-1)).squeeze(-1)
            active_local = [j for j, abs_pos in enumerate(blank_slots) if bool(mask_index[0, abs_pos])]
            if not active_local:
                break
            if remasking == "random":
                pick_j = active_local[int(torch.randint(len(active_local), (1,), generator=generator, device=model.device).item())]
            else:
                pick_j = max(active_local, key=lambda j: float(conf[0, j]))

            abs_pos = blank_slots[pick_j]
            token_id = int(proposed_ids[0, pick_j].item())
            x[0, abs_pos] = token_id
            cell_i = abs_pos - slot_start
            finalization_step[cell_i] = step
            finalization_conf[cell_i] = float(conf[0, pick_j].item())

    pred = []
    for cell_i in range(81):
        tok = int(x[0, slot_start + cell_i].item())
        if tok not in id_to_digit:
            raise RuntimeError(f"non-digit token survived in cell {cell_i}: {tok}")
        pred.append(id_to_digit[tok])
    return DecodeResult(pred, finalization_step, finalization_conf)
