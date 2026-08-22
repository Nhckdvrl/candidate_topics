from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class DecodeResult:
    predicted_digits: list[int]
    finalization_step: dict[int, int]
    confidence_at_finalization: dict[int, float]
    native_argmax_is_digit: dict[int, bool]

    @property
    def native_digit_argmax_fraction(self) -> float:
        if not self.native_argmax_is_digit:
            return float("nan")
        return sum(bool(v) for v in self.native_argmax_is_digit.values()) / len(self.native_argmax_is_digit)


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
        "Solve this Sudoku. A partially filled 9x9 answer-grid template is appended to your response. "
        "Its separators and given digits are fixed; masked cells are the unknown cells. "
        "Fill every masked cell with exactly one digit 1-9 so that the completed grid is valid.\n\n"
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
    """Build a readable fixed grid template and return exact cell token positions."""
    if len(puzzle) != 81:
        raise ValueError("puzzle must contain 81 cells")
    digit_ids = _exact_digit_token_ids(tokenizer)
    seq = list(prompt_ids)
    seq.extend(tokenizer.encode("\nAnswer grid template:\n", add_special_tokens=False))
    sep = tokenizer.encode(" | ", add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    cell_positions: list[int] = []
    for r in range(9):
        for c in range(9):
            i = r * 9 + c
            cell_positions.append(len(seq))
            seq.append(mask_id if int(puzzle[i]) == 0 else digit_ids[int(puzzle[i])])
            if c != 8:
                seq.extend(sep)
        if r != 8:
            seq.extend(newline)
    return seq, digit_ids, cell_positions


def _load_torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch is required for model execution") from e
    return torch


def load_model_and_tokenizer(model_id: str, device: str = "cuda", dtype: str = "bfloat16"):
    torch = _load_torch()
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
    """Instrumented LLaDA-style irreversible masked decoding for Sudoku cells.

    Content is constrained to digits 1..9 because every mutable slot is a Sudoku
    cell. Scheduling confidence is *not* renormalized over those digits: the
    selected valid digit is scored by its probability under the full vocabulary.
    When the native full-vocabulary argmax is already a digit, this is exactly the
    ordinary LLaDA confidence for that position. That fidelity rate is logged.
    """
    torch = _load_torch()
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if remasking not in {"low_confidence", "random"}:
        raise ValueError("remasking must be low_confidence or random")

    prompt_ids = make_prompt(tokenizer, puzzle_text)
    sequence, digit_ids, cell_positions = build_sequence(tokenizer, prompt_ids, puzzle, mask_id)
    blank_cells = [i for i, v in enumerate(puzzle) if int(v) == 0]
    blank_positions = [cell_positions[i] for i in blank_cells]
    if not blank_positions:
        raise ValueError("puzzle has no blanks")

    x = torch.tensor([sequence], dtype=torch.long, device=model.device)
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    allowed = torch.tensor([digit_ids[d] for d in range(1, 10)], dtype=torch.long, device=model.device)
    id_to_digit = {tok: d for d, tok in digit_ids.items()}

    finalization_step: dict[int, int] = {}
    finalization_conf: dict[int, float] = {}
    native_argmax_is_digit: dict[int, bool] = {}

    with torch.no_grad():
        for step in range(1, len(blank_positions) + 1):
            active = torch.tensor(
                [int(x[0, pos].item()) == mask_id for pos in blank_positions],
                dtype=torch.bool,
                device=model.device,
            )
            if not bool(active.any().item()):
                break

            logits = model(x).logits
            slot_logits = logits[:, blank_positions, :]
            digit_logits = slot_logits[:, :, allowed]

            if temperature > 0:
                noise = torch.rand(digit_logits.shape, generator=generator, device=digit_logits.device, dtype=torch.float64)
                gumbel = -torch.log(-torch.log(noise.clamp_(1e-12, 1 - 1e-12)))
                local_choice = (digit_logits.to(torch.float64) / max(temperature, 1e-8) + gumbel).argmax(dim=-1)
            else:
                local_choice = digit_logits.argmax(dim=-1)
            proposed_ids = allowed[local_choice]

            chosen_logits = slot_logits.gather(-1, proposed_ids.unsqueeze(-1)).squeeze(-1).to(torch.float32)
            log_z = torch.logsumexp(slot_logits.to(torch.float32), dim=-1)
            conf = torch.exp(chosen_logits - log_z)

            native_ids = slot_logits.argmax(dim=-1)
            native_is_digit = (native_ids.unsqueeze(-1) == allowed.view(1, 1, -1)).any(dim=-1)

            if remasking == "random":
                active_indices = torch.nonzero(active, as_tuple=False).flatten()
                pick = torch.randint(active_indices.numel(), (1,), generator=generator, device=model.device)
                pick_j = int(active_indices[pick].item())
            else:
                scores = conf[0].masked_fill(~active, float("-inf"))
                pick_j = int(scores.argmax().item())

            abs_pos = blank_positions[pick_j]
            cell_i = blank_cells[pick_j]
            token_id = int(proposed_ids[0, pick_j].item())
            x[0, abs_pos] = token_id
            finalization_step[cell_i] = step
            finalization_conf[cell_i] = float(conf[0, pick_j].item())
            native_argmax_is_digit[cell_i] = bool(native_is_digit[0, pick_j].item())

    pred: list[int] = []
    for cell_i, pos in enumerate(cell_positions):
        tok = int(x[0, pos].item())
        if tok not in id_to_digit:
            raise RuntimeError(f"non-digit token survived in cell {cell_i}: {tok}")
        pred.append(id_to_digit[tok])
    return DecodeResult(pred, finalization_step, finalization_conf, native_argmax_is_digit)
