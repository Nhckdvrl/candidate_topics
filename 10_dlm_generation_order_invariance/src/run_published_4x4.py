from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from instrumented_llada import load_model_and_tokenizer


SYSTEM_PROMPT = """Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where ' ' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid. Never leave it as ' '.

Respond in this exact format:
<answer>
[First raw of 4-character solution]
[Second raw of 4-character solution]
[Third raw of 4-character solution]
[Firth raw of 4-character solution]
</answer>"""


def published_prompt(tokenizer, puzzle: str) -> str:
    entered = ("\n" + puzzle[:4] + "\n" + puzzle[4:8] + "\n" + puzzle[8:12] + "\n" + puzzle[12:] + "\n").replace("0", " ")
    masked = entered.replace(" ", "<|mdm_mask|>")
    formatted_prompt = f"{SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {entered}\n"
    formatted_answer = f"\n<answer>{masked}</answer><|eot_id|>"
    # This tuple-as-content behavior is intentional: it is what the published
    # UPO evaluation code passes to apply_chat_template.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": (formatted_prompt, formatted_answer)}],
        add_generation_prompt=True,
        tokenize=False,
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decode_one(model, tokenizer, puzzle: str, solution: str, cfg: dict) -> dict:
    text = published_prompt(tokenizer, puzzle)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    x = encoded.input_ids.to(model.device)
    mask_positions = torch.nonzero(x[0].eq(cfg["mask_id"]), as_tuple=False).flatten()
    blank_indices = [i for i, ch in enumerate(puzzle) if ch == "0"]
    if len(mask_positions) != len(blank_indices):
        raise RuntimeError(f"published template mask mismatch: masks={len(mask_positions)} blanks={len(blank_indices)}")

    native_digit_argmax = []
    finalization_step = {}
    with torch.no_grad():
        for step in range(1, len(mask_positions) + 1):
            active = x[0, mask_positions].eq(cfg["mask_id"])
            logits = model(x).logits[0, mask_positions, :].float()
            log_probs = logits.log_softmax(dim=-1)
            values, token_ids = log_probs.max(dim=-1)
            pick = int(values.masked_fill(~active, float("-inf")).argmax().item())
            abs_pos = int(mask_positions[pick].item())
            x[0, abs_pos] = token_ids[pick]
            cell = blank_indices[pick]
            finalization_step[str(cell)] = step
            native_digit_argmax.append(int(token_ids[pick].item()) in {
                int(tokenizer.encode(str(d), add_special_tokens=False)[0]) for d in range(1, 5)
            })

    predicted_tokens = [int(x[0, int(pos)].item()) for pos in mask_positions]
    predicted_cells = [tokenizer.decode([tok], skip_special_tokens=False) for tok in predicted_tokens]
    correct = sum(predicted_cells[j] == solution[cell] for j, cell in enumerate(blank_indices))
    return {
        "puzzle": puzzle,
        "solution": solution,
        "blank_indices": blank_indices,
        "predicted_blank_tokens": predicted_cells,
        "blank_cell_correct": correct,
        "blank_cell_total": len(blank_indices),
        "exact_solution": correct == len(blank_indices),
        "valid_digit_predictions": all(x in {"1", "2", "3", "4"} for x in predicted_cells),
        "finalization_step": finalization_step,
        "metadata": {
            "protocol_version": cfg["protocol_version"],
            "native_scheduler_pick_same_fraction": 1.0,
            "native_digit_argmax_fraction": sum(native_digit_argmax) / len(native_digit_argmax),
            "template_tuple_content": True,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="LOCKED_CONFIG_V3.json")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results/v3_published_4x4.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num-shards>=1 and 0<=shard-index<num-shards")

    cfg = json.loads(Path(args.config).read_text())
    dataset = Path(args.dataset)
    if sha256(dataset) != cfg["dataset_sha256"]:
        raise RuntimeError("dataset SHA256 does not match LOCKED_CONFIG_V3.json")
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")
    rows = list(csv.DictReader(dataset.open(newline="")))
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.num_shards]

    model, tokenizer = load_model_and_tokenizer(cfg["model_id"], device=args.device, dtype=cfg["dtype"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            result = decode_one(model, tokenizer, row["Puzzle"], row["Solution"], cfg)
            result["row_index"] = i
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(rows)}] blank_acc={result['blank_cell_correct']}/{result['blank_cell_total']} exact={result['exact_solution']}", flush=True)


if __name__ == "__main__":
    main()
