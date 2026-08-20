from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from gsm8k_utils import extract_boxed_number, extract_number, extract_number_strict, gold_number, numerically_equal

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336
DEFAULT_STEPS = 128
DEFAULT_GEN_LENGTH = 512
DEFAULT_BLOCK_LENGTH = 32
# These are hidden_states tuple indices, matching the public dlm-probing implementation/paper layer numbering.
DEFAULT_HIDDEN_INDICES = [24, 25, 28]
DEFAULT_CAPTURE_STEPS = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 120, 124, 127]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate complete LLaDA correctness trajectories and sparse hidden-state captures.")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--num-examples", type=int, default=1000)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--gen-length", type=int, default=DEFAULT_GEN_LENGTH)
    p.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--hidden-indices", type=int, nargs="+", default=DEFAULT_HIDDEN_INDICES)
    p.add_argument("--capture-steps", type=int, nargs="+", default=DEFAULT_CAPTURE_STEPS)
    p.add_argument("--n-regions", type=int, default=1, choices=[1, 2, 4], help="G0 default=1; use 4 for closer reproduction of the probing paper.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-style", choices=["probing", "midtruth"], default="probing")
    p.add_argument("--output-dir", default="artifacts/raw")
    return p.parse_args()


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # Matches the public dlm-probing implementation rather than inventing a new sampler.
    if temperature == 0:
        return logits
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return logits.to(torch.float64) + noise * temperature


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    return torch.where(
        torch.arange(steps, device=mask_index.device).unsqueeze(0) < remainder,
        base + 1,
        base,
    )


def build_prompt(tokenizer, question: str, device: torch.device, style: str) -> torch.Tensor:
    if style == "probing":
        system = "Solve the math problem step by step. End your answer with #### followed by the final numeric answer."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Matches dLLM-MidTruth/eval/gsm8k.py: instruction inside a user message + <reasoning> prefix.
        system = ("You are a math expert. You will be given a question to solve. Solve it step by step. "
                  "Wrap the final answer in a \\boxed{}.\nRespond in the following format:\n"
                  "<reasoning>\nYour reasoning here\n</reasoning>\n<answer>\n\\boxed{...}\n</answer>")
        messages = [{"role": "user", "content": system + "\n\n" + question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<reasoning>"
    ids = tokenizer(text)["input_ids"]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def pool_generation(hidden: torch.Tensor, gen_start: int, gen_length: int, n_regions: int) -> np.ndarray:
    h = hidden[0, gen_start : gen_start + gen_length].float()
    region_size = gen_length // n_regions
    pooled = []
    for r in range(n_regions):
        a = r * region_size
        b = gen_length if r == n_regions - 1 else (r + 1) * region_size
        pooled.append(h[a:b].mean(dim=0).cpu().numpy())
    return np.stack(pooled).astype(np.float16)


def masked_baselines(logits: torch.Tensor, mask_index: torch.Tensor, chosen_x0: torch.Tensor) -> tuple[float, float]:
    idx = mask_index
    if not bool(idx.any()):
        return 0.0, 1.0
    masked_logits = logits[idx].float()
    logp = F.log_softmax(masked_logits, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1).mean().item()
    chosen = chosen_x0[idx].unsqueeze(-1)
    max_like = p.gather(-1, chosen).mean().item()
    return float(entropy), float(max_like)


def generate_one(model, tokenizer, question: str, gold: str, args, device: torch.device):
    prompt = build_prompt(tokenizer, question, device, args.prompt_style)
    gen_start = prompt.shape[1]
    x = torch.full((1, gen_start + args.gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :gen_start] = prompt

    if args.gen_length % args.block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = args.gen_length // args.block_length
    if args.steps % num_blocks != 0:
        raise ValueError("steps must be divisible by number of blocks for this reproduction")
    steps_per_block = args.steps // num_blocks
    capture_set = set(args.capture_steps)
    capture_map = {s: i for i, s in enumerate(args.capture_steps)}

    n_saved = len(args.capture_steps)
    hidden_dim = int(model.config.hidden_size)
    hidden = np.empty((n_saved, len(args.hidden_indices), args.n_regions, hidden_dim), dtype=np.float16)
    entropy = np.full(n_saved, np.nan, dtype=np.float32)
    maxprob = np.full(n_saved, np.nan, dtype=np.float32)
    frac_unmasked = np.full(n_saved, np.nan, dtype=np.float32)
    correct_all = np.zeros(args.steps, dtype=np.bool_)  # strict answer marker: primary trajectory label
    correct_all_fallback = np.zeros(args.steps, dtype=np.bool_)  # robustness: public probe fallback parser
    answer_all = np.full(args.steps, "", dtype="<U64")

    global_step = 0
    with torch.inference_mode():
        for num_block in range(num_blocks):
            block_start = gen_start + num_block * args.block_length
            block_end = gen_start + (num_block + 1) * args.block_length
            block_mask = x[:, block_start:block_end] == MASK_ID
            transfers = get_num_transfer_tokens(block_mask, steps_per_block)

            for local_step in range(steps_per_block):
                need_hidden = global_step in capture_set
                out = model(x, output_hidden_states=need_hidden)
                logits = out.logits
                noisy = add_gumbel_noise(logits, args.temperature)
                raw_pred = torch.argmax(noisy, dim=-1)
                mask_index = x == MASK_ID
                # x0 is a complete current prediction: keep committed tokens, fill all remaining masks.
                x0 = torch.where(mask_index, raw_pred, x)

                current_text = tokenizer.batch_decode(x0[:, gen_start:], skip_special_tokens=True)[0]
                if args.prompt_style == "probing":
                    strict_ans = extract_number_strict(current_text)
                else:
                    strict_ans = extract_boxed_number(current_text)
                fallback_ans = strict_ans if strict_ans is not None else extract_number(current_text)
                answer_all[global_step] = "" if fallback_ans is None else fallback_ans[:63]
                correct_all[global_step] = numerically_equal(strict_ans, gold)
                correct_all_fallback[global_step] = numerically_equal(fallback_ans, gold)

                p = F.softmax(logits.to(torch.float64), dim=-1)
                chosen_p = torch.gather(p, dim=-1, index=raw_pred.unsqueeze(-1)).squeeze(-1)

                if need_hidden:
                    si = capture_map[global_step]
                    for li, hs_idx in enumerate(args.hidden_indices):
                        if hs_idx >= len(out.hidden_states):
                            raise IndexError(f"hidden index {hs_idx} >= {len(out.hidden_states)} hidden states")
                        hidden[si, li] = pool_generation(out.hidden_states[hs_idx], gen_start, args.gen_length, args.n_regions)
                    e, mp = masked_baselines(logits[:, gen_start:], mask_index[:, gen_start:], raw_pred[:, gen_start:])
                    entropy[si] = e
                    maxprob[si] = mp
                    frac_unmasked[si] = 1.0 - float(mask_index[:, gen_start:].sum().item()) / args.gen_length

                n_transfer = int(transfers[0, local_step].item())
                if n_transfer > 0:
                    confidence = torch.where(mask_index, chosen_p, torch.tensor(-np.inf, device=device, dtype=chosen_p.dtype))
                    confidence[:, :block_start] = -np.inf
                    confidence[:, block_end:] = -np.inf
                    available = int(mask_index[0, block_start:block_end].sum().item())
                    k = min(n_transfer, available)
                    if k > 0:
                        _, indices = torch.topk(confidence[0], k=k)
                        x[0, indices] = raw_pred[0, indices]
                global_step += 1

    final_text = tokenizer.batch_decode(x[:, gen_start:], skip_special_tokens=True)[0]
    final_correct = numerically_equal(extract_number(final_text), gold)
    return {
        "correct_all": correct_all,
        "correct_all_fallback": correct_all_fallback,
        "answer_all": answer_all,
        "hidden": hidden,
        "entropy": entropy,
        "maxprob": maxprob,
        "frac_unmasked": frac_unmasked,
        "final_correct": final_correct,
        "final_text": final_text,
    }


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard index")
    if sorted(set(args.capture_steps)) != sorted(args.capture_steps):
        raise ValueError("capture steps must be unique and sorted")
    if max(args.capture_steps) >= args.steps:
        raise ValueError("capture step >= total steps")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    total = min(args.num_examples, len(ds))
    ids = np.array_split(np.arange(total), args.num_shards)[args.shard_index]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    records = []
    final_texts = []
    for problem_id in tqdm(ids.tolist(), desc=f"shard {args.shard_index}/{args.num_shards}"):
        inst = ds[int(problem_id)]
        gold = gold_number(inst["answer"])
        # Match the public probing code: reset generation seed per instance.
        torch.manual_seed(args.seed)
        r = generate_one(model, tokenizer, inst["question"], gold, args, device)
        records.append((problem_id, gold, r))
        final_texts.append({"problem_id": int(problem_id), "gold": gold, "final_correct": bool(r["final_correct"]), "final_text": r["final_text"]})

    if not records:
        raise RuntimeError("empty shard")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz"
    np.savez_compressed(
        path,
        problem_id=np.array([r[0] for r in records], dtype=np.int32),
        gold=np.array([r[1] for r in records], dtype="<U64"),
        capture_steps=np.array(args.capture_steps, dtype=np.int16),
        hidden_indices=np.array(args.hidden_indices, dtype=np.int16),
        correct_all=np.stack([r[2]["correct_all"] for r in records]),
        correct_all_fallback=np.stack([r[2]["correct_all_fallback"] for r in records]),
        answer_all=np.stack([r[2]["answer_all"] for r in records]),
        hidden=np.stack([r[2]["hidden"] for r in records]),
        entropy=np.stack([r[2]["entropy"] for r in records]),
        maxprob=np.stack([r[2]["maxprob"] for r in records]),
        frac_unmasked=np.stack([r[2]["frac_unmasked"] for r in records]),
        final_correct=np.array([r[2]["final_correct"] for r in records], dtype=np.bool_),
    )
    (out_dir / f"shard_{args.shard_index:02d}_final_texts.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in final_texts) + "\n", encoding="utf-8"
    )
    print(f"saved {path} ({len(records)} problems)")


if __name__ == "__main__":
    main()
