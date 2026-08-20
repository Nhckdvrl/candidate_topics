from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from generate_fates import (
    _extract_answers,
    generate_one as generate_one_llada,
    masked_baselines,
    pool_generation,
    chosen_token_probability,
)
from gsm8k_utils import gold_number, numerically_equal

MODEL_CFGS = {
    "llada": {
        "name": "GSAI-ML/LLaDA-8B-Instruct",
        "mask_id": 126336,
        "hidden_indices": [25, 28],
    },
    "dream": {
        "name": "Dream-org/Dream-v0-Instruct-7B",
        "mask_id": 151666,
        "hidden_indices": [22, 25],
    },
}

DEFAULT_CAPTURE_STEPS = [0, 4, 16, 63]


def select_indices(total: int, offset: int, num_examples: int) -> np.ndarray:
    if offset < 0 or offset >= total:
        raise ValueError(f"offset {offset} out of range for dataset of size {total}")
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    end = min(total, offset + num_examples)
    return np.arange(offset, end, dtype=np.int64)


def load_stage2_dataset(dataset_key: str):
    from datasets import load_dataset

    if dataset_key == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    elif dataset_key == "gsm1k":
        ds = load_dataset("ScaleAI/gsm1k", split="test")
    else:
        raise ValueError(f"unsupported dataset: {dataset_key}")
    return ds


def question_and_gold(dataset_key: str, instance: dict) -> tuple[str, str]:
    if dataset_key == "gsm8k":
        return instance["question"], gold_number(instance["answer"])
    if dataset_key == "gsm1k":
        return instance["question"], str(instance["answer"]).strip().replace(",", "")
    raise ValueError(dataset_key)


def build_dream_prompt(tokenizer, question: str, device: torch.device):
    system = (
        "You are a math expert. You will be given a question to solve. "
        "Solve it step by step. Wrap the final answer in a \\boxed{}. \n"
        "Respond in the following format:\n"
        "<reasoning>\nYour reasoning here\n</reasoning>\n"
        "<answer>\n\\boxed{...}\n</answer>"
    )
    messages = [{"role": "user", "content": system + "\n\n" + question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    return inputs.input_ids.to(device), inputs.attention_mask.to(device)


def dream_transfer_count(num_mask: int, step: int, steps: int, eps: float = 1e-3) -> int:
    if num_mask <= 0:
        return 0
    if step == steps - 1:
        return num_mask
    t = 1.0 + (eps - 1.0) * (step / steps)
    s = 1.0 + (eps - 1.0) * ((step + 1) / steps)
    return min(num_mask, int(num_mask * (1.0 - s / t)))


def _prepare_dream_attention(attention_mask: torch.Tensor, full_length: int):
    import torch.nn.functional as F

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask = F.pad(
            attention_mask,
            (0, full_length - attention_mask.shape[1]),
            value=1.0,
        )
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        full_attention = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        return full_attention, tok_idx
    return "full", None


def generate_one_dream(
    model,
    tokenizer,
    question: str,
    gold: str,
    args: argparse.Namespace,
    device: torch.device,
):
    """Deterministic Dream maskgit_plus trajectory.

    This follows Dream's official generation_utils.py: shifted logits, global
    linear time schedule, confidence-ranked transfer. Temperature is fixed to 0
    so future fate is not contaminated by future token-sampling randomness.
    """
    if args.temperature != 0:
        raise ValueError("Stage-2 Dream confirmation requires temperature=0")

    prompt, prompt_attention = build_dream_prompt(tokenizer, question, device)
    gen_start = prompt.shape[1]
    mask_id = MODEL_CFGS["dream"]["mask_id"]
    x = torch.full(
        (1, gen_start + args.gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :gen_start] = prompt
    dream_attention, tok_idx = _prepare_dream_attention(prompt_attention, x.shape[1])

    capture_set = set(args.capture_steps)
    capture_map = {s: i for i, s in enumerate(args.capture_steps)}
    n_saved = len(args.capture_steps)

    hidden = None
    entropy = np.full(n_saved, np.nan, dtype=np.float32)
    selected_prob = np.full(n_saved, np.nan, dtype=np.float32)
    clean_maxprob = np.full(n_saved, np.nan, dtype=np.float32)
    frac_unmasked = np.full(n_saved, np.nan, dtype=np.float32)
    if not args.surface_only:
        hidden = np.empty(
            (
                n_saved,
                len(args.hidden_indices),
                args.n_regions,
                int(model.config.hidden_size),
            ),
            dtype=np.float16,
        )

    correct_strict = np.zeros(args.steps, dtype=np.bool_)
    observed_strict = np.zeros(args.steps, dtype=np.bool_)
    correct_fallback = np.zeros(args.steps, dtype=np.bool_)
    observed_fallback = np.zeros(args.steps, dtype=np.bool_)
    answer_all = np.full(args.steps, "", dtype="<U64")

    with torch.inference_mode():
        for step in range(args.steps):
            need_capture = step in capture_set
            need_hidden = need_capture and not args.surface_only
            out = model(
                x,
                dream_attention,
                tok_idx,
                output_hidden_states=need_hidden,
            )
            # Dream predicts token i from the previous shifted position.
            logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
            raw_pred = torch.argmax(logits, dim=-1)
            mask_index = x == mask_id
            x0 = torch.where(mask_index, raw_pred, x)

            current_text = tokenizer.batch_decode(
                x0[:, gen_start:], skip_special_tokens=True
            )[0]
            strict_ans, fallback_ans = _extract_answers(current_text, "midtruth")
            observed_strict[step] = strict_ans is not None
            observed_fallback[step] = fallback_ans is not None
            correct_strict[step] = strict_ans is not None and numerically_equal(strict_ans, gold)
            correct_fallback[step] = fallback_ans is not None and numerically_equal(fallback_ans, gold)
            answer_all[step] = "" if fallback_ans is None else fallback_ans[:63]

            chosen_p = chosen_token_probability(logits, raw_pred)
            if need_capture and not args.surface_only:
                si = capture_map[step]
                if not hasattr(out, "hidden_states") or not out.hidden_states:
                    raise RuntimeError("Dream did not return hidden states")
                for li, hs_idx in enumerate(args.hidden_indices):
                    if hs_idx >= len(out.hidden_states):
                        raise IndexError(
                            f"hidden index {hs_idx} >= {len(out.hidden_states)} hidden states"
                        )
                    hidden[si, li] = pool_generation(
                        out.hidden_states[hs_idx],
                        gen_start,
                        args.gen_length,
                        args.n_regions,
                    )
                e, sp, mp = masked_baselines(
                    logits[:, gen_start:],
                    mask_index[:, gen_start:],
                    raw_pred[:, gen_start:],
                )
                entropy[si] = e
                selected_prob[si] = sp
                clean_maxprob[si] = mp
                frac_unmasked[si] = 1.0 - float(
                    mask_index[:, gen_start:].sum().item()
                ) / args.gen_length

            n_mask = int(mask_index[:, gen_start:].sum().item())
            n_transfer = dream_transfer_count(n_mask, step, args.steps)
            if n_transfer > 0:
                confidence = torch.where(
                    mask_index,
                    chosen_p,
                    torch.tensor(-np.inf, device=device, dtype=chosen_p.dtype),
                )
                confidence[:, :gen_start] = -np.inf
                _, indices = torch.topk(confidence[0], k=n_transfer)
                x[0, indices] = raw_pred[0, indices]

    final_text = tokenizer.batch_decode(x[:, gen_start:], skip_special_tokens=True)[0]
    final_strict, final_fallback = _extract_answers(final_text, "midtruth")
    return {
        "correct_strict": correct_strict,
        "observed_strict": observed_strict,
        "correct_fallback": correct_fallback,
        "observed_fallback": observed_fallback,
        "answer_all": answer_all,
        "entropy": entropy,
        "selected_prob": selected_prob,
        "clean_maxprob": clean_maxprob,
        "frac_unmasked": frac_unmasked,
        "prompt_tokens": int(gen_start),
        "final_text": final_text,
        "final_strict": final_strict,
        "final_fallback": final_fallback,
        "hidden": hidden,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 locked confirmation trajectory generator")
    p.add_argument("--model-family", choices=sorted(MODEL_CFGS), default="llada")
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", choices=["gsm8k", "gsm1k"], required=True)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--num-examples", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--capture-steps", type=int, nargs="+", default=DEFAULT_CAPTURE_STEPS)
    p.add_argument("--hidden-indices", type=int, nargs="+", default=None)
    p.add_argument("--n-regions", type=int, default=1, choices=[1, 2, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--surface-only", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard index")
    if args.steps != 64 or args.gen_length != 128:
        raise ValueError("Stage-2 confirmation freezes the 64-step / 128-token G0 geometry")
    if args.temperature != 0:
        raise ValueError("Stage-2 primary confirmation freezes temperature=0")
    if sorted(set(args.capture_steps)) != list(args.capture_steps):
        raise ValueError("capture steps must be sorted and unique")
    if 0 not in args.capture_steps or 4 not in args.capture_steps or 16 not in args.capture_steps:
        raise ValueError("locked confirmation requires capture steps 0, 4, and 16")


def main() -> None:
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    args = parse_args()
    _validate_args(args)
    cfg = MODEL_CFGS[args.model_family]
    args.model = args.model or cfg["name"]
    args.hidden_indices = args.hidden_indices or list(cfg["hidden_indices"])
    args.prompt_style = "midtruth"

    ds = load_stage2_dataset(args.dataset)
    all_ids = select_indices(len(ds), args.offset, args.num_examples)
    shard_ids = np.array_split(all_ids, args.num_shards)[args.shard_index]
    expected_total = len(all_ids)

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
    for problem_id in tqdm(shard_ids.tolist(), desc=f"stage2 shard {args.shard_index}/{args.num_shards}"):
        inst = ds[int(problem_id)]
        question, gold = question_and_gold(args.dataset, inst)
        torch.manual_seed(args.seed)
        if args.model_family == "llada":
            llada_args = SimpleNamespace(**vars(args))
            r = generate_one_llada(model, tokenizer, question, gold, llada_args, device)
        else:
            r = generate_one_dream(model, tokenizer, question, gold, args, device)
        records.append((problem_id, gold, r))
        final_texts.append(
            {
                "problem_id": int(problem_id),
                "gold": gold,
                "strict_answer": r["final_strict"],
                "fallback_answer": r["final_fallback"],
                "final_text": r["final_text"],
            }
        )

    if not records:
        raise RuntimeError("empty shard")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "stage": "G1_confirmation",
        "model_family": args.model_family,
        "model": args.model,
        "dataset": args.dataset,
        "dataset_offset": int(args.offset),
        "num_examples": int(expected_total),
        "num_shards": int(args.num_shards),
        "steps": int(args.steps),
        "gen_length": int(args.gen_length),
        "block_length": int(args.block_length),
        "temperature": float(args.temperature),
        "prompt_style": "midtruth",
        "dream_alg": "maskgit_plus" if args.model_family == "dream" else None,
        "seed": int(args.seed),
        "surface_only": bool(args.surface_only),
        "n_regions": int(args.n_regions),
    }
    arrays = {
        "problem_id": np.array([r[0] for r in records], dtype=np.int32),
        "gold": np.array([r[1] for r in records], dtype="<U64"),
        "capture_steps": np.array(args.capture_steps, dtype=np.int16),
        "hidden_indices": np.array(args.hidden_indices, dtype=np.int16),
        "correct_strict": np.stack([r[2]["correct_strict"] for r in records]),
        "observed_strict": np.stack([r[2]["observed_strict"] for r in records]),
        "correct_fallback": np.stack([r[2]["correct_fallback"] for r in records]),
        "observed_fallback": np.stack([r[2]["observed_fallback"] for r in records]),
        "answer_all": np.stack([r[2]["answer_all"] for r in records]),
        "entropy": np.stack([r[2]["entropy"] for r in records]),
        "selected_prob": np.stack([r[2]["selected_prob"] for r in records]),
        "clean_maxprob": np.stack([r[2]["clean_maxprob"] for r in records]),
        "frac_unmasked": np.stack([r[2]["frac_unmasked"] for r in records]),
        "prompt_tokens": np.array([r[2]["prompt_tokens"] for r in records], dtype=np.int16),
        "metadata_json": np.array(json.dumps(meta, sort_keys=True)),
    }
    if not args.surface_only:
        arrays["hidden"] = np.stack([r[2]["hidden"] for r in records])

    np.savez_compressed(
        out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz",
        **arrays,
    )
    (out_dir / f"shard_{args.shard_index:02d}_final_texts.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in final_texts) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
