#!/usr/bin/env python3
"""Topic 14: temporal persistence of a power-law head on 4-hop S5 state tracking.

Balanced slow/fast schedules have the same rank histogram in every block, the same
skill×rank occupancy over a cycle, and the exact same multiset of minibatch blocks.
Only the temporal ordering of those blocks changes.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SYMBOLS = 5
N_SKILLS = math.factorial(N_SYMBOLS)  # |S5| = 120
HOPS = 4
SEQ_LEN = N_SYMBOLS * HOPS


@dataclass(frozen=True)
class Profile:
    cycles: int
    block_steps: int
    eval_every: int
    eval_examples: int
    bin_eval_examples: int


PROFILES = {
    "smoke": Profile(1, 1, 60, 512, 128),          # 120 steps; engineering only
    "pilot": Profile(1, 100, 500, 4096, 512),      # 12,000 steps
    "confirm": Profile(2, 400, 1000, 8192, 1024),  # 96,000 steps
    "full": Profile(2, 833, 2000, 16384, 2048),    # 199,920 ≈ seed 200k
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def all_permutations() -> np.ndarray:
    return np.asarray(list(itertools.permutations(range(N_SYMBOLS))), dtype=np.int64)


def compose_numpy(perms: np.ndarray) -> np.ndarray:
    """Compose p0∘p1∘... where (sigma∘pi)[i] = sigma[pi[i]]."""
    out = perms[:, 0].copy()
    for h in range(1, perms.shape[1]):
        out = np.take_along_axis(out, perms[:, h], axis=1)
    return out


def zipf_prob(alpha: float) -> np.ndarray:
    ranks = np.arange(1, N_SKILLS + 1, dtype=np.float64)
    p = ranks ** (-alpha)
    return p / p.sum()


def largest_remainder_counts(prob: np.ndarray, total: int) -> np.ndarray:
    raw = prob * total
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    assert counts.sum() == total
    return counts


def make_base_order(mapping_seed: int) -> np.ndarray:
    return np.random.default_rng(mapping_seed).permutation(N_SKILLS).astype(np.int64)


def make_shift_schedule(mode: str, cycles: int, schedule_seed: int) -> np.ndarray:
    """Each balanced cycle contains every cyclic rank→skill shift exactly once."""
    if mode in {"uniform", "static"}:
        return np.zeros(cycles * N_SKILLS, dtype=np.int64)
    if mode == "balanced_slow":
        return np.tile(np.arange(N_SKILLS, dtype=np.int64), cycles)
    if mode == "balanced_fast":
        rng = np.random.default_rng(schedule_seed)
        chunks = []
        for _ in range(cycles):
            best, best_score = None, float("inf")
            for _ in range(256):
                cand = rng.permutation(N_SKILLS).astype(np.int64)
                score = abs(float(np.corrcoef(cand[:-1], cand[1:])[0, 1]))
                if score < best_score:
                    best, best_score = cand, score
                if score < 0.03:
                    break
            chunks.append(best)
        return np.concatenate(chunks)
    raise ValueError(mode)


def rank_to_skill_for_shift(base_order: np.ndarray, shift: int) -> np.ndarray:
    return base_order[(np.arange(N_SKILLS) + int(shift)) % N_SKILLS]


def block_stream_seed(condition: str, block_idx: int, shift: int, seed: int, stream_seed: int) -> int:
    """Balanced block RNG is keyed by (cycle, shift), making block multisets exact."""
    if condition.startswith("balanced_"):
        cycle = block_idx // N_SKILLS
        key = cycle * N_SKILLS + int(shift)
    else:
        key = block_idx
    return int(stream_seed + seed * 1_000_003 + key)


def schedule_audit(condition: str, shifts: np.ndarray, base_order: np.ndarray, alpha: float,
                   head_fraction: float = 0.2) -> dict:
    p = np.full(N_SKILLS, 1 / N_SKILLS) if condition == "uniform" else zipf_prob(alpha)
    weights = np.empty((len(shifts), N_SKILLS))
    ranks = np.empty((len(shifts), N_SKILLS), dtype=np.int64)
    occupancy = np.zeros((N_SKILLS, N_SKILLS), dtype=np.int64)

    for b, shift in enumerate(shifts):
        r2s = rank_to_skill_for_shift(base_order, int(shift))
        s2r = np.empty(N_SKILLS, dtype=np.int64)
        s2r[r2s] = np.arange(N_SKILLS)
        ranks[b] = s2r
        weights[b] = p[s2r]
        occupancy[r2s, np.arange(N_SKILLS)] += 1

    if condition == "uniform" or len(shifts) <= 1:
        lag1 = None
    else:
        x = np.log(weights[:-1].reshape(-1) + 1e-30)
        y = np.log(weights[1:].reshape(-1) + 1e-30)
        lag1 = float(np.corrcoef(x, y)[0, 1])

    head_k = max(1, round(N_SKILLS * head_fraction))
    is_head = ranks < head_k
    all_runs, max_runs = [], []
    for s in range(N_SKILLS):
        runs, cur = [], 0
        for flag in is_head[:, s]:
            if flag:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        all_runs.extend(runs)
        max_runs.append(max(runs) if runs else 0)

    return {
        "condition": condition,
        "n_blocks": int(len(shifts)),
        "alpha": float(alpha),
        "lag1_log_weight_corr": lag1,
        "head_fraction": float(head_fraction),
        "mean_head_run_length_blocks": float(np.mean(all_runs)) if all_runs else 0.0,
        "mean_max_head_run_blocks_per_skill": float(np.mean(max_runs)),
        "max_head_run_blocks": int(max(max_runs) if max_runs else 0),
        "occupancy_min": int(occupancy.min()),
        "occupancy_max": int(occupancy.max()),
        "occupancy_is_exactly_balanced": (
            bool(occupancy.min() == occupancy.max()) if condition.startswith("balanced_") else None
        ),
    }


def exact_rank_block(prob: np.ndarray, block_steps: int, batch_size: int, hops: int,
                     block_seed: int) -> np.ndarray:
    """Exact rank histogram per block, with randomized within-block order."""
    total = block_steps * batch_size * hops
    counts = largest_remainder_counts(prob, total)
    ranks = np.repeat(np.arange(N_SKILLS, dtype=np.int16), counts)
    rng = np.random.default_rng(block_seed)
    rng.shuffle(ranks)
    return ranks.reshape(block_steps, batch_size, hops)


class StateTrackingTransformer(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8, ff_mult=4, dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(N_SYMBOLS, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_SYMBOLS)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens) + self.pos_emb[:, : tokens.shape[1]]
        x = self.encoder(x)
        return self.head(self.ln(x[:, -N_SYMBOLS:]))


def make_batch_from_skills(skill_ids: np.ndarray, perm_table: np.ndarray):
    perms = perm_table[skill_ids]
    return perms.reshape(perms.shape[0], -1), compose_numpy(perms)


def fixed_eval(n: int, seed: int, perm_table: np.ndarray, pool=None):
    rng = np.random.default_rng(seed)
    pool = np.arange(N_SKILLS) if pool is None else np.asarray(pool)
    skills = rng.choice(pool, size=(n, HOPS), replace=True)
    x, y = make_batch_from_skills(skills, perm_table)
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


@torch.inference_mode()
def evaluate(model, x_cpu, y_cpu, device, batch_size):
    model.eval()
    loss_sum = token_ok = exact_ok = n_tok = n_ex = 0
    for start in range(0, len(x_cpu), batch_size):
        x = x_cpu[start:start + batch_size].to(device, non_blocking=True)
        y = y_cpu[start:start + batch_size].to(device, non_blocking=True)
        logits = model(x)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, N_SYMBOLS), y.reshape(-1), reduction="sum"
        ).item())
        pred = logits.argmax(-1)
        token_ok += int((pred == y).sum())
        exact_ok += int((pred == y).all(1).sum())
        n_tok += y.numel()
        n_ex += y.shape[0]
    model.train()
    return {
        "eval_loss": loss_sum / n_tok,
        "token_accuracy": token_ok / n_tok,
        "exact_accuracy": exact_ok / n_ex,
    }


def lr_at_step(step, total_steps, peak_lr, warmup, schedule="cosine", min_ratio=0.1):
    if warmup > 0 and step < warmup:
        return peak_lr * (step + 1) / warmup
    if schedule == "constant" or total_steps <= warmup:
        return peak_lr
    if schedule != "cosine":
        raise ValueError(schedule)
    progress = min(max((step - warmup) / max(1, total_steps - warmup - 1), 0.0), 1.0)
    return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["uniform", "static", "balanced_slow", "balanced_fast"])
    p.add_argument("--profile", default="pilot", choices=sorted(PROFILES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mapping-seed", type=int, default=1729)
    p.add_argument("--schedule-seed", type=int, default=2718)
    p.add_argument("--stream-seed", type=int, default=31415)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--peak-lr", type=float, default=2e-4)
    p.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("outputs"))
    for name in ["cycles", "block_steps", "eval_every", "eval_examples", "bin_eval_examples"]:
        p.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    profile = PROFILES[args.profile]
    cycles = args.cycles or profile.cycles
    block_steps = args.block_steps or profile.block_steps
    eval_every = args.eval_every or profile.eval_every
    eval_examples = args.eval_examples or profile.eval_examples
    bin_eval_examples = args.bin_eval_examples or profile.bin_eval_examples
    if cycles <= 0 or block_steps <= 0:
        raise ValueError("cycles/block_steps must be positive")

    n_blocks, total_steps = cycles * N_SKILLS, cycles * N_SKILLS * block_steps
    seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    run_dir = args.output / args.profile / f"seed{args.seed}" / args.condition
    run_dir.mkdir(parents=True, exist_ok=True)
    perm_table = all_permutations()
    base_order = make_base_order(args.mapping_seed)
    shifts = make_shift_schedule(args.condition, cycles, args.schedule_seed + args.seed)
    audit = schedule_audit(args.condition, shifts, base_order, args.alpha)

    prob = np.full(N_SKILLS, 1 / N_SKILLS) if args.condition == "uniform" else zipf_prob(args.alpha)
    rank_counts = largest_remainder_counts(prob, block_steps * args.batch_size * HOPS)
    realized = np.zeros(N_SKILLS, dtype=np.int64)
    for shift in shifts:
        realized[rank_to_skill_for_shift(base_order, int(shift))] += rank_counts
    audit.update({
        "positions_per_block": int(rank_counts.sum()),
        "rank_count_min_per_block": int(rank_counts.min()),
        "rank_count_max_per_block": int(rank_counts.max()),
        "realized_skill_count_min": int(realized.min()),
        "realized_skill_count_max": int(realized.max()),
        "realized_skill_counts_equal": bool(realized.min() == realized.max()),
    })
    (run_dir / "schedule_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    eval_x, eval_y = fixed_eval(eval_examples, 9000 + args.seed, perm_table)
    bins = []
    for b, ranks in enumerate(np.array_split(np.arange(N_SKILLS), 5)):
        bins.append(fixed_eval(
            bin_eval_examples, 10000 + args.seed * 10 + b, perm_table, base_order[ranks]
        ))

    seed_all(args.seed)
    model = StateTrackingTransformer(args.d_model, args.layers, args.heads, args.ff_mult).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr,
        betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay,
    )

    use_amp = device.type == "cuda" and args.precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    fp16_scaling = use_amp and args.precision == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=fp16_scaling)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=fp16_scaling)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=fp16_scaling)

    cfg = vars(args).copy()
    cfg["output"] = str(cfg["output"])
    cfg.update({
        "cycles_resolved": cycles,
        "block_steps_resolved": block_steps,
        "eval_every_resolved": eval_every,
        "eval_examples_resolved": eval_examples,
        "bin_eval_examples_resolved": bin_eval_examples,
        "n_blocks": n_blocks,
        "total_steps": total_steps,
        "n_skills": N_SKILLS,
        "hops": HOPS,
        "sequence_length": SEQ_LEN,
    })
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    fields = [
        "step", "block", "lr", "train_loss", "eval_loss", "token_accuracy", "exact_accuracy",
        "bin1_token_accuracy", "bin2_token_accuracy", "bin3_token_accuracy",
        "bin4_token_accuracy", "bin5_token_accuracy", "wall_seconds",
    ]
    metrics_f = (run_dir / "metrics.csv").open("w", newline="")
    writer = csv.DictWriter(metrics_f, fieldnames=fields)
    writer.writeheader()
    start_time, global_step, last_loss = time.time(), 0, float("nan")

    def log_eval(block_idx: int):
        main_m = evaluate(model, eval_x, eval_y, device, args.eval_batch_size)
        bin_acc = [evaluate(model, x, y, device, args.eval_batch_size)["token_accuracy"] for x, y in bins]
        row = {
            "step": global_step,
            "block": block_idx,
            "lr": opt.param_groups[0]["lr"],
            "train_loss": last_loss,
            **main_m,
            **{f"bin{i+1}_token_accuracy": a for i, a in enumerate(bin_acc)},
            "wall_seconds": time.time() - start_time,
        }
        writer.writerow(row)
        metrics_f.flush()
        print(json.dumps(row), flush=True)

    log_eval(0)
    for block_idx, shift in enumerate(shifts):
        rank_block = exact_rank_block(
            prob, block_steps, args.batch_size, HOPS,
            block_stream_seed(args.condition, block_idx, int(shift), args.seed, args.stream_seed),
        )
        r2s = rank_to_skill_for_shift(base_order, int(shift))
        for local_step in range(block_steps):
            skills = r2s[rank_block[local_step].astype(np.int64, copy=False)]
            x_np, y_np = make_batch_from_skills(skills, perm_table)
            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            y = torch.from_numpy(y_np).to(device, non_blocking=True)
            lr = lr_at_step(global_step, total_steps, args.peak_lr, args.warmup_steps, args.lr_schedule)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, N_SYMBOLS), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, N_SYMBOLS), y.reshape(-1))
                loss.backward()
                opt.step()
            last_loss = float(loss.detach())
            global_step += 1
            if global_step % eval_every == 0 or global_step == total_steps:
                log_eval(block_idx + 1)

    metrics_f.close()
    (run_dir / "done.json").write_text(json.dumps({
        "condition": args.condition,
        "profile": args.profile,
        "seed": args.seed,
        "total_steps": total_steps,
        "wall_seconds": time.time() - start_time,
        "schedule_audit": audit,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
