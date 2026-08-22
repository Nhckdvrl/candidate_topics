from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

PROTOCOL_VERSION = "topic14-v3-2026-08-22"
N_SYMBOLS = 5
N_SKILLS = math.factorial(N_SYMBOLS)
HOPS = 4
SEQ_LEN = N_SYMBOLS * HOPS
PAPER_LR_WARMUP_STEPS = 1000
PAPER_MIN_LR_RATIO = 0.1
LOCKED_FULL_SEEDS = (0, 1, 2, 3, 4)
LOCKED_PAPER_ANCHOR_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Profile:
    branch_warmup_steps: int
    core_steps: int
    eval_every: int
    eval_examples: int
    phase_steps: int
    lr_schedule: str


PROFILES = {
    "smoke": Profile(20, 200, 50, 512, 100, "constant"),
    "pilot": Profile(1000, 80_000, 2_000, 8_192, 40_000, "constant"),
    "full": Profile(1000, 160_000, 4_000, 16_384, 80_000, "constant"),
    # Near-paper anchor: no shared data warmup. Each arm receives its own
    # distribution from step 0 while LR follows 1000-step warmup + cosine.
    "paper_anchor": Profile(0, 200_000, 5_000, 16_384, 100_000, "paper_cosine"),
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_digest(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def all_permutations() -> np.ndarray:
    return np.asarray(list(itertools.permutations(range(N_SYMBOLS))), dtype=np.int64)


def compose_numpy(perms: np.ndarray) -> np.ndarray:
    """Compose g1∘g2∘... with (sigma∘pi)[i] = sigma[pi[i]]."""
    out = perms[:, 0].copy()
    for h in range(1, perms.shape[1]):
        out = np.take_along_axis(out, perms[:, h], axis=1)
    return out


def zipf_prob(alpha: float = 1.5) -> np.ndarray:
    ranks = np.arange(1, N_SKILLS + 1, dtype=np.float64)
    p = ranks ** (-alpha)
    return p / p.sum()


def mapping_seed_for_seed(base_seed: int, model_seed: int) -> int:
    """Predeclared mapping randomization across paired replication seeds."""
    return int(base_seed + 1009 * model_seed)


def base_order(mapping_seed: int) -> np.ndarray:
    return np.random.default_rng(mapping_seed).permutation(N_SKILLS).astype(np.int64)


def map_orders(mapping_seed: int) -> tuple[np.ndarray, np.ndarray]:
    # One random rank->skill assignment, then a fixed half-cycle shift. The
    # relation is identical across replications, while actual skill labels change.
    a = base_order(mapping_seed)
    b = np.roll(a, N_SKILLS // 2)
    return a, b


def head_overlap(a: np.ndarray, b: np.ndarray, head_fraction: float = 0.2) -> int:
    k = max(1, round(N_SKILLS * head_fraction))
    return len(set(map(int, a[:k])) & set(map(int, b[:k])))


def key_schedule(
    condition: str,
    core_steps: int,
    phase_steps: int,
    persistence_h: int | None = None,
) -> list[tuple[str, int]]:
    if core_steps % 2:
        raise ValueError("core_steps must be even")
    half = core_steps // 2
    if condition == "slow":
        if phase_steps != half:
            raise ValueError("slow requires phase_steps == core_steps/2")
        return [("A", i) for i in range(half)] + [("B", i) for i in range(half)]
    if condition == "fast":
        out: list[tuple[str, int]] = []
        for i in range(half):
            out.extend([("A", i), ("B", i)])
        return out
    if condition == "persistence":
        if persistence_h is None or persistence_h <= 0:
            raise ValueError("positive persistence_h required")
        out = []
        ia = ib = 0
        turn = "A"
        while ia < half or ib < half:
            if turn == "A" and ia < half:
                n = min(persistence_h, half - ia)
                out.extend(("A", i) for i in range(ia, ia + n))
                ia += n
            elif turn == "B" and ib < half:
                n = min(persistence_h, half - ib)
                out.extend(("B", i) for i in range(ib, ib + n))
                ib += n
            turn = "B" if turn == "A" else "A"
        return out
    raise ValueError(condition)


def schedule_digests(keys: Sequence[tuple[str, int]]) -> dict[str, str]:
    ordered = json.dumps(list(keys), separators=(",", ":")).encode()
    multiset = json.dumps(sorted(keys), separators=(",", ":")).encode()
    return {
        "temporal_digest": hashlib.sha256(ordered).hexdigest(),
        "multiset_digest": hashlib.sha256(multiset).hexdigest(),
    }


def max_map_run(keys: Sequence[tuple[str, int]]) -> int:
    best = cur = 0
    prev = None
    for m, _ in keys:
        if m == prev:
            cur += 1
        else:
            cur = 1
            prev = m
        best = max(best, cur)
    return best


def batch_seed(seed: int, map_id: str, occurrence_id: int, stream_seed: int) -> int:
    if map_id not in {"A", "B"}:
        raise ValueError(map_id)
    return int(
        stream_seed
        + seed * 1_000_003
        + (0 if map_id == "A" else 10_000_000)
        + occurrence_id
    )


def make_power_batch(
    seed: int,
    map_id: str,
    occurrence_id: int,
    batch_size: int,
    alpha: float,
    mapping_seed: int,
    stream_seed: int,
    perm_table: np.ndarray,
):
    a, b = map_orders(mapping_seed)
    r2s = a if map_id == "A" else b
    rng = np.random.default_rng(batch_seed(seed, map_id, occurrence_id, stream_seed))
    ranks = rng.choice(N_SKILLS, size=(batch_size, HOPS), replace=True, p=zipf_prob(alpha))
    perms = perm_table[r2s[ranks]]
    return perms.reshape(batch_size, -1), compose_numpy(perms)


def make_uniform_batch(
    seed: int,
    step: int,
    batch_size: int,
    stream_seed: int,
    perm_table: np.ndarray,
):
    rng = np.random.default_rng(stream_seed + seed * 1_000_003 + step)
    skills = rng.integers(0, N_SKILLS, size=(batch_size, HOPS))
    perms = perm_table[skills]
    return perms.reshape(batch_size, -1), compose_numpy(perms)


def fixed_eval(n: int, eval_seed: int, perm_table: np.ndarray):
    rng = np.random.default_rng(eval_seed)
    skills = rng.integers(0, N_SKILLS, size=(n, HOPS))
    perms = perm_table[skills]
    return torch.from_numpy(perms.reshape(n, -1)), torch.from_numpy(compose_numpy(perms))


def paper_lr_at_step(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int = PAPER_LR_WARMUP_STEPS,
    min_ratio: float = PAPER_MIN_LR_RATIO,
) -> float:
    """Seed-paper schedule: linear warmup, cosine decay to 0.1x peak."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps + 1:
        return peak_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


class StateTrackingTransformer(nn.Module):
    def __init__(self, d_model=256, layers=4, heads=8, ff_mult=4):
        super().__init__()
        self.token_emb = nn.Embedding(N_SYMBOLS, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_SYMBOLS)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        h = self.token_emb(x) + self.pos_emb[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(self.ln(h[:, -N_SYMBOLS:]))


def model_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    state = model.state_dict()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def branch_digest(model: nn.Module, optimizer: torch.optim.Optimizer) -> str:
    h = hashlib.sha256()
    h.update(model_digest(model).encode())
    obj = optimizer.state_dict()
    for pid in sorted(obj["state"], key=int):
        h.update(str(pid).encode())
        for k in sorted(obj["state"][pid]):
            v = obj["state"][pid][k]
            h.update(k.encode())
            h.update(
                v.detach().cpu().contiguous().numpy().tobytes()
                if torch.is_tensor(v)
                else repr(v).encode()
            )
    h.update(json.dumps(obj["param_groups"], sort_keys=True, default=str).encode())
    return h.hexdigest()


def profile_dict(name: str) -> dict:
    return asdict(PROFILES[name])
