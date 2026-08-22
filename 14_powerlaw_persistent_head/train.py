#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from core import (
    PAPER_LR_WARMUP_STEPS,
    PROFILES,
    PROTOCOL_VERSION,
    StateTrackingTransformer,
    all_permutations,
    branch_digest,
    canonical_digest,
    fixed_eval,
    key_schedule,
    make_power_batch,
    make_uniform_batch,
    mapping_seed_for_seed,
    model_digest,
    paper_lr_at_step,
    profile_dict,
    schedule_digests,
    seed_all,
)

EVAL_SEED = 424242
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["warmup", "arm"], required=True)
    p.add_argument("--condition", choices=["uniform", "static", "slow", "fast", "persistence"])
    p.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=2048)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--mapping-seed", type=int, default=1729, help="base mapping seed; effective seed is predeclared per replication seed")
    p.add_argument("--stream-seed", type=int, default=31415)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--peak-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--persistence-h", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=20000)
    return p.parse_args()


def device_for(x: str) -> torch.device:
    if x.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(x)


def resolve_precision(profile: str, requested: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if requested != "auto":
        return requested
    return "fp16" if profile == "paper_anchor" else "bf16"


def model_for(a, d):
    return StateTrackingTransformer(a.d_model, a.layers, a.heads, a.ff_mult).to(d)


def opt_for(m, a):
    return torch.optim.AdamW(
        m.parameters(),
        lr=a.peak_lr,
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=a.weight_decay,
    )


def scaler_for(device: torch.device, precision: str):
    return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))


def ac(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.inference_mode()
def evaluate(m, x0, y0, d, bs, precision):
    m.eval()
    exact = token = n_examples = n_tokens = 0
    loss_sum = 0.0
    for i in range(0, len(x0), bs):
        x = x0[i : i + bs].to(d)
        y = y0[i : i + bs].to(d)
        with ac(d, precision):
            z = m(x)
            loss = F.cross_entropy(z.reshape(-1, 5), y.reshape(-1), reduction="sum")
        pred = z.argmax(-1)
        loss_sum += float(loss)
        exact += int((pred == y).all(1).sum())
        token += int((pred == y).sum())
        n_examples += y.shape[0]
        n_tokens += y.numel()
    m.train()
    return {
        "exact_accuracy": exact / n_examples,
        "token_accuracy": token / n_tokens,
        "eval_loss": loss_sum / n_tokens,
    }


def train_step(m, o, scaler, x, y, d, precision):
    o.zero_grad(set_to_none=True)
    with ac(d, precision):
        z = m(x)
        loss = F.cross_entropy(z.reshape(-1, 5), y.reshape(-1))
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(o)
        scaler.update()
    else:
        loss.backward()
        o.step()
    return float(loss.detach())


def save_checkpoint(path: Path, m, o, scaler, step: int, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": m.state_dict(),
            "optimizer": o.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "extra": extra,
        },
        path,
    )


def load_checkpoint(path: Path, m, o, scaler, d):
    ck = torch.load(path, map_location=d, weights_only=False)
    m.load_state_dict(ck["model"])
    o.load_state_dict(ck["optimizer"])
    if ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck


def branch_config(a, precision: str) -> dict:
    pr = PROFILES[a.profile]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "profile": a.profile,
        "seed": a.seed,
        "branch_warmup_steps": pr.branch_warmup_steps,
        "branch_warmup_distribution": "uniform" if pr.branch_warmup_steps else "none",
        "batch_size": a.batch_size,
        "stream_seed": a.stream_seed,
        "d_model": a.d_model,
        "layers": a.layers,
        "heads": a.heads,
        "ff_mult": a.ff_mult,
        "peak_lr": a.peak_lr,
        "weight_decay": a.weight_decay,
        "betas": [BETA1, BETA2],
        "eps": EPS,
        "precision": precision,
    }


def warmup(a):
    pr = PROFILES[a.profile]
    root = a.output / a.profile / f"seed{a.seed}"
    root.mkdir(parents=True, exist_ok=True)
    out = root / "branch.pt"
    meta_path = root / "branch.json"
    d = device_for(a.device)
    precision = resolve_precision(a.profile, a.precision, d)
    expected_cfg = branch_config(a, precision)
    expected_sig = canonical_digest(expected_cfg)

    if out.exists() and a.resume:
        if not meta_path.exists():
            raise SystemExit(f"stale branch without metadata: {out}")
        old = json.loads(meta_path.read_text())
        if old.get("branch_signature") != expected_sig:
            raise SystemExit("STALE_OR_MISMATCHED_BRANCH: configuration changed; use a fresh output directory")
        print(f"branch exists and matches protocol: {out}")
        return

    seed_all(a.seed)
    m = model_for(a, d)
    o = opt_for(m, a)
    scaler = scaler_for(d, precision)
    perm = all_permutations()

    for s in range(pr.branch_warmup_steps):
        lr = a.peak_lr * (s + 1) / max(1, pr.branch_warmup_steps)
        for g in o.param_groups:
            g["lr"] = lr
        xn, yn = make_uniform_batch(a.seed, s, a.batch_size, a.stream_seed + 99_000_000, perm)
        x = torch.from_numpy(xn).long().to(d)
        y = torch.from_numpy(yn).long().to(d)
        train_step(m, o, scaler, x, y, d, precision)

    for g in o.param_groups:
        g["lr"] = a.peak_lr
    extra = {
        "protocol_version": PROTOCOL_VERSION,
        "branch_config": expected_cfg,
        "branch_signature": expected_sig,
        "model_digest": model_digest(m),
        "branch_digest": branch_digest(m, o),
    }
    save_checkpoint(out, m, o, scaler, pr.branch_warmup_steps, extra)
    meta_path.write_text(json.dumps(extra, indent=2) + "\n")


def latest_checkpoint(run_dir: Path) -> Path | None:
    found = []
    for p in run_dir.glob("checkpoint_*.pt"):
        try:
            step = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        found.append((step, p))
    return max(found, default=(None, None))[1]


def read_metric_rows(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def run_config(a, precision: str, branch_hash: str, schedule: dict | None) -> dict:
    pr = PROFILES[a.profile]
    eff_mapping = mapping_seed_for_seed(a.mapping_seed, a.seed)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "profile": a.profile,
        "profile_resolved": profile_dict(a.profile),
        "seed": a.seed,
        "condition": a.condition,
        "persistence_h": a.persistence_h,
        "batch_size": a.batch_size,
        "eval_batch_size": a.eval_batch_size,
        "alpha": a.alpha,
        "mapping_seed_base": a.mapping_seed,
        "mapping_seed_effective": eff_mapping,
        "stream_seed": a.stream_seed,
        "peak_lr": a.peak_lr,
        "lr_schedule": pr.lr_schedule,
        "paper_lr_warmup_steps": PAPER_LR_WARMUP_STEPS if pr.lr_schedule == "paper_cosine" else 0,
        "weight_decay": a.weight_decay,
        "betas": [BETA1, BETA2],
        "eps": EPS,
        "precision": precision,
        "d_model": a.d_model,
        "layers": a.layers,
        "heads": a.heads,
        "ff_mult": a.ff_mult,
        "eval_seed": EVAL_SEED,
        "branch_digest": branch_hash,
        "schedule": schedule,
    }


def arm(a):
    if a.condition is None:
        raise ValueError("--condition required")
    if a.profile == "paper_anchor" and a.condition not in {"uniform", "static"}:
        raise ValueError("paper_anchor is an anchor-only diagnostic: uniform/static only")

    pr = PROFILES[a.profile]
    root = a.output / a.profile / f"seed{a.seed}"
    branch = root / "branch.pt"
    if not branch.exists():
        raise SystemExit(f"missing common branch checkpoint: {branch}")
    name = a.condition if a.condition != "persistence" else f"persistence_h{a.persistence_h}"
    rd = root / name
    rd.mkdir(parents=True, exist_ok=True)
    done_path = rd / "done.json"
    cfg_path = rd / "config.json"
    metrics_path = rd / "metrics.csv"

    d = device_for(a.device)
    precision = resolve_precision(a.profile, a.precision, d)
    seed_all(a.seed)
    m = model_for(a, d)
    o = opt_for(m, a)
    scaler = scaler_for(d, precision)
    ck = load_checkpoint(branch, m, o, scaler, d)
    start_branch = branch_digest(m, o)
    if start_branch != ck["extra"]["branch_digest"]:
        raise SystemExit("branch checkpoint digest mismatch after load")
    if ck["extra"].get("protocol_version") != PROTOCOL_VERSION:
        raise SystemExit("STALE_BRANCH_PROTOCOL")

    perm = all_permutations()
    xe, ye = fixed_eval(pr.eval_examples, EVAL_SEED, perm)
    eff_mapping = mapping_seed_for_seed(a.mapping_seed, a.seed)
    keys = (
        key_schedule(a.condition, pr.core_steps, pr.phase_steps, a.persistence_h)
        if a.condition in {"slow", "fast", "persistence"}
        else None
    )
    schedule = None
    if keys is not None:
        schedule = schedule_digests(keys) | {
            "condition": a.condition,
            "persistence_h": a.persistence_h,
            "n_steps": len(keys),
        }
        (rd / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")

    cfg = run_config(a, precision, start_branch, schedule)
    cfg["run_signature"] = canonical_digest({k: v for k, v in cfg.items() if k != "run_signature"})

    if done_path.exists() and a.resume:
        if not cfg_path.exists():
            raise SystemExit("STALE_DONE_WITHOUT_CONFIG")
        old = json.loads(cfg_path.read_text())
        if old.get("run_signature") != cfg["run_signature"]:
            raise SystemExit("STALE_OR_MISMATCHED_RUN: completed output does not match requested protocol")
        print(f"done and protocol-matched: {rd}")
        return

    start_step = 0
    rows = []
    if a.resume:
        cp = latest_checkpoint(rd)
        if cp is not None:
            rck = load_checkpoint(cp, m, o, scaler, d)
            if rck.get("extra", {}).get("branch_digest") != start_branch:
                raise SystemExit("RESUME_CHECKPOINT_BRANCH_MISMATCH")
            if rck.get("extra", {}).get("run_signature") != cfg["run_signature"]:
                raise SystemExit("RESUME_CHECKPOINT_PROTOCOL_MISMATCH")
            start_step = int(rck["step"])
            rows = [r for r in read_metric_rows(metrics_path) if int(r["step"]) <= start_step]
            if not rows or int(rows[0]["step"]) != 0:
                raise SystemExit("RESUME_METRICS_MISSING_STEP0")
            print(f"resuming {name} seed={a.seed} from step {start_step}")

    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    fields = ["step", "exact_accuracy", "token_accuracy", "eval_loss"]
    if start_step == 0:
        rows = [{"step": 0, **evaluate(m, xe, ye, d, a.eval_batch_size, precision)}]

    with metrics_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
        f.flush()

        for s in range(start_step, pr.core_steps):
            if pr.lr_schedule == "paper_cosine":
                lr = paper_lr_at_step(s, pr.core_steps, a.peak_lr)
                for g in o.param_groups:
                    g["lr"] = lr
            else:
                for g in o.param_groups:
                    g["lr"] = a.peak_lr

            if a.condition == "uniform":
                xn, yn = make_uniform_batch(a.seed, s, a.batch_size, a.stream_seed, perm)
            elif a.condition == "static":
                xn, yn = make_power_batch(
                    a.seed, "A", s, a.batch_size, a.alpha, eff_mapping, a.stream_seed, perm
                )
            else:
                mid, occ = keys[s]
                xn, yn = make_power_batch(
                    a.seed, mid, occ, a.batch_size, a.alpha, eff_mapping, a.stream_seed, perm
                )
            x = torch.from_numpy(xn).long().to(d)
            y = torch.from_numpy(yn).long().to(d)
            train_step(m, o, scaler, x, y, d, precision)
            step = s + 1

            if step % pr.eval_every == 0 or step == pr.core_steps:
                row = {"step": step, **evaluate(m, xe, ye, d, a.eval_batch_size, precision)}
                rows.append(row)
                w.writerow(row)
                f.flush()

            if a.save_every > 0 and step % a.save_every == 0 and step < pr.core_steps:
                save_checkpoint(
                    rd / f"checkpoint_{step}.pt",
                    m,
                    o,
                    scaler,
                    step,
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "branch_digest": start_branch,
                        "run_signature": cfg["run_signature"],
                    },
                )

    final = {
        "protocol_version": PROTOCOL_VERSION,
        "run_signature": cfg["run_signature"],
        "branch_digest": start_branch,
        "final_model_digest": model_digest(m),
        "last": rows[-1],
    }
    done_path.write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


def main():
    a = parse_args()
    warmup(a) if a.mode == "warmup" else arm(a)


if __name__ == "__main__":
    main()
