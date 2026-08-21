from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .diffusion import DiffusionConfig, DiffusionPolicy
from .geometry import fiper_calibration_ranges
from .planar_arm import generate_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-tasks", type=int, default=300)
    p.add_argument("--modes-per-task", type=int, default=4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--null-gain", type=float, default=1.0)
    p.add_argument("--train-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--save-steps", type=str, default="1000,3000,10000,30000")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data, summary = generate_dataset(
        args.base_tasks, args.modes_per_task, args.horizon, args.seed,
        null_gain=args.null_gain,
    )
    obs_np, act_np = data["obs"], data["action"]
    np.savez_compressed(args.out / "train_dataset.npz", **data)
    ranges = fiper_calibration_ranges(act_np)
    np.save(args.out / "ace_calibration_ranges.npy", ranges)

    obs = torch.from_numpy(obs_np)
    act = torch.from_numpy(act_np)
    ds = TensorDataset(obs, act)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = DiffusionConfig(
        obs_dim=obs.shape[-1], action_dim=act.shape[-1], horizon=args.horizon,
        diffusion_steps=args.diffusion_steps, hidden=args.hidden,
    )
    policy = DiffusionPolicy(cfg).to(device)
    policy.set_normalizer(obs.to(device), act.to(device))
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    save_steps = {int(x) for x in args.save_steps.split(",") if x.strip()}
    save_steps.add(args.train_steps)

    iterator = iter(loader)
    log = []
    for step in range(1, args.train_steps + 1):
        try:
            ob, ac = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            ob, ac = next(iterator)
        ob, ac = ob.to(device), ac.to(device)
        loss = policy.training_loss(ob, ac)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": float(loss.detach().cpu())}
            log.append(row)
            print(json.dumps(row), flush=True)
        if step in save_steps:
            torch.save(policy.checkpoint_payload(), args.out / f"checkpoint_{step:06d}.pt")

    meta = {
        "args": vars(args) | {"out": str(args.out)},
        "dataset": summary,
        "device": str(device),
        "final_loss": log[-1]["loss"] if log else None,
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / "train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
