from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import torch

from .diffusion import DiffusionPolicy
from .geometry import decompose_chunks, fiper_ace
from .planar_arm import PlanarArm, sample_reachable_problem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--states", type=int, default=256)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--rollout-episodes", type=int, default=200)
    p.add_argument("--execute-steps", type=int, default=4)
    p.add_argument("--ace-alpha", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_policy(path: Path, device: torch.device) -> DiffusionPolicy:
    payload = torch.load(path, map_location=device)
    policy = DiffusionPolicy.from_payload(payload, map_location=device)
    policy.eval()
    return policy


def execute_chunk(arm: PlanarArm, q: np.ndarray, chunk: np.ndarray, n_steps: int) -> np.ndarray:
    q2 = np.asarray(q, dtype=np.float64).copy()
    for a in chunk[:n_steps]:
        q2 = arm.step(q2, a)
    return q2


def rollout_success(policy: DiffusionPolicy, arm: PlanarArm, rng: np.random.Generator, device: torch.device, episodes: int) -> float:
    wins = 0
    for _ in range(episodes):
        q, target = sample_reachable_problem(arm, rng)
        for _ in range(arm.cfg.max_steps):
            if arm.distance(q, target) < arm.cfg.success_tol:
                break
            obs = torch.tensor(np.concatenate([q, target]), dtype=torch.float32, device=device)
            chunk = policy.sample(obs, n_samples=1)[0, 0].detach().cpu().numpy()
            q = execute_chunk(arm, q, chunk, min(4, len(chunk)))
        wins += int(arm.distance(q, target) < arm.cfg.success_tol)
    return wins / episodes


def sample_eval_states(arm: PlanarArm, rng: np.random.Generator, n: int) -> Iterable[tuple[np.ndarray, np.ndarray, str]]:
    produced = 0
    while produced < n:
        q, target = sample_reachable_problem(arm, rng)
        sigma, tag = [(0.0, "id"), (0.15, "perturb015"), (0.30, "perturb030")][produced % 3]
        if sigma > 0:
            q = np.clip(q + rng.normal(scale=sigma, size=arm.n_joints), -2.6, 2.6)
        d = arm.distance(q, target)
        if 0.10 <= d <= 1.20:
            yield q, target, tag
            produced += 1


def state_metrics(
    policy: DiffusionPolicy,
    arm: PlanarArm,
    q: np.ndarray,
    target: np.ndarray,
    tag: str,
    ranges: np.ndarray,
    n_samples: int,
    execute_steps: int,
    ace_alpha: float,
    device: torch.device,
) -> dict:
    obs = torch.tensor(np.concatenate([q, target]), dtype=torch.float32, device=device)
    chunks = policy.sample(obs, n_samples=n_samples)[0].detach().cpu().numpy()
    geom = decompose_chunks(chunks, arm.jacobian(q))
    ace = fiper_ace(chunks, ranges, alpha=ace_alpha)
    ace0 = fiper_ace(chunks[:, :1, :], ranges, alpha=ace_alpha)
    d0 = arm.distance(q, target)
    final_d = np.array([
        arm.distance(execute_chunk(arm, q, c, execute_steps), target) for c in chunks
    ])
    progress = (d0 - final_d) / max(d0, 1e-6)
    bad = progress < 0.15
    return {
        "tag": tag,
        "initial_distance": float(d0),
        "ace": float(ace),
        "ace_step0": float(ace0),
        "task_var": float(geom["task_per_dim_sum"]),
        "null_var": float(geom["null_per_dim_sum"]),
        "task_total": float(geom["task_total_sum"]),
        "null_total": float(geom["null_total_sum"]),
        "task_fraction": float(geom["task_fraction"]),
        "total_variance": float(geom["total_variance_sum"]),
        "risk": float(bad.mean()),
        "mean_progress": float(progress.mean()),
        "task_rank": int(geom["task_rank"]),
        "null_rank": int(geom["null_rank"]),
    }


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = args.checkpoint
    if ckpt is None:
        ckpts = sorted(args.run_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            raise FileNotFoundError("no checkpoint_*.pt found")
        ckpt = ckpts[-1]
    policy = load_policy(ckpt, device)
    arm = PlanarArm()
    rng = np.random.default_rng(args.seed)
    ranges = np.load(args.run_dir / "ace_calibration_ranges.npy")

    success = rollout_success(policy, arm, rng, device, args.rollout_episodes)
    rows = []
    for i, (q, target, tag) in enumerate(sample_eval_states(arm, rng, args.states)):
        row = {"state_id": i}
        row.update(state_metrics(
            policy, arm, q, target, tag, ranges, args.samples,
            args.execute_steps, args.ace_alpha, device,
        ))
        rows.append(row)
        if (i + 1) % 16 == 0:
            print(f"evaluated {i+1}/{args.states}", flush=True)

    out_csv = args.run_dir / f"state_metrics_{ckpt.stem}.csv"
    save_csv(rows, out_csv)
    summary = {
        "checkpoint": str(ckpt),
        "rollout_success": float(success),
        "states": args.states,
        "samples_per_state": args.samples,
        "execute_steps": args.execute_steps,
        "ace_alpha": args.ace_alpha,
        "state_metrics_csv": str(out_csv),
    }
    (args.run_dir / f"eval_{ckpt.stem}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
