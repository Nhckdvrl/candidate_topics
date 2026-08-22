"""Reproduce the released evaluation of `lerobot/diffusion_pusht`.

This is the check that our inference path is faithful. The released model card reports
**65.4% success over 500 episodes on seeds 1000-1499** (avg max reward 0.955). If our
own closed-loop rollouts land near that, then the observation preprocessing, the
normalisation we had to restore by hand (see `policy_utils`), the crop mode, and the
sampler are all correct.

It matters more than usual here. A subtly broken policy still emits plausible-looking
actions; it just becomes uncertain everywhere. That failure mode would *manufacture* a
positive result for this topic, so it has to be excluded before any measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env_utils import make_env
from .policy_utils import (
    assert_normalization_loaded,
    load_pusht_policy,
    push_observation,
    reset_queue,
    sample_action_chunks,
)
from .sim_state import coverage


def run_episode(bundle, env, seed: int, max_steps: int = 300) -> dict:
    obs, _ = env.reset(seed=seed)
    reset_queue(bundle)
    push_observation(bundle, obs)
    max_cov = coverage(env)
    step = 0
    terminated = False
    while step < max_steps and not terminated:
        _pred, executed = sample_action_chunks(bundle, 1, seed=seed * 10007 + step)
        chunk = executed[0]
        for a in chunk:
            obs, _r, terminated, _t, info = env.step(a)
            push_observation(bundle, obs)
            max_cov = max(max_cov, float(info["coverage"]))
            step += 1
            if terminated or step >= max_steps:
                break
    return {"seed": seed, "success": bool(terminated), "max_coverage": max_cov, "steps": step}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="lerobot/diffusion_pusht")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=1000, help="the released eval seeds")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    bundle = load_pusht_policy(args.pretrained, device=args.device)
    assert_normalization_loaded(bundle)
    env = make_env(obs_type=bundle.obs_type)

    rows = []
    for i in range(args.episodes):
        rows.append(run_episode(bundle, env, args.seed_start + i, args.max_steps))
        if (i + 1) % 10 == 0:
            sr = np.mean([r["success"] for r in rows])
            print(f"[{i+1}/{args.episodes}] running success={sr:.3f}", flush=True)
    env.close()

    sr = float(np.mean([r["success"] for r in rows]))
    mc = float(np.mean([r["max_coverage"] for r in rows]))
    n = len(rows)
    se = float(np.sqrt(sr * (1 - sr) / n))
    summary = {
        "pretrained": args.pretrained,
        "episodes": n,
        "seed_start": args.seed_start,
        "success_rate": sr,
        "success_rate_se": se,
        "mean_max_coverage": mc,
        "released_success_rate": 0.654,
        "released_mean_max_reward": 0.9551,
        "within_2se_of_released": bool(abs(sr - 0.654) <= 2 * max(se, 1e-9)),
        "per_episode": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
