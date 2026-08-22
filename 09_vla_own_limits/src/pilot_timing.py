"""Measure real rollout cost so the G0 budget is a number, not a guess.

Runs a handful of full rollouts through the frozen protocol and reports seconds per
rollout, split into policy inference and simulator stepping, plus the projected wall clock
for the full discovery and confirmation panels at a given parallelism.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from .libero_common import (
    LIBERO_ENV_RESOLUTION,
    MAX_STEPS,
    get_task_suite,
    make_env,
    parse_int_spec,
    policy_element,
    settle_initial_state,
)
from .state_contract import deterministic_noise_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--task-ids", default="0,5")
    p.add_argument("--init-indices", default="0,1")
    p.add_argument("--policy-seeds", default="110000,110001")
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--wait-steps", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--parallel-streams", type=int, default=12)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    suite = get_task_suite(args.suite)
    rows = []

    for task_id in parse_int_spec(args.task_ids):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env = make_env(task, resolution=LIBERO_ENV_RESOLUTION, env_seed=args.env_seed)
        try:
            for init_idx in parse_int_spec(args.init_indices):
                for policy_seed in parse_int_spec(args.policy_seeds):
                    t0 = time.monotonic()
                    obs, _ = settle_initial_state(
                        env, init_states[init_idx], env_seed=args.env_seed, wait_steps=args.wait_steps
                    )
                    infer_s, sim_s, n_infer = 0.0, 0.0, 0
                    plan = collections.deque()
                    replan_idx, steps, done = 0, 0, False
                    while steps < MAX_STEPS[args.suite]:
                        if not plan:
                            el = policy_element(obs, task.language)
                            el["__topic09_noise_seed"] = deterministic_noise_seed(
                                policy_seed, suite=args.suite, task_id=task_id,
                                init_idx=init_idx, replan_idx=replan_idx,
                            )
                            t = time.monotonic()
                            chunk = client.infer(el)["actions"]
                            infer_s += time.monotonic() - t
                            n_infer += 1
                            plan.extend(chunk[: args.replan_steps])
                            replan_idx += 1
                        t = time.monotonic()
                        obs, _, done, _ = env.step(plan.popleft().tolist())
                        sim_s += time.monotonic() - t
                        steps += 1
                        if done:
                            break
                    rows.append({
                        "task_id": task_id, "init_idx": init_idx, "policy_seed": policy_seed,
                        "success": int(bool(done)), "steps": steps, "n_infer": n_infer,
                        "wall_s": time.monotonic() - t0,
                        "infer_s": infer_s, "sim_s": sim_s,
                    })
                    print(json.dumps(rows[-1]))
        finally:
            env.close()

    wall = np.asarray([r["wall_s"] for r in rows])
    infer = np.asarray([r["infer_s"] for r in rows])
    n_inf = np.asarray([r["n_infer"] for r in rows])
    mean_wall = float(wall.mean())
    # 3 checkpoints x 150 states x 8 seeds, then 2 checkpoints x 150 x 8
    disc, conf = 3 * 150 * 8, 2 * 150 * 8
    report = {
        "n_rollouts": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_wall_s": mean_wall,
        "mean_infer_s": float(infer.mean()),
        "mean_sim_s": float(np.mean([r["sim_s"] for r in rows])),
        "mean_infers_per_rollout": float(n_inf.mean()),
        "mean_ms_per_infer": float(1000 * infer.sum() / max(1, n_inf.sum())),
        "projection": {
            "parallel_streams": args.parallel_streams,
            "discovery_rollouts": disc,
            "confirmation_rollouts": conf,
            "discovery_hours": disc * mean_wall / 3600 / args.parallel_streams,
            "confirmation_hours": conf * mean_wall / 3600 / args.parallel_streams,
        },
        "rollouts": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rollouts"}, indent=2))


if __name__ == "__main__":
    main()
