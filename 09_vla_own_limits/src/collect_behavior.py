"""Collect repeated same-state LIBERO rollouts for one frozen checkpoint."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import pandas as pd

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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--task-ids", default="0-9")
    p.add_argument("--init-indices", required=True)
    p.add_argument("--policy-seeds", required=True)
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--wait-steps", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from openpi_client import websocket_client_policy

    if args.suite not in MAX_STEPS:
        raise ValueError(f"unknown suite {args.suite}")
    task_ids = parse_int_spec(args.task_ids)
    init_indices = parse_int_spec(args.init_indices)
    policy_seeds = parse_int_spec(args.policy_seeds)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    suite = get_task_suite(args.suite)
    rows = []

    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        if max(init_indices) >= len(init_states):
            raise IndexError(f"task {task_id} has only {len(init_states)} init states")
        env = make_env(task, resolution=LIBERO_ENV_RESOLUTION, env_seed=args.env_seed)
        expected_hash = {}
        try:
            for init_idx in init_indices:
                for policy_seed in policy_seeds:
                    obs, sim_hash = settle_initial_state(
                        env, init_states[init_idx], env_seed=args.env_seed, wait_steps=args.wait_steps
                    )
                    if init_idx in expected_hash and expected_hash[init_idx] != sim_hash:
                        raise RuntimeError(
                            f"settled state changed across repeats: task={task_id} init={init_idx}"
                        )
                    expected_hash[init_idx] = sim_hash

                    action_plan = collections.deque()
                    replan_idx = 0
                    done = False
                    steps = 0
                    while steps < MAX_STEPS[args.suite]:
                        if not action_plan:
                            element = policy_element(obs, task.language, resize_size=args.resize_size)
                            element["__topic09_noise_seed"] = deterministic_noise_seed(
                                policy_seed,
                                suite=args.suite,
                                task_id=task_id,
                                init_idx=init_idx,
                                replan_idx=replan_idx,
                            )
                            result = client.infer(element)
                            action_chunk = result["actions"]
                            if len(action_chunk) < args.replan_steps:
                                raise RuntimeError("policy returned an action chunk shorter than replan_steps")
                            action_plan.extend(action_chunk[: args.replan_steps])
                            replan_idx += 1
                        action = action_plan.popleft()
                        obs, _, done, _ = env.step(action.tolist())
                        steps += 1
                        if done:
                            break

                    rows.append({
                        "suite": args.suite,
                        "task_id": int(task_id),
                        "init_idx": int(init_idx),
                        "env_seed": int(args.env_seed),
                        "sim_state_hash": sim_hash,
                        "checkpoint": str(args.checkpoint),
                        "policy_seed": int(policy_seed),
                        "success": int(bool(done)),
                        "status": "ok",
                        "steps": int(steps),
                        "replans": int(replan_idx),
                    })
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(args.out, index=False)
        finally:
            env.close()

    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
