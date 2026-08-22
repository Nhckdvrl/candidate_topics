"""Technical preflight for state identity, controlled randomness, and feature capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .libero_common import LIBERO_ENV_RESOLUTION, get_task_suite, make_env, policy_element, settle_initial_state
from .state_contract import deterministic_noise_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from openpi_client import websocket_client_policy

    suite = get_task_suite(args.suite)
    task = suite.get_task(args.task_id)
    init = suite.get_task_init_states(args.task_id)[args.init_idx]
    env = make_env(task, resolution=LIBERO_ENV_RESOLUTION, env_seed=args.env_seed)
    try:
        obs1, h1 = settle_initial_state(env, init, env_seed=args.env_seed, wait_steps=10)
        _, h2 = settle_initial_state(env, init, env_seed=args.env_seed, wait_steps=10)
    finally:
        env.close()
    if h1 != h2:
        raise RuntimeError("same LIBERO initial state does not reproduce the same settled sim hash")

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    element = policy_element(obs1, task.language)
    s0 = deterministic_noise_seed(999001, suite=args.suite, task_id=args.task_id, init_idx=args.init_idx, replan_idx=0)
    s1 = deterministic_noise_seed(999002, suite=args.suite, task_id=args.task_id, init_idx=args.init_idx, replan_idx=0)

    def infer(seed):
        q = dict(element)
        q["__topic09_noise_seed"] = int(seed)
        q["__topic09_capture_feature"] = True
        return client.infer(q)

    a, b, c = infer(s0), infer(s0), infer(s1)
    aa, ab, ac = map(lambda z: np.asarray(z["actions"], float), [a, b, c])
    fa, fb = map(lambda z: np.asarray(z["topic09_feature"], float), [a, b])
    same_action_max_abs = float(np.max(np.abs(aa - ab)))
    same_feature_max_abs = float(np.max(np.abs(fa - fb)))
    different_noise_action_rms = float(np.sqrt(np.mean((aa - ac) ** 2)))

    report = {
        "settled_state_hash_reproducible": h1 == h2,
        "same_noise_action_max_abs": same_action_max_abs,
        "same_noise_feature_max_abs": same_feature_max_abs,
        "different_noise_action_rms": different_noise_action_rms,
        "feature_dim": int(fa.size),
        "feature_meta": a.get("topic09_feature_meta", {}),
        "pass": bool(
            h1 == h2
            and same_action_max_abs == 0.0
            and same_feature_max_abs == 0.0
            and different_noise_action_rms > 1e-6
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
