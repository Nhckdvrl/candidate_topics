"""Verify that the loaded checkpoints are actually different models.

Every downstream conclusion assumes the servers hold *different* pi0.5 checkpoints. If a
conversion silently wrote the same weights twice, or two servers were pointed at the same
directory, the pipeline would still run end to end and produce a beautifully clean null:
no competence crossover, and a relative score of exactly zero. That failure mode looks
like a scientific result, so it has to be excluded mechanically.

Given the same settled observation and the same policy-noise seed, two genuinely different
checkpoints must produce different actions and different layer-11 features.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .libero_common import LIBERO_ENV_RESOLUTION, get_task_suite, make_env, policy_element, settle_initial_state
from .state_contract import deterministic_noise_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--ports", type=int, nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--min-action-rms", type=float, default=1e-4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if len(args.ports) != len(args.names):
        raise ValueError("--ports and --names must align")
    if len(args.ports) < 2:
        raise ValueError("need at least two checkpoints to compare")

    from openpi_client import websocket_client_policy

    suite = get_task_suite(args.suite)
    task = suite.get_task(args.task_id)
    init = suite.get_task_init_states(args.task_id)[args.init_idx]
    env = make_env(task, resolution=LIBERO_ENV_RESOLUTION, env_seed=args.env_seed)
    try:
        obs, sim_hash = settle_initial_state(env, init, env_seed=args.env_seed, wait_steps=10)
    finally:
        env.close()

    element = policy_element(obs, task.language)
    seed = deterministic_noise_seed(
        999003, suite=args.suite, task_id=args.task_id, init_idx=args.init_idx, replan_idx=0
    )

    actions, feats = {}, {}
    for name, port in zip(args.names, args.ports, strict=True):
        client = websocket_client_policy.WebsocketClientPolicy(args.host, port)
        req = dict(element)
        req["__topic09_noise_seed"] = int(seed)
        req["__topic09_capture_feature"] = True
        out = client.infer(req)
        actions[name] = np.asarray(out["actions"], float)
        feats[name] = np.asarray(out["topic09_feature"], float)

    pairs = []
    ok = True
    for x, y in itertools.combinations(args.names, 2):
        a_rms = float(np.sqrt(np.mean((actions[x] - actions[y]) ** 2)))
        f_rms = float(np.sqrt(np.mean((feats[x] - feats[y]) ** 2)))
        passed = a_rms > args.min_action_rms and f_rms > 0.0
        ok = ok and passed
        pairs.append({"a": x, "b": y, "action_rms": a_rms, "feature_rms": f_rms, "differ": passed})

    report = {
        "suite": args.suite,
        "task_id": int(args.task_id),
        "init_idx": int(args.init_idx),
        "sim_state_hash": sim_hash,
        "shared_noise_seed": int(seed),
        "checkpoints": list(args.names),
        "pairs": pairs,
        "pass": bool(ok),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
