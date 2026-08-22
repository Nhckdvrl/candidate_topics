"""Extract repeated layer-11 pi0.5 features at identical settled LIBERO states."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .libero_common import (
    LIBERO_ENV_RESOLUTION,
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
    p.add_argument("--feature-seeds", required=True)
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--wait-steps", type=int, default=10)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from openpi_client import websocket_client_policy

    task_ids = parse_int_spec(args.task_ids)
    init_indices = parse_int_spec(args.init_indices)
    feature_seeds = parse_int_spec(args.feature_seeds)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    suite = get_task_suite(args.suite)

    state_id, checkpoint, hashes, fseed, feats = [], [], [], [], []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env = make_env(task, resolution=LIBERO_ENV_RESOLUTION, env_seed=args.env_seed)
        try:
            for init_idx in init_indices:
                obs, sim_hash = settle_initial_state(
                    env, init_states[init_idx], env_seed=args.env_seed, wait_steps=args.wait_steps
                )
                base = policy_element(obs, task.language, resize_size=args.resize_size)
                sid = f"{args.suite}|t={task_id}|i={init_idx}|e={args.env_seed}"
                for fs in feature_seeds:
                    req = dict(base)
                    req["__topic09_noise_seed"] = deterministic_noise_seed(
                        fs,
                        suite=args.suite,
                        task_id=task_id,
                        init_idx=init_idx,
                        replan_idx=0,
                    )
                    req["__topic09_capture_feature"] = True
                    out = client.infer(req)
                    feat = np.asarray(out["topic09_feature"], dtype=np.float32)
                    if feat.ndim != 1 or not np.isfinite(feat).all():
                        raise RuntimeError(f"bad feature shape/value: {feat.shape}")
                    state_id.append(sid)
                    checkpoint.append(str(args.checkpoint))
                    hashes.append(sim_hash)
                    fseed.append(int(fs))
                    feats.append(feat)
        finally:
            env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        state_id=np.asarray(state_id),
        checkpoint=np.asarray(checkpoint),
        sim_state_hash=np.asarray(hashes),
        feature_seed=np.asarray(fseed, dtype=np.int64),
        feature=np.stack(feats, axis=0),
    )
    print(f"wrote {len(state_id)} feature rows to {args.out}")


if __name__ == "__main__":
    main()
