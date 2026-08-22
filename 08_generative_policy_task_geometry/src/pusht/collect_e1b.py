"""E1b - does the choice of sampled action chunk change how the episode actually goes?

E1 measures the task outcome 8 steps after the probe. That is the horizon the policy
commits to, but it is myopic: in-contact outcome dispersion there is ~1 px on a 512 px
workspace, and a 1 px difference in block pose is not obviously "task uncertainty" at
all. Small differences could compound into different episode outcomes, or wash out
entirely. Nothing in E1 distinguishes those.

E1b resolves it by branching. At a probe state, K sampled chunks are executed from the
identical restored state and each branch is then continued **closed-loop under the same
policy** for a further horizon. The branch outcome is therefore an episode-level fact:
where the block ended up, and whether that branch reached the goal.

This is also what makes the decision-level question answerable. A runtime monitor like
FIPER thresholds a scalar entropy at a state and intervenes. The operationally meaningful
error is flagging a state as uncertain when *every* sampled action would have led to the
same place. E1b measures exactly that, with episode-level ground truth.

The K branches are stepped in lockstep across K env copies so the policy call is batched;
K branches cost little more than one.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np

from .ace import ace, action_dispersion, calibration_cell_size
from .collect_e1 import keypoint_dispersion
from .env_utils import agent_block_gap, make_env
from .policy_utils import (
    assert_normalization_loaded,
    load_pusht_policy,
    make_frame,
    push_observation,
    reset_queue,
    sample_action_chunks,
    sample_chunks_for_queues,
)
from .sim_state import block_keypoints, coverage, restore_sim_state, save_sim_state


def branch_from_state(bundle, envs, state, queue_src, chunks, extra_steps):
    """Run K branches from one restored state, then continue closed-loop in lockstep."""
    k = len(envs)
    queues = [deque(list(queue_src), maxlen=bundle.n_obs_steps) for _ in range(k)]
    done = [False] * k
    max_cov = [0.0] * k

    for i, env in enumerate(envs):
        restore_sim_state(env, state)
        max_cov[i] = coverage(env)
        for a in chunks[i]:
            obs, _r, term, _t, info = env.step(a)
            queues[i].append(make_frame(bundle, obs))
            max_cov[i] = max(max_cov[i], float(info["coverage"]))
            if term:
                done[i] = True
                break

    steps = 0
    while steps < extra_steps and not all(done):
        nxt = sample_chunks_for_queues(bundle, queues, seed=None)
        for i, env in enumerate(envs):
            if done[i]:
                continue
            for a in nxt[i]:
                obs, _r, term, _t, info = env.step(a)
                queues[i].append(make_frame(bundle, obs))
                max_cov[i] = max(max_cov[i], float(info["coverage"]))
                if term:
                    done[i] = True
                    break
        steps += bundle.n_action_steps

    final_kp = np.stack([block_keypoints(e) for e in envs])
    final_cov = np.array([coverage(e) for e in envs], dtype=np.float64)
    return {
        "final_keypoints": final_kp,
        "final_coverage": final_cov,
        "max_coverage": np.asarray(max_cov, dtype=np.float64),
        "reached_goal": np.asarray(done, dtype=bool),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pretrained", default="lerobot/diffusion_pusht")
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=100000)
    p.add_argument("--samples", type=int, default=256, help="samples used to score ACE")
    p.add_argument("--branches", type=int, default=32, help="K chunks continued closed-loop")
    p.add_argument("--extra-steps", type=int, default=88, help="closed-loop steps after the chunk")
    p.add_argument("--probe-every", type=int, default=3, help="branch at every Nth replan")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--calib-rollouts", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--null-control", action="store_true",
                   help="give every branch the SAME chunk; the resulting dispersion is the "
                        "downstream-stochasticity floor, not an effect of the action choice")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bundle = load_pusht_policy(args.pretrained, device=args.device)
    assert_normalization_loaded(bundle)
    env = make_env(obs_type=bundle.obs_type)
    envs = [make_env(obs_type=bundle.obs_type) for _ in range(args.branches)]
    for e in envs:
        e.reset(seed=0)
    t0 = time.time()

    # ACE cell size from a calibration rollout set, exactly as in E1
    calib = []
    for i in range(args.calib_rollouts):
        seed = args.seed_base - 1 - i
        obs, _ = env.reset(seed=seed)
        reset_queue(bundle)
        push_observation(bundle, obs)
        step, term = 0, False
        while step < args.max_steps and not term:
            pred, ch = sample_action_chunks(bundle, 16, seed=seed * 7 + step)
            calib.append(pred)
            for a in ch[0]:
                obs, _r, term, _t, _i = env.step(a)
                push_observation(bundle, obs)
                step += 1
                if term or step >= args.max_steps:
                    break
    cell = calibration_cell_size(np.concatenate(calib, axis=0))
    np.save(args.out / "ace_cell_size.npy", cell)
    print(f"[calib] cell_size={cell} ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for r in range(args.rollouts):
        ep_seed = args.seed_base + r
        obs, _ = env.reset(seed=ep_seed)
        reset_queue(bundle)
        push_observation(bundle, obs)
        step, probe, term = 0, 0, False
        while step < args.max_steps and not term:
            predicted, chunks = sample_action_chunks(bundle, args.samples, seed=ep_seed * 1000 + probe)

            if probe % args.probe_every == 0:
                st = save_sim_state(env)
                kp0, cov0, gap0 = block_keypoints(env), coverage(env), agent_block_gap(env)
                sel = chunks[: args.branches]
                if args.null_control:
                    # Every branch executes chunk 0. Any dispersion that survives is
                    # produced purely by the policy's own sampling during the closed-loop
                    # continuation, so it is the floor that the real measurement must
                    # clear before it can be attributed to the choice of action chunk.
                    sel = np.repeat(chunks[:1], args.branches, axis=0)
                out = branch_from_state(bundle, envs, st, list(bundle.queue), sel, args.extra_steps)
                fk, fc = out["final_keypoints"], out["final_coverage"]
                row = {
                    "rollout": int(ep_seed),
                    "probe": probe,
                    "step": step,
                    "cov_before": cov0,
                    "agent_block_gap_px": gap0,
                    "ace": ace(predicted, cell),
                    "branches": int(args.branches),
                    "extra_steps": int(args.extra_steps),
                    # episode-level outcome dispersion across branches
                    "branch_final_kp_dispersion_px": keypoint_dispersion(fk),
                    "branch_final_cov_std": float(fc.std(ddof=1)),
                    "branch_final_cov_mean": float(fc.mean()),
                    "branch_max_cov_std": float(out["max_coverage"].std(ddof=1)),
                    "branch_goal_rate": float(out["reached_goal"].mean()),
                    # 0 when every branch agrees, 0.5 at maximal disagreement
                    "branch_goal_disagreement": float(
                        min(out["reached_goal"].mean(), 1 - out["reached_goal"].mean())
                    ),
                    "kp_shift_from_probe_px": float(
                        np.sqrt(((fk.mean(axis=0) - kp0) ** 2).sum(axis=-1).mean())
                    ),
                }
                row.update(action_dispersion(chunks))
                rows.append(row)
                restore_sim_state(env, st)

            for a in chunks[0]:
                obs, _r, term, _t, _i = env.step(a)
                push_observation(bundle, obs)
                step += 1
                if term or step >= args.max_steps:
                    break
            probe += 1
        print(f"[rollout {r+1}/{args.rollouts}] seed={ep_seed} branched={len(rows)} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    env.close()
    for e in envs:
        e.close()

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "branch_states.csv", index=False)
    meta = {
        "pretrained": args.pretrained,
        "rollouts": args.rollouts,
        "branches": args.branches,
        "extra_steps": args.extra_steps,
        "probe_every": args.probe_every,
        "null_control": bool(args.null_control),
        "n_branch_states": int(len(df)),
        "elapsed_s": time.time() - t0,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
