"""E1 - the existence test.

At every point where the policy re-plans during a closed-loop PushT rollout:

  1. sample B action chunks from the *same* observation;
  2. save the exact simulator state;
  3. execute each of the B chunks open-loop from that identical state;
  4. record the true task outcome (T-block pose) of each;
  5. continue the rollout with one of those B chunks (chunk 0), so the probe is free.

That yields, per probe state, a scalar action-diversity score (FIPER ACE, plus
estimator-free dispersion measures) and a *measured* task-outcome dispersion. The
existence question is whether the first determines the second.

The outcome is produced by pymunk contact dynamics and never touches the action-space
projections used later for geometry, so it cannot be an algebraic restatement of the
diversity measure -- which is exactly the defect that killed the planar-arm prototype
(see AUDIT.md, finding A1).

Per-sample chunks and outcomes are also written to an .npz per run so the task-geometry
stage can reuse them without re-simulating. Those files stay out of git.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from .ace import ace, action_dispersion, calibration_cell_size
from .env_utils import agent_block_gap, execute_chunk, make_env
from .policy_utils import (
    assert_normalization_loaded,
    load_pusht_policy,
    push_observation,
    reset_queue,
    sample_action_chunks,
)
from .sim_state import block_keypoints, coverage, restore_sim_state, save_sim_state


def keypoint_dispersion(kps: np.ndarray) -> float:
    """RMS keypoint deviation from the mean shape, in pixels.

    `kps` is [B, K, 2]. Folds the block's translation and rotation into one
    pixel-valued scalar without us choosing a weight between position and angle.
    """
    mean = kps.mean(axis=0, keepdims=True)
    return float(np.sqrt(((kps - mean) ** 2).sum(axis=-1).mean()))


def mean_pairwise_keypoint_distance(kps: np.ndarray) -> float:
    """Estimator-free companion to `keypoint_dispersion` (no mean-shape reference)."""
    b = kps.shape[0]
    flat = kps.reshape(b, -1)
    sq = (flat**2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * flat @ flat.T, 0.0)
    d = np.sqrt(d2 / kps.shape[1])
    return float(d[np.triu_indices(b, k=1)].mean())


def run_rollout(bundle, env, ep_seed, n_samples, max_steps, cell, sample_seed_base):
    obs, _ = env.reset(seed=ep_seed)
    reset_queue(bundle)
    push_observation(bundle, obs)
    rows, chunk_log, kp_log, cov_log = [], [], [], []
    step, probe, terminated = 0, 0, False

    while step < max_steps and not terminated:
        predicted, chunks = sample_action_chunks(bundle, n_samples, seed=sample_seed_base + probe)

        st = save_sim_state(env)
        kp0 = block_keypoints(env)
        cov0 = coverage(env)
        gap0 = agent_block_gap(env)

        kps, covs, contacts = [], [], []
        for b in range(n_samples):
            restore_sim_state(env, st)
            out = execute_chunk(env, chunks[b])
            kps.append(out["keypoints"])
            covs.append(out["coverage"])
            contacts.append(out["n_contacts"])
        kps = np.stack(kps)
        covs = np.asarray(covs, dtype=np.float64)

        row = {
            "rollout": int(ep_seed),
            "probe": probe,
            "step": step,
            "cov_before": cov0,
            "agent_block_gap_px": gap0,
            # FIPER scores the full predicted chunk; `ace_executed` is the same estimator
            # restricted to the steps actually run, reported so no conclusion depends on
            # which window the baseline is given.
            "ace": ace(predicted, cell),
            "ace_executed": ace(chunks, cell),
            "outcome_kp_dispersion_px": keypoint_dispersion(kps),
            "outcome_kp_pairwise_px": mean_pairwise_keypoint_distance(kps),
            "outcome_kp_mean_shift_px": float(np.sqrt(((kps.mean(axis=0) - kp0) ** 2).sum(axis=-1).mean())),
            "outcome_cov_mean": float(covs.mean()),
            "outcome_cov_std": float(covs.std(ddof=1)),
            "outcome_cov_delta": float(covs.mean() - cov0),
            "frac_samples_with_contact": float(np.mean(np.asarray(contacts) > 0)),
            "mean_contacts": float(np.mean(contacts)),
        }
        row.update(action_dispersion(chunks))
        rows.append(row)
        chunk_log.append(chunks.astype(np.float32))
        kp_log.append(kps.astype(np.float32))
        cov_log.append(covs.astype(np.float32))

        # Continue the rollout with sample 0 -- an honest draw from the same policy.
        restore_sim_state(env, st)
        for a in chunks[0]:
            obs, _r, terminated, _tr, _info = env.step(a)
            push_observation(bundle, obs)
            step += 1
            if terminated or step >= max_steps:
                break
        probe += 1

    for r in rows:
        r["rollout_terminated"] = bool(terminated)
        r["rollout_len"] = step
    raw = {
        "chunks": np.stack(chunk_log) if chunk_log else np.zeros((0,)),
        "keypoints": np.stack(kp_log) if kp_log else np.zeros((0,)),
        "coverage": np.stack(cov_log) if cov_log else np.zeros((0,)),
    }
    return rows, raw


def collect_calibration(bundle, env, seed_base, n_rollouts, max_steps):
    """FIPER derives ACE cell widths from a calibration rollout set disjoint from the
    states that get scored. Seeds are taken *below* the measurement seed base."""
    chunks = []
    for i in range(n_rollouts):
        seed = seed_base - 1 - i
        obs, _ = env.reset(seed=seed)
        reset_queue(bundle)
        push_observation(bundle, obs)
        step, term = 0, False
        while step < max_steps and not term:
            pred, ch = sample_action_chunks(bundle, 16, seed=seed * 7 + step)
            chunks.append(pred)
            for a in ch[0]:
                obs, _r, term, _t, _i = env.step(a)
                push_observation(bundle, obs)
                step += 1
                if term or step >= max_steps:
                    break
    return np.concatenate(chunks, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=100000,
                   help="episode seed base; held out from the released eval seeds 1000..1049")
    p.add_argument("--samples", type=int, default=256)  # FIPER push_t uses batch_size 256
    p.add_argument("--max-steps", type=int, default=300)  # matches the released eval episode limit
    p.add_argument("--calib-rollouts", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pretrained", default="lerobot/diffusion_pusht")
    p.add_argument("--save-raw", action="store_true", default=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bundle = load_pusht_policy(args.pretrained, device=args.device)
    assert_normalization_loaded(bundle)
    env = make_env(obs_type=bundle.obs_type)
    t0 = time.time()

    calib = collect_calibration(bundle, env, args.seed_base, args.calib_rollouts, args.max_steps)
    cell = calibration_cell_size(calib)
    np.save(args.out / "ace_cell_size.npy", cell)
    print(f"[calib] chunks={calib.shape} cell_size={cell} ({time.time()-t0:.0f}s)", flush=True)

    all_rows, raw_store = [], {}
    for i in range(args.rollouts):
        ep_seed = args.seed_base + i
        rows, raw = run_rollout(bundle, env, ep_seed, args.samples, args.max_steps, cell,
                                sample_seed_base=ep_seed * 1000)
        all_rows.extend(rows)
        if args.save_raw:
            for k, v in raw.items():
                raw_store[f"{k}_{ep_seed}"] = v
        print(f"[rollout {i+1}/{args.rollouts}] seed={ep_seed} probes={len(rows)} "
              f"total={len(all_rows)} elapsed={time.time()-t0:.0f}s", flush=True)
    env.close()

    import pandas as pd

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out / "probe_states.csv", index=False)
    if args.save_raw:
        np.savez_compressed(args.out / "raw_samples.npz", **raw_store)

    meta = {
        "pretrained": args.pretrained,
        "rollouts": args.rollouts,
        "seed_base": args.seed_base,
        "samples_per_state": args.samples,
        "max_steps": args.max_steps,
        "calib_rollouts": args.calib_rollouts,
        "n_probe_states": int(len(df)),
        "ace_cell_size": cell.tolist(),
        "elapsed_s": time.time() - t0,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
