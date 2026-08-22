"""Technical preflight. Nothing scientific is interpreted until all of this passes.

Checks, in order of how badly a failure would invalidate the experiment:

  P0  restore round-trip: saving a state and restoring it must leave the *observable*
      state unchanged. P1/P2 cannot catch a restore that is wrong-but-consistent.
  P1  simulator determinism: the same action sequence from the same restored state
      produces a bit-identical outcome.
  P2  restore fidelity: saving a state, perturbing the sim, restoring, and replaying
      reproduces the un-perturbed continuation exactly.
  P3  `step_physics` (no render) == `env.step` (with render) on the block trajectory.
  P4  ACE reproduces released-FIPER numbers: our D-dimensional generalisation equals the
      released 3-D code path on zero-padded 2-D actions.
  P5  the policy loads, produces distinct samples from one observation, and its actions
      are inside the env's action bounds.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def p0_restore_round_trip(seed: int = 3) -> dict:
    """save -> restore must be the identity on the observable state.

    P1 and P2 only compare two replays with each other, so a restore that is wrong the
    *same way* every time passes both. This check compares the restored state against the
    state that was saved. It is the check that caught the block being teleported by up to
    ~59 px, which silently zeroed every outcome dispersion in the first smoke run.
    """
    from .env_utils import make_env
    from .sim_state import block_keypoints, coverage, restore_sim_state, save_sim_state

    env = make_env(obs_type="state")
    env.reset(seed=seed)
    for _ in range(6):
        env.step(np.array([300.0, 250.0], dtype=np.float32))

    st = save_sim_state(env)
    kp, cov = block_keypoints(env), coverage(env)
    restore_sim_state(env, st)
    kp2, cov2 = block_keypoints(env), coverage(env)
    st2 = save_sim_state(env)
    env.close()

    return {
        "name": "P0_restore_round_trip",
        "max_keypoint_diff_px": float(np.max(np.abs(kp - kp2))),
        "max_state_diff": float(np.max(np.abs(st.as_array() - st2.as_array()))),
        "coverage_diff": float(abs(cov - cov2)),
        "pass": bool(np.max(np.abs(kp - kp2)) == 0.0 and abs(cov - cov2) == 0.0),
    }


def p1_determinism(seed: int = 0) -> dict:
    """Same restored state + same actions must give a bit-identical outcome.

    Also reports what happens *without* the space rebuild, because that number is the
    reason `restore_sim_state` rebuilds: Chipmunk's warm-start arbiter cache survives a
    naive state restore and silently perturbs the replay.
    """
    from .env_utils import execute_chunk, make_env
    from .sim_state import restore_sim_state, save_sim_state

    env = make_env(obs_type="state")
    env.reset(seed=seed)
    for _ in range(5):
        env.step(np.array([256.0, 300.0], dtype=np.float32))
    st = save_sim_state(env)
    chunk = np.array([[200.0 + 10 * i, 300.0 - 5 * i] for i in range(8)], dtype=np.float64)

    restore_sim_state(env, st)
    a = execute_chunk(env, chunk)
    restore_sim_state(env, st)
    b = execute_chunk(env, chunk)

    restore_sim_state(env, st, rebuild=False)
    c = execute_chunk(env, chunk)
    env.close()

    err = float(np.max(np.abs(a["keypoints"] - b["keypoints"])))
    err_no_rebuild = float(np.max(np.abs(a["keypoints"] - c["keypoints"])))
    return {
        "name": "P1_determinism",
        "max_keypoint_diff_px": err,
        "max_keypoint_diff_px_without_space_rebuild": err_no_rebuild,
        "pass": err == 0.0,
    }


def p2_restore_fidelity(seed: int = 1) -> dict:
    """Restore must survive an intervening, unrelated rollout."""
    from .env_utils import execute_chunk, make_env
    from .sim_state import restore_sim_state, save_sim_state

    env = make_env(obs_type="state")
    env.reset(seed=seed)
    for _ in range(7):
        env.step(np.array([300.0, 250.0], dtype=np.float32))
    st = save_sim_state(env)
    chunk = np.array([[150.0 + 20 * i, 350.0 - 10 * i] for i in range(8)], dtype=np.float64)

    restore_sim_state(env, st)
    ref = execute_chunk(env, chunk)

    # Contaminate the simulator with a long unrelated rollout, then restore.
    restore_sim_state(env, st)
    for _ in range(40):
        env.step(np.array([500.0, 20.0], dtype=np.float32))
    restore_sim_state(env, st)
    again = execute_chunk(env, chunk)
    env.close()

    err = float(np.max(np.abs(ref["keypoints"] - again["keypoints"])))
    return {"name": "P2_restore_fidelity", "max_keypoint_diff_px": err, "pass": err == 0.0}


def p3_step_physics_matches_env_step(seed: int = 2) -> dict:
    from .env_utils import make_env, step_physics
    from .sim_state import block_keypoints, restore_sim_state, save_sim_state

    env = make_env(obs_type="state")
    env.reset(seed=seed)
    st = save_sim_state(env)
    chunk = np.array([[250.0 + 15 * i, 260.0 + 8 * i] for i in range(8)], dtype=np.float64)

    restore_sim_state(env, st)
    for a in chunk:
        env.step(a.astype(np.float32))
    kp_env = block_keypoints(env)

    restore_sim_state(env, st)
    for a in chunk:
        step_physics(env, a.astype(np.float32))
    kp_fast = block_keypoints(env)
    env.close()

    err = float(np.max(np.abs(kp_env - kp_fast)))
    return {"name": "P3_step_physics_matches_env_step", "max_keypoint_diff_px": err, "pass": err < 1e-9}


def p4_ace_matches_released_fiper(seed: int = 3) -> dict:
    """Our generalised ACE vs a literal transcription of FIPER's 3-D `_entropy_endpoints`."""
    from scipy.stats import entropy as shannon

    from .ace import ace, calibration_cell_size

    rng = np.random.default_rng(seed)
    calib = rng.normal(scale=30.0, size=(200, 16, 2)) + 256.0
    chunks = rng.normal(scale=12.0, size=(64, 16, 2)) + 256.0

    cell2 = calibration_cell_size(calib)
    ours = ace(chunks, cell2)

    # --- literal FIPER path: pad to 3-D, released cell-size rule, released grid code ---
    calib3 = np.concatenate([calib, np.zeros((*calib.shape[:2], 1))], axis=-1)
    chunks3 = np.concatenate([chunks, np.zeros((*chunks.shape[:2], 1))], axis=-1)
    pos = calib3.reshape(-1, 3)
    ranges = pos.max(axis=0) - pos.min(axis=0)
    ranges = np.where(ranges == 0, ranges.max(), ranges)
    cell = ranges * 0.03

    def fiper_entropy_endpoints(endpoints):
        lims = []
        for k in range(3):
            mn, mx = endpoints[:, k].min(), endpoints[:, k].max()
            buf = 0.01 * (mx - mn)
            lims.append((mn - buf, mx + buf))
        grids = [np.arange(lo, hi + cell[k], cell[k]) for k, (lo, hi) in enumerate(lims)]
        idx = [np.digitize(endpoints[:, k], grids[k]) - 1 for k in range(3)]
        n = [max(len(g) - 1, 1) for g in grids]
        counts = np.zeros(n, dtype=int)
        for i in range(len(endpoints)):
            counts[idx[0][i], idx[1][i], idx[2][i]] += 1
        return float(shannon(counts.flatten(), base=2))

    ref = float(np.mean([fiper_entropy_endpoints(chunks3[:, h, :]) for h in range(chunks3.shape[1])]))
    err = abs(ours - ref)
    return {
        "name": "P4_ace_matches_released_fiper",
        "ours_bits": ours,
        "released_path_bits": ref,
        "abs_diff": err,
        "pass": err < 1e-9,
    }


def p5_policy_sampling(device: str = "cuda", n: int = 16, pretrained: str = "lerobot/diffusion_pusht") -> dict:
    from .env_utils import make_env
    from .policy_utils import (
        assert_normalization_loaded,
        load_pusht_policy,
        push_observation,
        sample_action_chunks,
    )

    bundle = load_pusht_policy(pretrained, device=device)
    assert_normalization_loaded(bundle)  # raises if the checkpoint's stats went missing
    env = make_env(obs_type=bundle.obs_type)
    obs, _ = env.reset(seed=7)
    push_observation(bundle, obs)
    predicted, chunks = sample_action_chunks(bundle, n, seed=0)
    env.close()

    spread = float(np.mean(np.std(chunks, axis=0)))
    lo, hi = float(chunks.min()), float(chunks.max())
    return {
        "name": "P5_policy_sampling",
        "predicted_shape": list(predicted.shape),
        "executed_shape": list(chunks.shape),
        "mean_std_across_samples_px": spread,
        "action_min": lo,
        "action_max": hi,
        "pass": bool(spread > 1e-3 and -50.0 <= lo and hi <= 562.0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--pretrained", default="lerobot/diffusion_pusht")
    p.add_argument("--skip-policy", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = [p0_restore_round_trip(), p1_determinism(), p2_restore_fidelity(), p3_step_physics_matches_env_step(), p4_ace_matches_released_fiper()]
    if not args.skip_policy:
        results.append(p5_policy_sampling(device=args.device, pretrained=args.pretrained))

    ok = all(r["pass"] for r in results)
    payload = {"all_pass": ok, "checks": results}
    print(json.dumps(payload, indent=2))
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
