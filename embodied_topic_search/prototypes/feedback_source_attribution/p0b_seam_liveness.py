"""P0b: is the VLA -> WBC seam actually state-dependent?

P0 proved the two replay instruments are lossless: with no disturbance, both
replay conditions reproduce the live rollout exactly (max door deviation 0.0 rad).
That "perfect" result is also the reason P0 is not sufficient. If the whole-body
controller were a pure feedforward function of `vla_cmd`, `vla_replay` would
reproduce the live rollout exactly as well, and the later quantity

    S_vla_replay - S_actuator_replay

would be structurally zero rather than informative. P0 shows the plumbing is
lossless; it cannot show the two seams carry different feedback.

P0b is a **command-level** test, not a behavioural one. Nothing is pushed and
nothing is stepped after the measurement, so contact, dynamics, controller
history and trajectory phase cannot enter. At one frozen tick of an already
P0-passing canonical rollout it asks a question about a function:

    same config, same tick, same fixed vla_cmd, same WBC internal state,
    same clock -- only the proprioceptive observation differs.

        canonical proprio  -> WBC -> target_q^0
        perturbed proprio  -> WBC -> target_q^1

    D_wbc = |target_q^1 - target_q^0|

The canonical side is re-evaluated live rather than read back from the recorded
tape, so interpolation state, clock position and previous-tick history are held
identical by construction and cannot contribute to the difference.

Frozen before any run
---------------------
tick          `round(TICK_FRACTION * len(tape))`, derived only from the
              unperturbed canonical rollout. Approach phase: the door joint must
              still sit at its initial value, which is recorded and checked.
perturbation  a single +0.05 rad body-frame roll offset applied to the
              floating-base orientation and to the torso IMU quaternion. One
              magnitude, no sweep, no per-config choice. Chosen as the
              proprioceptive signature of a small lateral nudge: ~2.9 deg, far
              above float64 noise and far below any joint or balance limit.
gate          structural, not scientific. There is no "> 0.01 rad" threshold.

                repeatability  same observation + same command + restored state
                               -> difference at numerical zero
                liveness       perturbed observation + same command
                               -> difference clearly above that floor

              A `restore` probe re-runs the canonical evaluation *after* the
              perturbed one. If state restoration were incomplete the probe
              would not return the original vector, and the run reports an
              instrument failure rather than a liveness claim.

The gate asks only whether the whole post-WBC command vector changes
deterministically. Per-joint deltas are recorded for description; picking the
prettiest joint afterwards is not permitted to influence the verdict.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import sys
import time as _real_time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p0_runner import TASKS, VirtualClock, build_vla_cmd, _make_sonic_config  # noqa: E402

TICK_FRACTION = 0.4
ROLL_PERTURB_RAD = 0.05
# Numerical-separation factor, not a scientific effect size: it only prevents
# float jitter from being read as feedback.
SEPARATION_FACTOR = 100.0

# Snapshotting these would be pointless or impossible: shared read-only models,
# network clients, ONNX sessions, and the raw observation dict.
SKIP_KEYS = {
    "robot", "client", "_dwbc_robot_model", "_wbc_policy", "robot_model",
    "_last_observation", "policy_1", "policy_2", "config",
}


# ---------------------------------------------------------------------------
# State snapshot / restore
# ---------------------------------------------------------------------------
def snapshot(obj) -> tuple[dict[str, Any], list[str]]:
    kept, skipped = {}, []
    for k, v in vars(obj).items():
        if k in SKIP_KEYS:
            skipped.append(k)
            continue
        try:
            kept[k] = copy.deepcopy(v)
        except Exception:
            skipped.append(k)
    return kept, skipped


def restore(obj, snap: dict[str, Any]) -> None:
    for k, v in snap.items():
        setattr(obj, k, copy.deepcopy(v))


def snapshot_all(agent) -> tuple[list[tuple[Any, dict]], list[str]]:
    wbc = agent._wbc_policy
    targets = [agent, wbc, wbc.upper_body_policy, wbc.lower_body_policy]
    snaps, skipped = [], []
    for t in targets:
        s, sk = snapshot(t)
        snaps.append((t, s))
        skipped += [f"{type(t).__name__}.{k}" for k in sk]
    return snaps, skipped


def restore_all(snaps: list[tuple[Any, dict]]) -> None:
    for obj, snap in snaps:
        restore(obj, snap)


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z) convention as used by MuJoCo."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def roll_offset(theta: float) -> np.ndarray:
    return np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0])


def copy_proprio(proprio: dict[str, Any]) -> dict[str, Any]:
    """`prepare_obs` hands back live views into mjData; detach them."""
    return {
        k: (np.array(v, dtype=np.float64) if isinstance(v, np.ndarray) else v)
        for k, v in proprio.items()
    }


def perturb_proprio(proprio: dict[str, Any], theta: float) -> dict[str, Any]:
    """Apply one body-frame roll offset to the two orientation channels."""
    out = copy_proprio(proprio)
    dq = roll_offset(theta)
    fbp = out["floating_base_pose"].copy()
    fbp[3:7] = quat_mul(fbp[3:7], dq)
    out["floating_base_pose"] = fbp
    out["secondary_imu_quat"] = quat_mul(out["secondary_imu_quat"], dq)
    return out


# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch
    from gymnasium.wrappers import TimeLimit
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import simple.envs as _  # noqa: F401
    import simple.agents.sonic_decoupled_wbc_agent as sonic_mod
    import simple.baselines.psi0_decoupled_wbc as psi0_mod
    from simple.datasets.lerobot import get_episode_lerobot
    from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent
    import mujoco

    spec = TASKS[args.env_id]
    sonic_config = _make_sonic_config()
    control_dt = 4 * sonic_config["SIMULATE_DT"]
    if args.clock != "virtual":
        # P0b holds the clock identical across the paired evaluations by
        # construction; a wall clock cannot do that.
        raise SystemExit("P0b requires --clock virtual")
    clock = VirtualClock(control_dt)
    sonic_mod.time = clock
    psi0_mod.time = clock

    ds = LeRobotDataset(repo_id=args.env_id, root=args.data_dir, video_backend="pyav")
    raw_env = gym.make(
        args.env_id, sim_mode=args.sim_mode, render_hz=ds.meta.fps,
        headless=True, sonic_config=sonic_config,
    )
    task = raw_env.unwrapped.task
    max_steps = args.max_episode_steps or task.metadata.get("max_episode_steps")
    env = TimeLimit(raw_env, max_episode_steps=max_steps)
    task.success_criteria = args.success_criteria
    robot = task.robot
    agent = Psi0DecoupledWbcAgent(robot, args.host, args.port, sonic_config=sonic_config)

    model = raw_env.unwrapped.mjModel
    data = raw_env.unwrapped.mjData
    body_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
        for j in np.asarray(robot.body_joint_index)
    ]

    tape_dir = Path(args.tape_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dr_level = Path(args.data_dir).name

    def eval_seam(snaps, clock_t, tape_row, proprio, observation, info):
        """Run the upstream seam verbatim with one injected observation."""
        restore_all(snaps)
        clock.t = clock_t
        agent._action_queue = collections.deque()
        agent.queue_action(build_vla_cmd(tape_row))
        robot.prepare_obs = lambda: proprio  # type: ignore[method-assign]
        try:
            cmd = agent.get_action(observation, info=info, instruction=task.instruction)
        finally:
            del robot.prepare_obs
        return (
            np.asarray(cmd["target_q"], dtype=np.float64).copy(),
            np.asarray(cmd["left_hand_q"], dtype=np.float64).copy(),
            np.asarray(cmd["right_hand_q"], dtype=np.float64).copy(),
        )

    for eps_idx in args.configs:
        config_id = f"{dr_level}:{eps_idx}"
        tape_path = tape_dir / f"{dr_level}_cfg{eps_idx}.json"
        if not tape_path.exists():
            print(f"[skip] {config_id}: no P0 tape at {tape_path}", flush=True)
            continue
        tape = json.loads(tape_path.read_text())
        tick = int(round(TICK_FRACTION * len(tape["vla"])))
        t0 = _real_time.perf_counter()

        random.seed(args.config_seed_base + eps_idx)
        np.random.seed(args.config_seed_base + eps_idx)
        env_conf, _ = get_episode_lerobot(ds, eps_idx)
        observation, info = env.reset(options={"state_dict": env_conf})
        sonic_env = raw_env.unwrapped

        clock.reset()
        agent.reset()
        agent._wbc_policy.lower_body_policy.use_policy_action = True
        stabilize_steps = 0
        while not robot.stabilized and stabilize_steps < 400:
            action = agent.get_stabilize_action(observation)
            observation, *_, info = sonic_env.step(action)
            sonic_env.update_viewer()
            sonic_env.update_reward()
            stabilize_steps += 1
        agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1,), dtype=torch.float32)

        # Drive the rollout to the frozen tick exactly as `vla_replay` does.
        for row in tape["vla"]:
            agent.queue_action(build_vla_cmd(row))
        door_q_init = float(data.joint(spec["effect_joint"]).qpos[0])
        for _ in range(tick):
            cmd = agent.get_action(observation, info=info, instruction=task.instruction)
            observation, _r, terminated, truncated, info = env.step(cmd)
            if terminated or truncated:
                break
        door_q_at_tick = float(data.joint(spec["effect_joint"]).qpos[0])

        # --- the paired seam measurement -----------------------------------
        proprio0 = copy_proprio(robot.prepare_obs())
        proprio1 = perturb_proprio(proprio0, args.roll_perturb_rad)
        snaps, skipped = snapshot_all(agent)
        clock_t = clock.t
        tape_row = tape["vla"][tick]

        q0 = eval_seam(snaps, clock_t, tape_row, proprio0, observation, info)
        q0r = eval_seam(snaps, clock_t, tape_row, proprio0, observation, info)
        q1 = eval_seam(snaps, clock_t, tape_row, proprio1, observation, info)
        q0rr = eval_seam(snaps, clock_t, tape_row, proprio0, observation, info)

        def dmax(a, b):
            return float(max(np.max(np.abs(x - y)) for x, y in zip(a, b)))

        def dl2(a, b):
            return float(np.linalg.norm(np.concatenate([x - y for x, y in zip(a, b)])))

        d_repeat = dmax(q0r, q0)
        d_restore = dmax(q0rr, q0)
        d_perturb = dmax(q1, q0)
        per_joint = np.abs(q1[0] - q0[0])
        changed = [
            {"joint": body_joint_names[i], "delta_rad": float(per_joint[i])}
            for i in np.argsort(-per_joint)
            if per_joint[i] > max(d_repeat, 0.0)
        ]

        if d_restore > 0.0:
            verdict = "INSTRUMENT_FAIL_STATE_RESTORE"
        elif d_perturb > 0.0 and d_perturb >= SEPARATION_FACTOR * d_repeat:
            verdict = "SEAM_LIVE"
        else:
            verdict = "NO_LIVENESS"

        row = {
            "task": spec["short_name"],
            "config_id": config_id,
            "tick": tick,
            "tape_len": len(tape["vla"]),
            "tick_fraction": TICK_FRACTION,
            "roll_perturb_rad": args.roll_perturb_rad,
            "door_q_init": door_q_init,
            "door_q_at_tick": door_q_at_tick,
            "door_untouched_at_tick": abs(door_q_at_tick - door_q_init) < 1e-9,
            "d_repeat_rad": d_repeat,
            "d_restore_rad": d_restore,
            "d_perturb_rad": d_perturb,
            "d_perturb_l2_rad": dl2(q1, q0),
            "d_perturb_target_q_max_rad": float(np.max(np.abs(q1[0] - q0[0]))),
            "d_perturb_left_hand_max_rad": float(np.max(np.abs(q1[1] - q0[1]))),
            "d_perturb_right_hand_max_rad": float(np.max(np.abs(q1[2] - q0[2]))),
            "separation_ratio": (
                float("inf") if d_repeat == 0.0 and d_perturb > 0.0
                else float(d_perturb / (d_repeat + 1e-18))
            ),
            "n_joints_changed": len(changed),
            "joints_changed": changed,
            "per_joint_delta_rad": {
                body_joint_names[i]: float(per_joint[i]) for i in range(len(per_joint))
            },
            "unsnapshotted_attributes": sorted(set(skipped)),
            "verdict": verdict,
            "clock": args.clock,
            "seconds": round(_real_time.perf_counter() - t0, 1),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[{config_id}] tick={tick}/{len(tape['vla'])} door_untouched={row['door_untouched_at_tick']} "
            f"repeat={d_repeat:.3e} restore={d_restore:.3e} perturb={d_perturb:.3e} "
            f"(hands {row['d_perturb_left_hand_max_rad']:.3e}/{row['d_perturb_right_hand_max_rad']:.3e}) "
            f"joints={len(changed)} -> {verdict} ({row['seconds']}s)",
            flush=True,
        )

    raw_env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="simple/G1WholebodyCloseDoorTeleop-v0")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tape-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=22085)
    p.add_argument("--sim-mode", default="mujoco_isaac")
    p.add_argument("--clock", choices=("virtual", "real"), default="virtual")
    p.add_argument("--configs", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--roll-perturb-rad", type=float, default=ROLL_PERTURB_RAD)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--success-criteria", type=float, default=0.5)
    p.add_argument("--config-seed-base", type=int, default=20260824)
    run(p.parse_args())


if __name__ == "__main__":
    main()
