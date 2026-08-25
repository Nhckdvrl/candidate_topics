"""P0b' for XMoveBendPickTeleop: is the VLA -> WBC seam actually state-dependent
on this task, re-verified fresh rather than assumed from CloseDoor?

CloseDoor's P0b already showed the WBC seam is command-level state-dependent,
and that the response is confined to the 12 leg + 3 waist joints (arms/hands
open-loop below the VLA seam). This file does not assume that transfers to a
different task on the architecture-level claim alone -- it re-runs the exact
same command-level, no-simulation-step measurement on XMoveBendPickTeleop.

    same config, same tick, same fixed vla_cmd, same WBC internal state,
    same clock -- only the proprioceptive observation differs.

        canonical proprio  -> WBC -> target_q^0
        perturbed proprio  -> WBC -> target_q^1

    D_wbc = |target_q^1 - target_q^0|

Frozen before any run, unchanged from CloseDoor's P0b: tick =
round(0.4 * len(tape)); perturbation = +0.05 rad body-frame roll offset on
the floating-base orientation and the torso IMU quaternion; gate is
structural (repeatability floor vs. perturbed divergence), not a scientific
threshold. The one thing that must be re-checked rather than copied is the
"approach phase" condition: CloseDoor checked the door hadn't moved yet;
here the check is that the target object hasn't started lifting yet
(`info["target"][2]` unchanged from its initial value), consistent with
XMoveBendPick's own height-based success predicate.
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

from p0_xmove_runner import (  # noqa: E402
    ENV_ID, VirtualClock, build_vla_cmd, _make_sonic_config, target_height,
)

TICK_FRACTION = 0.4
ROLL_PERTURB_RAD = 0.05
SEPARATION_FACTOR = 100.0

SKIP_KEYS = {
    "robot", "client", "_dwbc_robot_model", "_wbc_policy", "robot_model",
    "_last_observation", "policy_1", "policy_2", "config",
}


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


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
    return {
        k: (np.array(v, dtype=np.float64) if isinstance(v, np.ndarray) else v)
        for k, v in proprio.items()
    }


def perturb_proprio(proprio: dict[str, Any], theta: float) -> dict[str, Any]:
    out = copy_proprio(proprio)
    dq = roll_offset(theta)
    fbp = out["floating_base_pose"].copy()
    fbp[3:7] = quat_mul(fbp[3:7], dq)
    out["floating_base_pose"] = fbp
    out["secondary_imu_quat"] = quat_mul(out["secondary_imu_quat"], dq)
    return out


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

    sonic_config = _make_sonic_config()
    control_dt = 4 * sonic_config["SIMULATE_DT"]
    clock = VirtualClock(control_dt)
    sonic_mod.time = clock
    psi0_mod.time = clock

    ds = LeRobotDataset(repo_id=ENV_ID, root=args.data_dir, video_backend="pyav")
    raw_env = gym.make(
        ENV_ID, sim_mode=args.sim_mode, render_hz=ds.meta.fps,
        headless=True, sonic_config=sonic_config,
    )
    task = raw_env.unwrapped.task
    max_steps = args.max_episode_steps or task.metadata.get("max_episode_steps")
    env = TimeLimit(raw_env, max_episode_steps=max_steps)
    robot = task.robot
    agent = Psi0DecoupledWbcAgent(robot, args.host, args.port, sonic_config=sonic_config)

    body_joint_names: list[str] = []
    tape_dir = Path(args.tape_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dr_level = Path(args.data_dir).name

    def eval_seam(snaps, clock_t, tape_row, proprio, observation, info):
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
            print(f"[skip] {config_id}: no P0' tape at {tape_path}", flush=True)
            continue
        tape = json.loads(tape_path.read_text())
        tick = int(round(TICK_FRACTION * len(tape["vla"])))
        t0 = _real_time.perf_counter()

        random.seed(args.config_seed_base + eps_idx)
        np.random.seed(args.config_seed_base + eps_idx)
        env_conf, _ = get_episode_lerobot(ds, eps_idx)
        observation, info = env.reset(options={"state_dict": env_conf})
        sonic_env = raw_env.unwrapped
        model = sonic_env.mjModel
        if not body_joint_names:
            body_joint_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                for j in np.asarray(robot.body_joint_index)
            ]

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

        for row in tape["vla"]:
            agent.queue_action(build_vla_cmd(row))
        height_init = target_height(info)
        for _ in range(tick):
            cmd = agent.get_action(observation, info=info, instruction=task.instruction)
            observation, _r, terminated, truncated, info = env.step(cmd)
            if terminated or truncated:
                break
        height_at_tick = target_height(info)

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

        approach_untouched = (
            height_init is not None and height_at_tick is not None
            and abs(height_at_tick - height_init) < 1e-6
        )

        row = {
            "task": "xmove_bend_pick",
            "config_id": config_id,
            "tick": tick,
            "tape_len": len(tape["vla"]),
            "tick_fraction": TICK_FRACTION,
            "roll_perturb_rad": args.roll_perturb_rad,
            "height_init": height_init,
            "height_at_tick": height_at_tick,
            "approach_untouched_at_tick": approach_untouched,
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
            "clock": "virtual",
            "seconds": round(_real_time.perf_counter() - t0, 1),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[{config_id}] tick={tick}/{len(tape['vla'])} approach_untouched={approach_untouched} "
            f"repeat={d_repeat:.3e} restore={d_restore:.3e} perturb={d_perturb:.3e} "
            f"(hands {row['d_perturb_left_hand_max_rad']:.3e}/{row['d_perturb_right_hand_max_rad']:.3e}) "
            f"joints={len(changed)} -> {verdict} ({row['seconds']}s)",
            flush=True,
        )

    raw_env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tape-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=26085)
    p.add_argument("--sim-mode", default="mujoco_isaac")
    p.add_argument("--configs", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--roll-perturb-rad", type=float, default=ROLL_PERTURB_RAD)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--config-seed-base", type=int, default=20260824)
    run(p.parse_args())


if __name__ == "__main__":
    main()
