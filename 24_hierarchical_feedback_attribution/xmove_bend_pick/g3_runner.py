"""Topic 24 G3 (corrected): fresh vs vla_replay under physical disturbance on
XMoveBendPickTeleop. See G3_VLA_FEEDBACK_PREREGISTRATION.md.

Only two conditions are collected -- no actuator_replay. The WBC is fully
live in both:

    fresh        live observation -> live VLA -> live WBC -> robot
    vla_replay   recorded pre-disturbance VLA plan -> live WBC (live proprio) -> robot

Target/contact tracking, timing anchor and horizon reuse
canonical_reconnaissance.py's already-verified instrument exactly:
target_body_id from env.unwrapped.mujoco.mj_objects["target"].id, right hand
as the right_wrist_roll_link kinematic subtree, push_tick per config already
computed and frozen in CANONICAL_RECONNAISSANCE.md.
"""
from __future__ import annotations

import argparse
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
from canonical_reconnaissance import (  # noqa: E402
    resolve_target_body_id, right_hand_body_set, hand_target_contact, RIGHT_HAND_ROOT,
)

CONDITIONS = ("fresh", "vla_replay")


def extend(tape: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if not tape:
        raise ValueError("empty tape")
    return list(tape) + [tape[-1]] * max(0, n - len(tape))


def base_yaw(data) -> float:
    w, x, y, z = (float(v) for v in np.asarray(data.qpos[3:7], dtype=float))
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Pusher:
    def __init__(self, model, data, force_n: float, direction: str, tick: int | None, control_dt: float):
        import mujoco

        self.data = data
        self.force_n = float(force_n)
        self.direction = direction
        self.tick = tick
        self.n_ticks = int(round(0.2 / control_dt))
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.body < 0:
            raise RuntimeError("torso_link not in model")
        self.applied_ticks = 0
        self.vec = np.zeros(3)

    def latch_direction(self) -> None:
        yaw = base_yaw(self.data)
        sign = 1.0 if self.direction == "left" else -1.0
        self.vec = sign * self.force_n * np.array([-np.sin(yaw), np.cos(yaw), 0.0])

    def step(self, frame_idx: int) -> None:
        if self.force_n <= 0.0 or self.tick is None:
            return
        if frame_idx == self.tick:
            self.latch_direction()
        active = self.tick <= frame_idx < self.tick + self.n_ticks
        f = np.zeros(6)
        if active:
            f[:3] = self.vec
            self.applied_ticks += 1
        self.data.xfrc_applied[self.body] = f


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

    sonic_config = _make_sonic_config()
    control_dt = 4 * sonic_config["SIMULATE_DT"]
    clock = VirtualClock(control_dt)
    sonic_mod.time = clock
    psi0_mod.time = clock

    _seen: dict[str, Any] = {"vla": None}
    _orig_get_action = None

    def install_hook():
        nonlocal _orig_get_action
        from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
        _orig_get_action = SonicDecoupledWbcAgent.get_action

        def _hooked(self, observation, instruction=None, **kwargs):
            cmd = _orig_get_action(self, observation, instruction, **kwargs)
            if cmd is not None and cmd.type == "vla_cmd":
                _seen["vla"] = cmd
            return cmd

        SonicDecoupledWbcAgent.get_action = _hooked

    install_hook()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tape_dir = Path(args.tape_dir)
    tape_dir.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, float, str, str]] = set()
    if out_path.exists() and args.resume:
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["config_id"], float(r["force_n"]), r["direction"], r["condition"]))
        print(f"[resume] {len(done)} records already present", flush=True)

    ds = LeRobotDataset(repo_id=ENV_ID, root=args.data_dir, video_backend="pyav")
    raw_env = gym.make(
        ENV_ID, sim_mode=args.sim_mode, render_hz=ds.meta.fps,
        headless=True, sonic_config=sonic_config,
    )
    task = raw_env.unwrapped.task
    horizon = args.horizon or task.metadata.get("max_episode_steps")
    env = TimeLimit(raw_env, max_episode_steps=horizon)
    robot = task.robot
    agent = Psi0DecoupledWbcAgent(robot, args.host, args.port, sonic_config=sonic_config)
    dr_level = Path(args.data_dir).name

    cells: list[tuple[float, str]] = [(0.0, "none")]
    for force in args.forces:
        for direction in ("left", "right"):
            cells.append((force, direction))

    hand_bodies_cache: set[int] | None = None

    for eps_idx in args.episodes:
        config_id = f"{dr_level}:{eps_idx}"
        recon = args.recon_by_config.get(eps_idx)
        if recon is None:
            print(f"[skip] {config_id}: not in eligible reconnaissance panel", flush=True)
            continue
        tape_path = tape_dir / f"{dr_level}_cfg{eps_idx}.json"

        for force_n, direction in cells:
            for condition in CONDITIONS:
                key = (config_id, force_n, direction, condition)
                if key in done:
                    continue
                is_canonical = force_n == 0.0 and condition == "fresh"
                if not is_canonical and not tape_path.exists():
                    print(f"[wait] {key}: canonical tape missing", flush=True)
                    continue
                tape = None if is_canonical else json.loads(tape_path.read_text())
                t0 = _real_time.perf_counter()

                random.seed(args.config_seed_base + eps_idx)
                np.random.seed(args.config_seed_base + eps_idx)
                env_conf, _ = get_episode_lerobot(ds, eps_idx)
                observation, info = env.reset(options={"state_dict": env_conf})
                sonic_env = raw_env.unwrapped
                model, data = sonic_env.mjModel, sonic_env.mjData

                target_body_id, _diag = resolve_target_body_id(raw_env, task, model)
                if hand_bodies_cache is None:
                    hand_bodies_cache = right_hand_body_set(model, RIGHT_HAND_ROOT)

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

                push_tick = None if is_canonical else int(tape["push_tick"])
                pusher = (
                    Pusher(model, data, force_n, direction, push_tick, control_dt)
                    if (push_tick is not None and force_n > 0.0) else None
                )

                if condition == "vla_replay":
                    vla_seq = extend(tape["vla"], horizon)
                    for row in vla_seq:
                        agent.queue_action(build_vla_cmd(row))

                vla_tape: list[dict[str, Any]] = []
                base_xy_trace: list[list[float]] = []
                init_height = target_height(info)
                server_queries = 0
                frame_idx = 0
                exhausted = False
                episode_over = False

                while not episode_over and frame_idx < horizon:
                    if pusher is not None:
                        pusher.step(frame_idx)
                    n_queued = len(agent._action_queue)
                    try:
                        action_cmd = agent.get_action(
                            observation, info=info, instruction=task.instruction
                        )
                    except StopIteration:
                        exhausted = True
                        break
                    if n_queued == 0:
                        server_queries += 1
                    if is_canonical and _seen["vla"] is not None:
                        pose = _seen["vla"]["target_upper_body_pose"]
                        vla_tape.append({
                            "target_upper_body_pose": {str(k): float(v) for k, v in pose.items()},
                            "navigate_cmd": [float(v) for v in np.asarray(_seen["vla"]["navigate_cmd"])],
                            "base_height_command": [float(v) for v in np.asarray(_seen["vla"]["base_height_command"])],
                        })
                        _seen["vla"] = None

                    observation, _r, terminated, truncated, info = env.step(action_cmd)
                    episode_over = terminated or truncated
                    frame_idx += 1
                    base_xy_trace.append([float(data.qpos[0]), float(data.qpos[1])])

                if pusher is not None:
                    data.xfrc_applied[pusher.body] = np.zeros(6)
                official_success = bool(raw_env.unwrapped._success)
                final_height = target_height(info)

                if is_canonical:
                    push_tick_for_tape = recon["push_tick"]
                    tape_path.write_text(json.dumps({
                        "env_id": ENV_ID, "config_id": config_id,
                        "clock": "virtual", "stabilize_steps": stabilize_steps,
                        "steps": frame_idx, "success": official_success,
                        "push_tick": push_tick_for_tape,
                        "vla": vla_tape,
                        "base_xy": base_xy_trace,
                    }))

                # How far the push moved the robot from where the canonical
                # (unperturbed) trajectory was at the same tick.
                displacement = None
                if pusher is not None and tape is not None and tape.get("base_xy"):
                    canon_xy = tape["base_xy"]
                    end = min(pusher.tick + pusher.n_ticks, len(base_xy_trace), len(canon_xy)) - 1
                    if end > pusher.tick:
                        displacement = float(np.linalg.norm(
                            np.asarray(base_xy_trace[end]) - np.asarray(canon_xy[end])
                        ))

                base_now = [float(data.qpos[0]), float(data.qpos[1])]
                row = {
                    "task": "xmove_bend_pick", "env_id": ENV_ID,
                    "config_id": config_id, "dr_level": dr_level,
                    "condition": condition, "force_n": force_n, "direction": direction,
                    "success": official_success,
                    "steps": frame_idx, "horizon": horizon,
                    "tape_recorded_len": (len(tape["vla"]) if tape else len(vla_tape)),
                    "tape_exhausted_early": exhausted,
                    "server_queries": server_queries,
                    "stabilize_steps": stabilize_steps,
                    "push_tick": push_tick,
                    "push_applied": bool(pusher is not None and pusher.applied_ticks > 0),
                    "push_applied_ticks": (pusher.applied_ticks if pusher else 0),
                    "push_displacement_m": displacement,
                    "init_target_height": init_height,
                    "final_target_height": final_height,
                    "final_target_lift_m": (
                        (final_height - init_height) if (final_height is not None and init_height is not None) else None
                    ),
                    "base_xy_final": base_now,
                    "clock": "virtual",
                    "seconds": round(_real_time.perf_counter() - t0, 1),
                }
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"[{config_id}/{force_n:.0f}N{direction}/{condition}] "
                    f"success={official_success} steps={frame_idx} q={server_queries} "
                    f"push@{push_tick} applied={row['push_applied_ticks']} "
                    f"lift={row['final_target_lift_m']} ({row['seconds']}s)",
                    flush=True,
                )

    raw_env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--recon-file", required=True,
                    help="canonical_reconnaissance.jsonl, to read this DR level's push_tick per config")
    p.add_argument("--out", required=True)
    p.add_argument("--tape-dir", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=26085)
    p.add_argument("--sim-mode", default="mujoco_isaac")
    p.add_argument("--forces", type=float, nargs="+", default=[50.0, 100.0, 150.0])
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--config-seed-base", type=int, default=20260824)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()

    dr_level = Path(args.data_dir).name
    recon_by_config: dict[int, dict] = {}
    for line in Path(args.recon_file).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["dr_level"] != dr_level or not r["timing_eligible"]:
            continue
        eps_idx = int(r["config_id"].split(":")[1])
        recon_by_config[eps_idx] = r
    args.recon_by_config = recon_by_config
    args.episodes = [
        i for i in range(args.episode_start, min(10, args.episode_start + args.num_episodes))
        if i in recon_by_config
    ]
    print(f"[recon] {len(recon_by_config)} eligible configs on {dr_level}; "
          f"running episodes {args.episodes}", flush=True)
    run(args)


if __name__ == "__main__":
    main()
