"""P0' replay-fidelity runner for XMoveBendPickTeleop, cross-task re-verification.

This is a fresh instrument test on a new task, not an inherited assumption from
CloseDoor's P0. No perturbation is applied anywhere in this file.

XMoveBendPickTeleop's success predicate is not a single hinge joint like
CloseDoor's door. Per the task source
(`simple/tasks/g1_wholebody_xmove_bend_pick_teleop.py`), success is:

    reward = clip((target_height - initial_target_height) / 0.1, 0, 1) >= 0.8
    target_height = info["target"][2]

`info["target"]` is returned by `env.step()`/`env.reset()` directly (the
task's own `compute_reward` reads it the same way), so no MuJoCo body lookup
is needed to track it descriptively here. `env.unwrapped._success` remains the
frozen success signal, unchanged from every other Topic 24 panel.

P0' deliberately runs on `dr-level-0` only. The released Psi0 checkpoint's
published upstream competence on this task is `10 | 9 | 9` across the three DR
levels (SIMPLE README benchmark table) -- unlike CloseDoor, this policy is not
10/10 everywhere, so a >=0.90 fidelity gate would misread the upstream
policy's own natural failures on dr-level-1/2 as an instrument failure.
dr-level-0's own upstream number is 10/10, so it is the level where a fidelity
gate is actually clean. Cross-DR competence is a separate question for the
canonical-reconnaissance step, not conflated with instrument validation here.

Horizon is read from the task's own `max_episode_steps` (800 for this task,
not CloseDoor's 450) via `task.metadata`, exactly as the shared runner already
does -- not hand-copied from another task.

Frozen upstream, same as every other Topic 24 panel:
    SIMPLE b49c1aea2dd57309bb533219d0d34d6020f3d943
    Psi0   9ad917526394c1cacc72dba08562629936505987
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

CONDITIONS = ("fresh", "vla_replay", "actuator_replay")
ENV_ID = "simple/G1WholebodyXMoveBendPickTeleop-v0"


# ---------------------------------------------------------------------------
# Clock (identical to the shared p0_runner.py)
# ---------------------------------------------------------------------------
class VirtualClock:
    def __init__(self, dt: float, t0: float = 10_000.0) -> None:
        self.dt = float(dt)
        self.t0 = float(t0)
        self.t = float(t0)
        self.n_calls = 0

    def reset(self) -> None:
        self.t = self.t0
        self.n_calls = 0

    def monotonic(self) -> float:
        t = self.t
        self.t += self.dt
        self.n_calls += 1
        return t


class RealClock:
    def __init__(self) -> None:
        self.n_calls = 0

    def reset(self) -> None:
        self.n_calls = 0

    def monotonic(self) -> float:
        self.n_calls += 1
        return _real_time.monotonic()


# ---------------------------------------------------------------------------
# Tape (identical to the shared p0_runner.py -- these seams are task-agnostic)
# ---------------------------------------------------------------------------
def _f(x) -> list[float]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(a)):
        raise ValueError("command contains non-finite values")
    return [float(v) for v in a]


def record_vla(cmd) -> dict[str, Any]:
    pose = cmd["target_upper_body_pose"]
    return {
        "target_upper_body_pose": {str(k): float(v) for k, v in pose.items()},
        "navigate_cmd": _f(cmd["navigate_cmd"]),
        "base_height_command": _f(cmd["base_height_command"]),
    }


def record_act(cmd) -> dict[str, Any]:
    return {
        "target_q": _f(cmd["target_q"]),
        "left_hand_q": _f(cmd["left_hand_q"]),
        "right_hand_q": _f(cmd["right_hand_q"]),
    }


def build_vla_cmd(row: dict[str, Any]):
    from simple.core.action import ActionCmd

    return ActionCmd(
        "vla_cmd",
        target_upper_body_pose=dict(row["target_upper_body_pose"]),
        navigate_cmd=np.asarray(row["navigate_cmd"], dtype=np.float32),
        base_height_command=np.asarray(row["base_height_command"], dtype=np.float32),
    )


def build_act_cmd(row: dict[str, Any]):
    from simple.core.action import ActionCmd

    return ActionCmd(
        "decoupled_wbc",
        target_q=np.asarray(row["target_q"], dtype=np.float64),
        left_hand_q=np.asarray(row["left_hand_q"], dtype=np.float64),
        right_hand_q=np.asarray(row["right_hand_q"], dtype=np.float64),
    )


def _make_sonic_config() -> dict[str, Any]:
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def state_digest(data) -> dict[str, float]:
    qpos = np.asarray(data.qpos, dtype=float)
    qvel = np.asarray(data.qvel, dtype=float)
    return {
        "qpos_l1": float(np.abs(qpos).sum()),
        "qvel_l1": float(np.abs(qvel).sum()),
        "base_xyz": [float(v) for v in qpos[:3]],
    }


def target_height(info: dict[str, Any]) -> float | None:
    """Same field the task's own compute_reward reads; no MuJoCo lookup needed."""
    t = info.get("target")
    return float(t[2]) if t is not None else None


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch
    from gymnasium.wrappers import TimeLimit
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import simple.envs as _  # noqa: F401
    import simple.agents.sonic_decoupled_wbc_agent as sonic_mod
    import simple.baselines.psi0_decoupled_wbc as psi0_mod
    from simple.datasets.lerobot import get_episode_lerobot
    from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
    from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent

    sonic_config = _make_sonic_config()
    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt
    print(f"[cadence] sim_dt={sim_dt} control_dt={control_dt} (={1/control_dt:.1f} Hz)", flush=True)

    clock = VirtualClock(control_dt) if args.clock == "virtual" else RealClock()
    sonic_mod.time = clock
    psi0_mod.time = clock

    _seen: dict[str, Any] = {"vla": None}
    _orig_get_action = SonicDecoupledWbcAgent.get_action

    def _hooked(self, observation, instruction=None, **kwargs):
        cmd = _orig_get_action(self, observation, instruction, **kwargs)
        if cmd is not None and cmd.type == "vla_cmd":
            _seen["vla"] = cmd
        return cmd

    SonicDecoupledWbcAgent.get_action = _hooked

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tape_dir = Path(args.tape_dir)
    tape_dir.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    if out_path.exists() and args.resume:
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["config_id"], r["condition"]))
        print(f"[resume] {len(done_keys)} records already present", flush=True)

    ds = LeRobotDataset(repo_id=ENV_ID, root=args.data_dir, video_backend="pyav")
    render_hz = ds.meta.fps
    print(f"dataset {args.data_dir}: {ds.num_episodes} episodes @ {render_hz} Hz", flush=True)

    raw_env = gym.make(
        ENV_ID, sim_mode=args.sim_mode, render_hz=render_hz,
        headless=True, sonic_config=sonic_config,
    )
    task = raw_env.unwrapped.task
    max_steps = args.max_episode_steps or task.metadata.get("max_episode_steps")
    print(f"[task] max_episode_steps from task.metadata = {max_steps}", flush=True)
    env = TimeLimit(raw_env, max_episode_steps=max_steps)
    task.success_criteria = args.success_criteria
    task.metadata["success_criteria"] = args.success_criteria
    robot = task.robot

    agent = Psi0DecoupledWbcAgent(robot, args.host, args.port, sonic_config=sonic_config)

    dr_level = Path(args.data_dir).name
    for eps_idx in range(args.episode_start, min(ds.num_episodes, args.episode_start + args.num_episodes)):
        config_id = f"{dr_level}:{eps_idx}"
        tape_path = tape_dir / f"{dr_level}_cfg{eps_idx}.json"

        for condition in args.conditions:
            key = (config_id, condition)
            if key in done_keys:
                print(f"[skip] {key}", flush=True)
                continue
            if condition != "fresh" and not tape_path.exists():
                print(f"[wait] {key}: no canonical tape at {tape_path}", flush=True)
                continue

            tape = json.loads(tape_path.read_text()) if condition != "fresh" else None
            t0 = _real_time.perf_counter()

            random.seed(args.config_seed_base + eps_idx)
            np.random.seed(args.config_seed_base + eps_idx)
            env_conf, _episode = get_episode_lerobot(ds, eps_idx)

            observation, info = env.reset(options={"state_dict": env_conf})
            sonic_env = raw_env.unwrapped
            data = sonic_env.mjData

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
            settled = state_digest(data)
            agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1,), dtype=torch.float32)

            if condition == "vla_replay":
                for row in tape["vla"]:
                    agent.queue_action(build_vla_cmd(row))

            vla_tape: list[dict[str, Any]] = []
            act_tape: list[dict[str, Any]] = []
            height_trace: list[float] = []
            init_height = target_height(info)
            server_queries = 0
            frame_idx = 0
            episode_over = False
            instruction = task.instruction
            exhausted = False

            while not episode_over:
                if condition == "actuator_replay":
                    if frame_idx >= len(tape["actuator"]):
                        exhausted = True
                        break
                    action_cmd = build_act_cmd(tape["actuator"][frame_idx])
                else:
                    n_queued = len(agent._action_queue)
                    try:
                        action_cmd = agent.get_action(
                            observation, info=info, instruction=instruction
                        )
                    except StopIteration:
                        exhausted = True
                        break
                    if n_queued == 0:
                        server_queries += 1
                    if _seen["vla"] is not None:
                        vla_tape.append(record_vla(_seen["vla"]))
                        _seen["vla"] = None
                    act_tape.append(record_act(action_cmd))

                observation, _reward, terminated, truncated, info = env.step(action_cmd)
                episode_over = terminated or truncated
                frame_idx += 1

                h = target_height(info)
                if h is not None:
                    height_trace.append(h)

            official_success = bool(raw_env.unwrapped._success)
            terminal = state_digest(data)

            if condition == "fresh":
                if len(vla_tape) != len(act_tape):
                    raise RuntimeError(
                        f"seam desync: {len(vla_tape)} vla rows vs {len(act_tape)} actuator rows"
                    )
                tape_path.write_text(json.dumps({
                    "env_id": ENV_ID,
                    "config_id": config_id,
                    "clock": args.clock,
                    "stabilize_steps": stabilize_steps,
                    "settled": settled,
                    "steps": frame_idx,
                    "success": official_success,
                    "init_target_height": init_height,
                    "vla": vla_tape,
                    "actuator": act_tape,
                }))

            lift = (
                (height_trace[-1] - init_height)
                if (height_trace and init_height is not None) else None
            )
            row = {
                "task": "xmove_bend_pick", "env_id": ENV_ID,
                "config_id": config_id, "dr_level": dr_level,
                "condition": condition, "force_n": 0.0, "direction": "none",
                "success": official_success,
                "steps": frame_idx,
                "tape_len": (len(tape["actuator"]) if tape else len(act_tape)),
                "tape_exhausted_early": exhausted,
                "server_queries": server_queries,
                "stabilize_steps": stabilize_steps,
                "settled_qpos_l1": settled["qpos_l1"],
                "settled_qvel_l1": settled["qvel_l1"],
                "init_target_height": init_height,
                "final_target_lift_m": lift,
                "terminal_qpos_l1": terminal["qpos_l1"],
                "terminal_base_xyz": terminal["base_xyz"],
                "clock": args.clock,
                "seconds": round(_real_time.perf_counter() - t0, 1),
            }
            trace_path = tape_dir / f"{dr_level}_cfg{eps_idx}_{condition}_trace.json"
            trace_path.write_text(json.dumps({
                "config_id": config_id, "condition": condition,
                "height_trace": height_trace,
            }))
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"[{config_id}/{condition}] success={official_success} steps={frame_idx}"
                f"/{row['tape_len']} exhausted={exhausted} queries={server_queries} "
                f"lift={lift} stab={stabilize_steps} ({row['seconds']}s)",
                flush=True,
            )

    raw_env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tape-dir", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=22085)
    p.add_argument("--sim-mode", default="mujoco_isaac")
    p.add_argument("--clock", choices=("virtual", "real"), default="virtual")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--success-criteria", type=float, default=0.5)
    p.add_argument("--config-seed-base", type=int, default=20260824)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    run(p.parse_args())


if __name__ == "__main__":
    main()
