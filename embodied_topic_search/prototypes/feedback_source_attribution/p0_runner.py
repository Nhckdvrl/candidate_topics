"""P0 replay-fidelity runner for the feedback-source attribution candidate.

This is an **instrument test**, not a scientific experiment. No perturbation is
applied anywhere in this file. The only question it can answer is:

    when the world is not disturbed at all, does replaying a recorded command
    tape at either causal seam reproduce the live system's behaviour?

Two seams exist in the deployed Psi0 + SIMPLE stack (see
`simple/baselines/psi0_decoupled_wbc.py`):

    VLA  -> WBC       ActionCmd("vla_cmd",  target_upper_body_pose, navigate_cmd,
                                base_height_command)
    WBC  -> actuator  ActionCmd("decoupled_wbc", target_q, left_hand_q, right_hand_q)

Conditions
----------
fresh            live policy -> live WBC -> robot; records both tapes.
vla_replay       recorded vla_cmd tape -> live WBC (live proprioception) -> robot.
actuator_replay  recorded post-WBC tape -> actuator servo -> robot.

Clock
-----
The whole-body controller is a wall-clock interpolator: `set_goal` stamps
`target_time = time.monotonic() + 1/control_freq` and `get_action(time=...)`
samples the spline at the real time of the call. Under `--clock real` the
policy-server latency present in `fresh` but absent in the replay conditions
would change the WBC's own output, which would confound the replay comparison
with a timing artefact. `--clock virtual` (default) advances a monotonic
surrogate by exactly one control period per WBC invocation in every condition,
i.e. the nominal real-time schedule the controller was designed for. Both modes
are implemented so the choice can be measured rather than asserted.

P0b: seam liveness
------------------
P0 as frozen can only show that the tapes are lossless. With no disturbance the
state never leaves the recorded trajectory, so a whole-body controller that
ignored proprioception entirely would replay just as exactly. That would make the
later `vla_replay - actuator_replay` gap structurally zero and uninterpretable —
the same class of failure that killed Topic 23, where the treatment did not remove
the object the claim named.

`--push-force` therefore drives an instrument-only measurement: with a fixed VLA
tape, does the live whole-body controller's commanded `target_q` actually diverge
from the recorded one when the body is off-nominal? That divergence
(`max_cmd_dev_rad`) is a property of the seam, not of the hypothesis, and it is
reported without reference to task success.

Frozen upstream:
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

# Task-effect predicate, copied from the frozen Topic 23 contract.
TASKS = {
    "simple/G1WholebodyCloseDoorTeleop-v0": {
        "short_name": "close_door",
        "effect_joint": "articulate_joint_1",
        "predicate": lambda q: q < -0.16,
    },
    "simple/G1WholebodyOpenFaucetTeleop-v0": {
        "short_name": "open_faucet",
        "effect_joint": "articulate_joint_0",
        "predicate": lambda q: abs(q) > 0.7,
    },
}


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
class VirtualClock:
    """A monotonic surrogate advancing one control period per call.

    Installed in place of the `time` module inside the two agent modules that
    drive the whole-body controller. Every condition then sees the identical
    control timeline, so a replay difference cannot be a scheduling artefact.
    """

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
# Tape
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


# ---------------------------------------------------------------------------
def _make_sonic_config() -> dict[str, Any]:
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def door_body_set(model, effect_joint: str) -> set[int]:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, effect_joint)
    if jid < 0:
        raise RuntimeError(f"effect joint {effect_joint!r} not in model")
    root = int(model.jnt_bodyid[jid])
    out = set()
    for b in range(model.nbody):
        x = b
        while x > 0:
            if x == root:
                out.add(b)
                break
            x = int(model.body_parentid[x])
    return out


def touching_object(model, data, door_bodies: set[int]) -> bool:
    geom_body = np.asarray(model.geom_bodyid)
    for i in range(data.ncon):
        c = data.contact[i]
        in1 = int(geom_body[c.geom1]) in door_bodies
        in2 = int(geom_body[c.geom2]) in door_bodies
        if in1 != in2:
            return True
    return False


def base_yaw(data) -> float:
    w, x, y, z = (float(v) for v in np.asarray(data.qpos[3:7], dtype=float))
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def state_digest(data) -> dict[str, float]:
    qpos = np.asarray(data.qpos, dtype=float)
    qvel = np.asarray(data.qvel, dtype=float)
    return {
        "qpos_l1": float(np.abs(qpos).sum()),
        "qvel_l1": float(np.abs(qvel).sum()),
        "base_xyz": [float(v) for v in qpos[:3]],
    }


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
    from simple.envs.wrappers.video_recorder import VideoRecorder

    spec = TASKS[args.env_id]
    sonic_config = _make_sonic_config()
    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt

    clock = VirtualClock(control_dt) if args.clock == "virtual" else RealClock()
    # The whole-body controller reads wall time through these two modules only.
    sonic_mod.time = clock
    psi0_mod.time = clock

    # Capture the popped `vla_cmd` at the VLA -> WBC seam.
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

    ds = LeRobotDataset(repo_id=args.env_id, root=args.data_dir, video_backend="pyav")
    render_hz = ds.meta.fps
    print(f"dataset {args.data_dir}: {ds.num_episodes} episodes @ {render_hz} Hz", flush=True)

    raw_env = gym.make(
        args.env_id, sim_mode=args.sim_mode, render_hz=render_hz,
        headless=True, sonic_config=sonic_config,
    )
    task = raw_env.unwrapped.task
    max_steps = args.max_episode_steps or task.metadata.get("max_episode_steps")
    env_t = TimeLimit(raw_env, max_episode_steps=max_steps)
    task.success_criteria = args.success_criteria
    task.metadata["success_criteria"] = args.success_criteria
    robot = task.robot

    agent = Psi0DecoupledWbcAgent(robot, args.host, args.port, sonic_config=sonic_config)
    video_root = Path(args.video_dir)

    dr_level = Path(args.data_dir).name
    for eps_idx in range(args.episode_start, min(ds.num_episodes, args.episode_start + args.num_episodes)):
        config_id = f"{dr_level}:{eps_idx}"
        tape_path = tape_dir / f"{dr_level}_cfg{eps_idx}.json"

        for condition in args.conditions:
            key = (config_id, condition)
            if key in done_keys:
                print(f"[skip] {key}", flush=True)
                continue
            if not tape_path.exists() and (condition != "fresh" or args.push_force > 0.0):
                # Replay needs the commands; a perturbed `fresh` still needs the
                # canonical rollout to place the push. Record the nominal pass first.
                print(f"[wait] {key}: no canonical tape at {tape_path}", flush=True)
                continue

            # `fresh` consults the tape only for push timing, never for commands.
            tape = json.loads(tape_path.read_text()) if tape_path.exists() else None
            t0 = _real_time.perf_counter()

            # Matched configs: identical scene draw for every condition.
            random.seed(args.config_seed_base + eps_idx)
            np.random.seed(args.config_seed_base + eps_idx)
            env_conf, _episode = get_episode_lerobot(ds, eps_idx)

            env = env_t
            if args.save_video:
                video_root.mkdir(parents=True, exist_ok=True)
                env = VideoRecorder(
                    env=env_t, video_folder=str(video_root),
                    name_prefix=f"{dr_level}_cfg{eps_idx}_{condition}",
                    framerate=render_hz, write_png=False,
                )

            observation, info = env.reset(options={"state_dict": env_conf})
            sonic_env = raw_env.unwrapped
            model, data = sonic_env.mjModel, sonic_env.mjData
            import mujoco as _mj

            effect_joint = spec["effect_joint"]
            door_bodies = door_body_set(model, effect_joint)

            clock.reset()
            agent.reset()
            agent._wbc_policy.lower_body_policy.use_policy_action = True

            # --- stabilization: identical procedure in every condition --------
            stabilize_steps = 0
            while not robot.stabilized and stabilize_steps < 400:
                action = agent.get_stabilize_action(observation)
                observation, *_, info = sonic_env.step(action)
                sonic_env.update_viewer()
                sonic_env.update_reward()
                stabilize_steps += 1
            if isinstance(env, VideoRecorder):
                env._init_writers(observation)
            settled = state_digest(data)
            agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1,), dtype=torch.float32)

            # --- pre-load the VLA tape for the vla_replay condition ------------
            if condition == "vla_replay":
                for row in tape["vla"]:
                    agent.queue_action(build_vla_cmd(row))

            # Push timing comes only from this config's unperturbed canonical
            # rollout: `push_lead_s` before the first robot/object contact. The
            # perturbed outcome is never consulted.
            push_lo = push_hi = None
            if args.push_force > 0.0:
                fc = tape.get("first_contact_step")
                if fc is None:
                    raise RuntimeError(
                        f"{config_id}: tape has no first_contact_step; re-record fresh"
                    )
                push_lo = max(0, int(fc) - int(round(args.push_lead_s / control_dt)))
                push_hi = push_lo + int(round(args.push_duration_s / control_dt))
            push_bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, args.push_body)
            if push_bid < 0:
                raise RuntimeError(f"push body {args.push_body!r} not in model")
            # `G1Sonic` writes the elastic-band force into `xfrc_applied` on the
            # same body each step; a push written there would be silently erased.
            band_link = getattr(robot, "band_attached_link", None)
            if args.push_force > 0.0 and band_link is not None and int(band_link) == push_bid:
                raise RuntimeError(
                    f"push body {args.push_body!r} is the elastic-band link; the "
                    "band would overwrite the disturbance"
                )

            vla_tape: list[dict[str, Any]] = []
            act_tape: list[dict[str, Any]] = []
            door_trace: list[float] = []
            base_trace: list[list[float]] = []
            first_effect_step = None
            first_contact_step = None
            cmd_dev = 0.0
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
                    if condition == "vla_replay" and frame_idx < len(tape["actuator"]):
                        # Seam liveness: how far the live controller's command
                        # moves away from the one it issued on the nominal run.
                        ref = np.asarray(tape["actuator"][frame_idx]["target_q"])
                        cmd_dev = max(cmd_dev, float(np.max(np.abs(
                            np.asarray(action_cmd["target_q"], dtype=float) - ref
                        ))))

                if push_lo is not None and push_lo <= frame_idx < push_hi:
                    yaw = base_yaw(data)
                    sign = 1.0 if args.push_dir == "left" else -1.0
                    # Lateral in the robot's own frame, so the disturbance means
                    # the same thing regardless of where the base happens to face.
                    data.xfrc_applied[push_bid, :3] = np.array([
                        -sign * args.push_force * np.sin(yaw),
                        sign * args.push_force * np.cos(yaw),
                        0.0,
                    ])
                else:
                    data.xfrc_applied[push_bid, :3] = 0.0

                observation, _reward, terminated, truncated, info = env.step(action_cmd)
                episode_over = terminated or truncated
                frame_idx += 1

                door_q = float(data.joint(effect_joint).qpos[0])
                door_trace.append(door_q)
                base_trace.append([float(data.qpos[0]), float(data.qpos[1])])
                if first_effect_step is None and spec["predicate"](door_q):
                    first_effect_step = frame_idx
                if first_contact_step is None and touching_object(model, data, door_bodies):
                    first_contact_step = frame_idx

            official_success = bool(raw_env.unwrapped._success)
            terminal = state_digest(data)
            terminal_qpos = [float(v) for v in np.asarray(data.qpos, dtype=float)]

            if condition == "fresh" and args.push_force == 0.0:
                if len(vla_tape) != len(act_tape):
                    raise RuntimeError(
                        f"seam desync: {len(vla_tape)} vla rows vs {len(act_tape)} actuator rows"
                    )
                tape_path.write_text(json.dumps({
                    "env_id": args.env_id,
                    "config_id": config_id,
                    "clock": args.clock,
                    "stabilize_steps": stabilize_steps,
                    "settled": settled,
                    "first_contact_step": first_contact_step,
                    "steps": frame_idx,
                    "success": official_success,
                    "vla": vla_tape,
                    "actuator": act_tape,
                }))

            row = {
                "task": spec["short_name"],
                "env_id": args.env_id,
                "config_id": config_id,
                "dr_level": dr_level,
                "condition": condition,
                "force_n": float(args.push_force),
                "direction": args.push_dir if args.push_force > 0.0 else "none",
                "success": official_success,
                "steps": frame_idx,
                "tape_len": (
                    len(tape["actuator"]) if (tape and condition != "fresh") else len(act_tape)
                ),
                "tape_exhausted_early": exhausted,
                "server_queries": server_queries,
                "stabilize_steps": stabilize_steps,
                "settled_qpos_l1": settled["qpos_l1"],
                "settled_qvel_l1": settled["qvel_l1"],
                "effect_qpos": door_trace[-1] if door_trace else None,
                "effect_predicate_reached": first_effect_step is not None,
                "first_effect_step": first_effect_step,
                "first_contact_step": first_contact_step,
                "max_cmd_dev_rad": cmd_dev,
                "push_window": [push_lo, push_hi] if push_lo is not None else None,
                "door_q_init": door_trace[0] if door_trace else None,
                "door_q_min": min(door_trace) if door_trace else None,
                "door_q_max": max(door_trace) if door_trace else None,
                "terminal_qpos_l1": terminal["qpos_l1"],
                "terminal_base_xyz": terminal["base_xyz"],
                "clock": args.clock,
                "seconds": round(_real_time.perf_counter() - t0, 1),
            }
            # Trajectory-level fidelity: `success` alone cannot distinguish a
            # faithful replay from one that diverges and still happens to finish.
            trace_path = tape_dir / f"{dr_level}_cfg{eps_idx}_{condition}_trace.json"
            trace_path.write_text(json.dumps({
                "config_id": config_id,
                "condition": condition,
                "door_trace": door_trace,
                "base_trace": base_trace,
                "terminal_qpos": terminal_qpos,
            }))
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"[{config_id}/{condition}] success={official_success} steps={frame_idx}"
                f"/{row['tape_len']} exhausted={exhausted} queries={server_queries} "
                f"door=[{row['door_q_min']:.3f},{row['door_q_max']:.3f}] "
                f"stab={stabilize_steps} qpos_l1={settled['qpos_l1']:.4f} ({row['seconds']}s)",
                flush=True,
            )
            if isinstance(env, VideoRecorder):
                env.release()

    raw_env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="simple/G1WholebodyCloseDoorTeleop-v0")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tape-dir", required=True)
    p.add_argument("--video-dir", default="videos")
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
    p.add_argument("--push-force", type=float, default=0.0,
                   help="lateral disturbance in newtons; 0 = P0 (no perturbation)")
    p.add_argument("--push-dir", choices=("left", "right"), default="left")
    p.add_argument("--push-duration-s", type=float, default=0.2)
    p.add_argument("--push-lead-s", type=float, default=1.0)
    p.add_argument("--push-body", default="torso_link")
    p.add_argument("--save-video", action="store_true", default=False)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    run(p.parse_args())


if __name__ == "__main__":
    main()
