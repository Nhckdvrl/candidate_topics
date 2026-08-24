"""Topic 24 G0 runner: three-level feedback attribution under a physical push.

Frozen upstream:
    SIMPLE b49c1aea2dd57309bb533219d0d34d6020f3d943
    Psi0   9ad917526394c1cacc72dba08562629936505987

The two cut points are the seams that exist in the released source
(`simple/baselines/psi0_decoupled_wbc.py`), not probes we invented:

    VLA  -> WBC       ActionCmd("vla_cmd",  target_upper_body_pose, navigate_cmd,
                                base_height_command)
    WBC  -> actuator  ActionCmd("decoupled_wbc", target_q, left_hand_q, right_hand_q)

Conditions, all three receiving the identical disturbance at the identical tick:

    fresh            live obs -> live VLA -> live WBC -> robot
    vla_replay       recorded nominal vla_cmd tape -> live WBC (live proprio) -> robot
    actuator_replay  recorded nominal post-WBC tape -> actuator servo -> robot

Both replay interventions are the upstream data path rather than a
reimplementation of it. `vla_replay` pre-loads the recorded tape into
`SonicDecoupledWbcAgent._action_queue`, which the released agent only refills by
querying the policy server when it is empty, so the VLA leaves the loop while
`_build_wbc_observation` still runs on live proprioception every tick.
`server_queries == 0` on every replay row is the recorded proof.

Instrument prerequisites, both already passed before this topic was registered:
    P0   replay fidelity       10/10 in all three conditions, 0.0 divergence
    P0b  WBC seam liveness     D = 2.4e-02 .. 4.6e-02 rad over a zero floor
See `../embodied_topic_search/prototypes/feedback_source_attribution/`.
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

from g0_core import CONDITIONS, DIRECTIONS, FORCES_N, PUSH_DURATION_S  # noqa: E402

TASKS = {
    "simple/G1WholebodyCloseDoorTeleop-v0": {
        "short_name": "close_door",
        "effect_joint": "articulate_joint_1",
        "predicate": lambda q: q < -0.16,
    },
}

# Frozen timing rule. The push lands one second before the moment the canonical
# unperturbed rollout first touches the task object, so it disturbs the approach
# rather than the manipulation itself. Derived only from the unperturbed rollout;
# no perturbed outcome is ever inspected to choose it.
PUSH_LEAD_TICKS = 50          # 1.0 s at 50 Hz
PUSH_HORIZON = 450            # upstream max_episode_steps for this task


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
class VirtualClock:
    """Nominal 50 Hz control clock, identical in every condition.

    The whole-body controller interpolates against `time.monotonic()`, so
    policy-server latency is otherwise an input to the controller: present in
    `fresh`, absent in both replays. Measured on this exact task, the released
    real-clock stack is not even reproducible across runs of the same config,
    while this surrogate reproduces bit-identically. See
    `../embodied_topic_search/prototypes/feedback_source_attribution/P0_RESULTS.md`.
    """

    def __init__(self, dt: float, t0: float = 10_000.0) -> None:
        self.dt = float(dt)
        self.t0 = float(t0)
        self.t = float(t0)

    def reset(self) -> None:
        self.t = self.t0

    def monotonic(self) -> float:
        t = self.t
        self.t += self.dt
        return t


# ---------------------------------------------------------------------------
# Tape
# ---------------------------------------------------------------------------
def _f(x) -> list[float]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(a)):
        raise ValueError("command contains non-finite values")
    return [float(v) for v in a]


def record_vla(cmd) -> dict[str, Any]:
    return {
        "target_upper_body_pose": {str(k): float(v) for k, v in cmd["target_upper_body_pose"].items()},
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


def extend(tape: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Hold the final recorded command out to the shared horizon.

    All three conditions must get the same step budget, otherwise `fresh` could
    win simply by being allowed to run longer than the tape. Holding the last
    nominal command is the faithful reading of the intervention: the recorded
    plan ends and nothing new is issued.
    """
    if not tape:
        raise ValueError("empty tape")
    return list(tape) + [tape[-1]] * max(0, n - len(tape))


# ---------------------------------------------------------------------------
# Contact attribution (canonical pass only) and the push
# ---------------------------------------------------------------------------
class ContactProbe:
    """Which robot parts touch the task object, straight from MuJoCo contacts."""

    def __init__(self, model, data, effect_joint: str) -> None:
        import mujoco

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, effect_joint)
        if jid < 0:
            raise RuntimeError(f"effect joint {effect_joint!r} not in model")
        root = int(model.jnt_bodyid[jid])
        self.model, self.data = model, data
        self.obj_bodies = {b for b in range(model.nbody) if self._descends(b, root)}
        self.geom_body = np.asarray(model.geom_bodyid)

    def _descends(self, body: int, root: int) -> bool:
        b = body
        while b > 0:
            if b == root:
                return True
            b = int(self.model.body_parentid[b])
        return b == root

    def touching(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            in1 = int(self.geom_body[c.geom1]) in self.obj_bodies
            in2 = int(self.geom_body[c.geom2]) in self.obj_bodies
            if in1 != in2:
                return True
        return False


def base_yaw(data) -> float:
    w, x, y, z = np.asarray(data.qpos[3:7], dtype=float)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Pusher:
    """A frozen lateral shove on the torso, identical in every condition.

    The force is applied in the world frame along the robot's own lateral axis at
    the push tick, held for `PUSH_DURATION_S`, then cleared. It acts on the
    simulator, not on any command, so no condition can see it except through its
    physical consequences.
    """

    def __init__(self, model, data, force_n: float, direction: str, tick: int, control_dt: float):
        import mujoco

        self.data = data
        self.force_n = float(force_n)
        self.direction = direction
        self.tick = int(tick)
        self.n_ticks = int(round(PUSH_DURATION_S / control_dt))
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.body < 0:
            raise RuntimeError("torso_link not in model")
        self.applied_ticks = 0
        self.vec = np.zeros(3)

    def latch_direction(self) -> None:
        """Resolve 'left'/'right' against the robot's heading at the push tick."""
        yaw = base_yaw(self.data)
        sign = 1.0 if self.direction == "left" else -1.0
        # Lateral axis of the base frame in world coordinates.
        self.vec = sign * self.force_n * np.array([-np.sin(yaw), np.cos(yaw), 0.0])

    def step(self, frame_idx: int) -> None:
        if self.force_n <= 0.0:
            return
        if frame_idx == self.tick:
            self.latch_direction()
        active = self.tick <= frame_idx < self.tick + self.n_ticks
        f = np.zeros(6)
        if active:
            f[:3] = self.vec
            self.applied_ticks += 1
        self.data.xfrc_applied[self.body] = f


def _make_sonic_config() -> dict[str, Any]:
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


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
    from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
    from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent

    spec = TASKS[args.env_id]
    sonic_config = _make_sonic_config()
    control_dt = 4 * sonic_config["SIMULATE_DT"]
    clock = VirtualClock(control_dt)
    sonic_mod.time = clock
    psi0_mod.time = clock

    _seen: dict[str, Any] = {"vla": None}
    _orig = SonicDecoupledWbcAgent.get_action

    def _hooked(self, observation, instruction=None, **kwargs):
        cmd = _orig(self, observation, instruction, **kwargs)
        if cmd is not None and cmd.type == "vla_cmd":
            _seen["vla"] = cmd
        return cmd

    SonicDecoupledWbcAgent.get_action = _hooked

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

    ds = LeRobotDataset(repo_id=args.env_id, root=args.data_dir, video_backend="pyav")
    raw_env = gym.make(
        args.env_id, sim_mode=args.sim_mode, render_hz=ds.meta.fps,
        headless=True, sonic_config=sonic_config,
    )
    task = raw_env.unwrapped.task
    horizon = args.horizon or task.metadata.get("max_episode_steps") or PUSH_HORIZON
    env = TimeLimit(raw_env, max_episode_steps=horizon)
    task.success_criteria = args.success_criteria
    task.metadata["success_criteria"] = args.success_criteria
    robot = task.robot
    agent = Psi0DecoupledWbcAgent(robot, args.host, args.port, sonic_config=sonic_config)
    dr_level = Path(args.data_dir).name

    # Force cells, in a fixed order: the control column first, then the full grid.
    cells: list[tuple[float, str]] = [(0.0, "none")]
    for force in FORCES_N:
        for direction in DIRECTIONS:
            cells.append((force, direction))

    for eps_idx in range(args.episode_start, min(ds.num_episodes, args.episode_start + args.num_episodes)):
        config_id = f"{dr_level}:{eps_idx}"
        tape_path = tape_dir / f"{dr_level}_cfg{eps_idx}.json"

        for force_n, direction in cells:
            for condition in args.conditions:
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
                probe = ContactProbe(model, data, spec["effect_joint"]) if is_canonical else None

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

                vla_seq = act_seq = None
                if condition == "vla_replay":
                    vla_seq = extend(tape["vla"], horizon)
                    for row in vla_seq:
                        agent.queue_action(build_vla_cmd(row))
                elif condition == "actuator_replay":
                    act_seq = extend(tape["actuator"], horizon)

                vla_tape: list[dict[str, Any]] = []
                act_tape: list[dict[str, Any]] = []
                door_trace: list[float] = []
                base_xy: list[list[float]] = []
                first_contact_tick = None
                server_queries = 0
                frame_idx = 0
                exhausted = False
                episode_over = False

                while not episode_over and frame_idx < horizon:
                    if pusher is not None:
                        pusher.step(frame_idx)
                    if condition == "actuator_replay":
                        if frame_idx >= len(act_seq):
                            exhausted = True
                            break
                        action_cmd = build_act_cmd(act_seq[frame_idx])
                    else:
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
                        if is_canonical:
                            if _seen["vla"] is not None:
                                vla_tape.append(record_vla(_seen["vla"]))
                                _seen["vla"] = None
                            act_tape.append(record_act(action_cmd))

                    observation, _r, terminated, truncated, info = env.step(action_cmd)
                    episode_over = terminated or truncated
                    frame_idx += 1

                    door_trace.append(float(data.joint(spec["effect_joint"]).qpos[0]))
                    base_xy.append([float(v) for v in np.asarray(data.qpos[:2], dtype=float)])
                    if probe is not None and first_contact_tick is None and probe.touching():
                        first_contact_tick = frame_idx

                if pusher is not None:
                    data.xfrc_applied[pusher.body] = np.zeros(6)
                official_success = bool(raw_env.unwrapped._success)

                if is_canonical:
                    if len(vla_tape) != len(act_tape):
                        raise RuntimeError(
                            f"seam desync: {len(vla_tape)} vla vs {len(act_tape)} actuator rows"
                        )
                    if first_contact_tick is None:
                        print(f"[warn] {config_id}: no object contact in canonical rollout", flush=True)
                    contact = first_contact_tick if first_contact_tick is not None else len(act_tape)
                    tape_path.write_text(json.dumps({
                        "env_id": args.env_id, "config_id": config_id,
                        "clock": "virtual", "stabilize_steps": stabilize_steps,
                        "steps": frame_idx, "success": official_success,
                        "first_contact_tick": first_contact_tick,
                        "push_tick": max(0, contact - PUSH_LEAD_TICKS),
                        "push_lead_ticks": PUSH_LEAD_TICKS,
                        "base_xy": base_xy,
                        "vla": vla_tape, "actuator": act_tape,
                    }))

                # How far the push moved the robot away from where the canonical
                # rollout was at the same tick. Structural check on the intervention.
                displacement = None
                if pusher is not None and tape is not None:
                    end = min(pusher.tick + pusher.n_ticks, len(base_xy), len(tape["base_xy"])) - 1
                    if end > pusher.tick:
                        displacement = float(np.linalg.norm(
                            np.asarray(base_xy[end]) - np.asarray(tape["base_xy"][end])
                        ))

                row = {
                    "task": spec["short_name"], "env_id": args.env_id,
                    "config_id": config_id, "dr_level": dr_level,
                    "condition": condition, "force_n": force_n, "direction": direction,
                    "success": official_success,
                    "steps": frame_idx, "horizon": horizon,
                    "tape_recorded_len": (len(tape["actuator"]) if tape else len(act_tape)),
                    "tape_exhausted_early": exhausted,
                    "server_queries": server_queries,
                    "stabilize_steps": stabilize_steps,
                    "push_tick": push_tick,
                    "push_applied": bool(pusher is not None and pusher.applied_ticks > 0),
                    "push_applied_ticks": (pusher.applied_ticks if pusher else 0),
                    "push_displacement_m": displacement,
                    "first_contact_tick": first_contact_tick,
                    "effect_qpos": door_trace[-1] if door_trace else None,
                    "effect_predicate_reached": any(spec["predicate"](q) for q in door_trace),
                    "door_q_init": door_trace[0] if door_trace else None,
                    "door_q_min": min(door_trace) if door_trace else None,
                    "door_q_max": max(door_trace) if door_trace else None,
                    "clock": "virtual",
                    "seconds": round(_real_time.perf_counter() - t0, 1),
                }
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"[{config_id}/{force_n:.0f}N{direction}/{condition}] "
                    f"success={official_success} steps={frame_idx} q={server_queries} "
                    f"push@{push_tick} applied={row['push_applied_ticks']} "
                    f"disp={displacement if displacement is None else round(displacement, 3)} "
                    f"door=[{row['door_q_min']:.3f},{row['door_q_max']:.3f}] ({row['seconds']}s)",
                    flush=True,
                )

    raw_env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="simple/G1WholebodyCloseDoorTeleop-v0")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tape-dir", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=22085)
    p.add_argument("--sim-mode", default="mujoco_isaac")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--success-criteria", type=float, default=0.5)
    p.add_argument("--config-seed-base", type=int, default=20260824)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    run(p.parse_args())


if __name__ == "__main__":
    main()
