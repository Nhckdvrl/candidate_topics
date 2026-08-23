"""Topic 23 G0 rollout runner: motor-condition interventions on SIMPLE + Psi0.

Frozen upstream:
    SIMPLE b49c1aea2dd57309bb533219d0d34d6020f3d943
    Psi0   9ad917526394c1cacc72dba08562629936505987

The runner mirrors `simple.cli.eval_decoupled_wbc._run_eval_worker` (the official
entry point for `*Teleop` tasks) and adds three things upstream does not have:

  1) a latched joint-level motor clamp applied AFTER the whole-body controller,
     i.e. at the true actuator boundary, so the clamp cannot be undone by the WBC;
  2) MuJoCo contact attribution between robot body parts and the task object, so a
     success can be assigned to a physical route instead of being asserted;
  3) per-condition terminal records in the frozen `g0_core` schema.

`success` always comes from the unmodified upstream evaluator (`env.unwrapped._success`).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from g0_core import Condition  # noqa: E402
from g0_core import task_effect_success  # noqa: E402
from g0_simple_psi0 import TASKS, make_record, read_effect_state  # noqa: E402

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
# Topic 23 revision 2026-08-24: `left_disabled` and `both_arms_disabled` were added
# after reading the upstream task. Without them a `right_disabled` success cannot be
# separated from "the arm was never on the causal path" (see VALIDATION_AUDIT).
CLAMP_SPEC: dict[str, tuple[str, ...]] = {
    "canonical": (),
    "right_frozen": ("right_arm", "right_hand"),
    "right_disabled": ("right_arm", "right_hand"),
    "left_disabled": ("left_arm", "left_hand"),
    "both_arms_disabled": ("right_arm", "right_hand", "left_arm", "left_hand"),
    "full_hold": ("right_arm", "right_hand", "left_arm", "left_hand", "waist"),
    # The oracle runs under exactly the `right_disabled` constraint.
    "oracle_right_disabled": ("right_arm", "right_hand"),
}
# `right_frozen` is a locked-joint fault: the limb keeps the pose it already had, so
# it loses its articulation but stays where it is. Every other clamped condition
# retracts the limb to the neutral at-side pose, removing it as an effector.
FREEZE_IN_PLACE = {"right_frozen"}
FREEZE_BASE = {"full_hold"}  # zero (vx, vy, vyaw), hold height and heading

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [j.replace("left_", "right_") for j in LEFT_ARM_JOINTS]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]


def _part_of_body(name: str) -> str:
    n = name.lower()
    if "right_hand" in n or "right_wrist" in n:
        return "right_hand"
    if n.startswith("right_") and ("shoulder" in n or "elbow" in n or "arm" in n):
        return "right_arm"
    if "left_hand" in n or "left_wrist" in n:
        return "left_hand"
    if n.startswith("left_") and ("shoulder" in n or "elbow" in n or "arm" in n):
        return "left_arm"
    if "waist" in n or "torso" in n or "pelvis" in n:
        return "torso"
    if "hip" in n or "knee" in n or "ankle" in n or "leg" in n or "foot" in n:
        return "leg"
    if "head" in n or "d435" in n or "camera" in n:
        return "head"
    return f"other:{name}"


class ContactProbe:
    """Attribute object motion to a physical route using MuJoCo contacts."""

    def __init__(self, model, data, effect_joint: str) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, effect_joint)
        if jid < 0:
            raise RuntimeError(f"effect joint {effect_joint!r} not in model")
        door_root = int(model.jnt_bodyid[jid])
        self.door_root = door_root
        self.door_bodies = {
            b for b in range(model.nbody) if self._is_descendant(b, door_root)
        }
        self.geom_body = np.asarray(model.geom_bodyid)
        self.body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            for b in range(model.nbody)
        ]
        self.part_of_body = [_part_of_body(n) for n in self.body_names]

    def _is_descendant(self, body: int, root: int) -> bool:
        b = body
        while b > 0:
            if b == root:
                return True
            b = int(self.model.body_parentid[b])
        return b == root

    def effect_body_xpos(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.door_root], dtype=float)

    def parts_touching_door(self) -> set[str]:
        parts: set[str] = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.geom_body[c.geom1])
            b2 = int(self.geom_body[c.geom2])
            in1, in2 = b1 in self.door_bodies, b2 in self.door_bodies
            if in1 == in2:
                continue
            robot_body = b2 if in1 else b1
            parts.add(self.part_of_body[robot_body])
        return parts


# Neutral MJCF pose for the G1 arms is all-zeros: arms hanging at the sides.
# `right_disabled` retracts to this pose and holds it, i.e. the limb is in a sling.
# Locking the limb at its *current* configuration is not enough for CloseDoor:
# the WBC default pose already holds the forearms forward, so a joint-locked right
# hand is still carried into the door by locomotion and still closes it. That
# intervention removes the arm's degrees of freedom without removing the effector,
# which cannot identify motor substitution. See VALIDATION_AUDIT.
SLING_ARM_QPOS = np.zeros(7)
SLING_HAND_QPOS = np.zeros(7)
RETRACT_STEPS = 60  # ramp to the sling during stabilization, ~1.2 s at 50 Hz
# Contact attribution window before the task predicate first trips. The object can
# keep moving ballistically after the robot lets go (the door coasts ~0.1 rad), so
# the contact that caused the effect is not necessarily present at the trip step.
CLOSE_WINDOW = 20


class MotorClamp:
    """Retract-and-hold clamp applied at the actuator boundary.

    The affected limb is ramped to the robot's neutral at-side pose during the
    stabilization phase and then PD-held there for the whole episode, so it is
    unavailable as an effector before the policy is ever engaged. The clamp runs
    after the whole-body controller, at the actuator boundary, so the WBC cannot
    undo it.
    """

    def __init__(self, condition: str, robot, model) -> None:
        import mujoco

        self.condition = condition
        self.groups = CLAMP_SPEC[condition]
        self.freeze_base = condition in FREEZE_BASE
        self.freeze_in_place = condition in FREEZE_IN_PLACE
        self.robot = robot
        self.latched = False
        self.hold_body_q: np.ndarray | None = None
        self.hold_left_hand: np.ndarray | None = None
        self.hold_right_hand: np.ndarray | None = None
        self.hold_height: float | None = None
        self.hold_yaw: float | None = None
        self.last_commanded_delta = 0.0
        self.ramp_step = 0
        self.start_body_q: np.ndarray | None = None
        self.start_left_hand: np.ndarray | None = None
        self.start_right_hand: np.ndarray | None = None

        # `target_q` and `prepare_obs()["body_q"]` are both ordered by
        # `robot.body_joint_index` (MuJoCo joint ids), so resolve group slices there.
        self.body_joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
            for j in np.asarray(robot.body_joint_index)
        ]
        lut = {n: i for i, n in enumerate(self.body_joint_names)}
        self.idx: dict[str, list[int]] = {
            "right_arm": [lut[n] for n in RIGHT_ARM_JOINTS if n in lut],
            "left_arm": [lut[n] for n in LEFT_ARM_JOINTS if n in lut],
            "waist": [lut[n] for n in WAIST_JOINTS if n in lut],
        }
        for g, want in (("right_arm", 7), ("left_arm", 7), ("waist", 3)):
            if len(self.idx[g]) != want:
                raise RuntimeError(
                    f"joint-group contract changed: {g} resolved {len(self.idx[g])}/{want} "
                    f"from {self.body_joint_names}"
                )

    def latch(self, proprio: dict[str, Any], base_yaw: float, height: float) -> None:
        """Capture the start pose and build the constant sling target."""
        self.start_body_q = np.asarray(proprio["body_q"], dtype=float).copy()
        self.start_left_hand = np.asarray(proprio["left_hand_q"], dtype=float).copy()
        self.start_right_hand = np.asarray(proprio["right_hand_q"], dtype=float).copy()

        self.hold_body_q = self.start_body_q.copy()
        if not self.freeze_in_place:
            for g in ("right_arm", "left_arm"):
                if g in self.groups:
                    self.hold_body_q[self.idx[g]] = SLING_ARM_QPOS
        if "waist" in self.groups:  # full_hold freezes the waist where it stands
            self.hold_body_q[self.idx["waist"]] = self.start_body_q[self.idx["waist"]]
        sling_hands = ("left_hand" in self.groups) and not self.freeze_in_place
        self.hold_left_hand = (
            SLING_HAND_QPOS.copy() if sling_hands else self.start_left_hand.copy()
        )
        self.hold_right_hand = (
            SLING_HAND_QPOS.copy()
            if ("right_hand" in self.groups and not self.freeze_in_place)
            else self.start_right_hand.copy()
        )
        self.hold_height = float(height)
        self.hold_yaw = float(base_yaw)
        self.ramp_step = 0
        self.latched = True

    def _blend(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = min(1.0, self.ramp_step / RETRACT_STEPS)
        return (
            (1 - a) * self.start_body_q + a * self.hold_body_q,
            (1 - a) * self.start_left_hand + a * self.hold_left_hand,
            (1 - a) * self.start_right_hand + a * self.hold_right_hand,
        )

    def apply_pre_wbc(self, action_cmd) -> None:
        """Only used by `full_hold`: remove intentional base motion."""
        if not self.freeze_base:
            return
        nav = np.asarray(action_cmd["navigate_cmd"], dtype=float).copy()
        nav[0:3] = 0.0          # vx, vy, turning flag
        nav[3] = self.hold_yaw  # absolute heading target, world frame
        action_cmd.parameters["navigate_cmd"] = nav
        action_cmd.parameters["base_height_command"] = np.atleast_1d(
            np.asarray([self.hold_height], dtype=float)
        )

    def apply_post_wbc(self, action_cmd) -> None:
        if not self.groups or not self.latched:
            return
        tq = np.asarray(action_cmd["target_q"], dtype=float).copy()
        body_t, lh_t, rh_t = self._blend()
        self.last_commanded_delta = 0.0
        for g in ("right_arm", "left_arm", "waist"):
            if g in self.groups and self.idx.get(g):
                ids = self.idx[g]
                self.last_commanded_delta = max(
                    self.last_commanded_delta,
                    float(np.max(np.abs(tq[ids] - body_t[ids]))),
                )
                tq[ids] = body_t[ids]
        action_cmd.parameters["target_q"] = tq
        if "right_hand" in self.groups:
            action_cmd.parameters["right_hand_q"] = rh_t
        if "left_hand" in self.groups:
            action_cmd.parameters["left_hand_q"] = lh_t
        self.ramp_step += 1


def _make_sonic_config() -> dict[str, Any]:
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def _base_yaw(data) -> float:
    q = np.asarray(data.qpos[3:7], dtype=float)  # w, x, y, z
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch
    from gymnasium.wrappers import TimeLimit
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import simple.envs as _  # noqa: F401
    from simple.datasets.lerobot import get_episode_lerobot
    from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
    from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent
    from simple.envs.wrappers.video_recorder import VideoRecorder

    # `full_hold` has to remove base motion from the *policy* command, before the
    # whole-body controller consumes it. Psi0DecoupledWbcAgent.get_action pops the
    # queued `vla_cmd` through its super(), so that is the pre-WBC seam.
    _active: dict[str, Any] = {"clamp": None}
    _orig_get_action = SonicDecoupledWbcAgent.get_action

    def _hooked_get_action(self, observation, instruction=None, **kwargs):
        cmd = _orig_get_action(self, observation, instruction, **kwargs)
        c = _active["clamp"]
        if c is not None and cmd is not None and cmd.type == "vla_cmd":
            c.apply_pre_wbc(cmd)
        return cmd

    SonicDecoupledWbcAgent.get_action = _hooked_get_action

    spec = TASKS[args.env_id]
    sonic_config = _make_sonic_config()
    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists() and args.resume:
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["config_id"], r["condition"]))
        print(f"[resume] {len(done_keys)} records already present")

    ds = LeRobotDataset(repo_id=args.env_id, root=args.data_dir, video_backend="pyav")
    render_hz = ds.meta.fps
    print(f"dataset {args.data_dir}: {ds.num_episodes} episodes @ {render_hz} Hz")

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

    # Scripted feasibility oracle: same env, same config, same right-side clamp,
    # left arm driven by MuJoCo Jacobian IK. Needs no policy server.
    from topic23_oracle import LeftArmOracle, build_oracle_action

    _upper_idx = {n: i for i, n in enumerate(agent.sonic_upper_joint_names)}
    _upper_base = agent._dwbc_robot_model.get_initial_upper_body_pose()

    video_root = Path(args.video_dir)
    video_root.mkdir(parents=True, exist_ok=True)

    dr_level = Path(args.data_dir).name
    for eps_idx in range(args.episode_start, min(ds.num_episodes, args.episode_start + args.num_episodes)):
        config_id = f"{dr_level}:{eps_idx}"
        for condition in args.conditions:
            key = (config_id, condition)
            if key in done_keys:
                print(f"[skip] {key}")
                continue
            t0 = time.perf_counter()

            # Matched configs: identical scene draw for every condition.
            random.seed(args.config_seed_base + eps_idx)
            np.random.seed(args.config_seed_base + eps_idx)
            env_conf, _episode = get_episode_lerobot(ds, eps_idx)

            env = env_t
            if args.save_video:
                env = VideoRecorder(
                    env=env_t, video_folder=str(video_root),
                    name_prefix=f"{dr_level}_cfg{eps_idx}_{condition}",
                    framerate=render_hz, write_png=False,
                )

            observation, info = env.reset(options={"state_dict": env_conf})
            sonic_env = raw_env.unwrapped
            model, data = sonic_env.mjModel, sonic_env.mjData
            probe = ContactProbe(model, data, spec["effect_joint"])
            import mujoco as _mj

            # The authored `*_hand_palm_link` is a geom on the wrist body, not a
            # standalone body in the MuJoCo representation (same folding Topic 19
            # hit). Evaluate the authored palm point on the wrist frame instead.
            palm_bid = {
                side: _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
                for side in ("left", "right")
            }
            palm_off = {"left": np.array([0.0415, 0.003, 0.0]),
                        "right": np.array([0.0415, -0.003, 0.0])}
            if min(palm_bid.values()) < 0:
                raise RuntimeError(f"wrist frame contract changed: {palm_bid}")
            clamp = MotorClamp(condition, robot, model)

            is_oracle = condition == Condition.ORACLE_RIGHT_DISABLED.value
            agent.reset()
            agent._wbc_policy.lower_body_policy.use_policy_action = True

            # Retract the affected limb during stabilization so the policy's first
            # observation already contains the constraint.
            clamp.latch(robot.prepare_obs(), _base_yaw(data), float(agent._last_base_height_cmd))
            sim_cnt = 0
            while (not robot.stabilized or clamp.ramp_step < RETRACT_STEPS) and sim_cnt < 400:
                action = agent.get_stabilize_action(observation)
                clamp.apply_post_wbc(action)
                observation, *_, info = sonic_env.step(action)
                sonic_env.update_viewer()
                sonic_env.update_reward()
                sim_cnt += 1
            if isinstance(env, VideoRecorder):
                env._init_writers(observation)
            clamp.hold_yaw = _base_yaw(data)
            _active["clamp"] = clamp

            agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1,), dtype=torch.float32)

            oracle = (
                LeftArmOracle(robot, model, data, spec["effect_joint"], agent)
                if is_oracle else None
            )
            oracle_log: list[dict[str, Any]] = []

            q0 = np.asarray(observation["joint_qpos"], dtype=float).copy()
            prev = q0
            motion = defaultdict(float)
            parts_ever: set[str] = set()
            parts_at_close: set[str] = set()
            door_trace: list[float] = []
            first_closed_step = None
            clamp_leak = 0.0
            cmd_delta = 0.0
            contact_window: deque[set[str]] = deque(maxlen=CLOSE_WINDOW)
            recent_contacts: set[str] = set()
            # Threshold-free route attribution: whoever was touching the object on
            # the step it moved fastest is what caused the effect. A fixed lookback
            # window is not enough -- the door coasts after the robot lets go, and
            # the coast length varies with the initial angle.
            contact_hist: list[set[str]] = []
            dq_hist: list[float] = []
            last_contact_step = None
            last_contact_parts: set[str] = set()
            base_xy0 = np.asarray(data.qpos[:2], dtype=float).copy()
            base_path = 0.0
            base_prev = base_xy0.copy()
            palm_min = {"left": float("inf"), "right": float("inf")}
            ra_lo = ra_hi = None
            la_lo = la_hi = None

            frame_idx = 0
            episode_over = False
            instruction = task.instruction
            while not episode_over:
                try:
                    if is_oracle:
                        oracle_log.append(oracle.step())
                        action_cmd = build_oracle_action(
                            agent, oracle, _upper_idx, _upper_base
                        )
                    else:
                        action_cmd = agent.get_action(
                            observation, info=info, instruction=instruction
                        )
                except StopIteration:
                    break
                clamp.apply_post_wbc(action_cmd)
                observation, _reward, terminated, truncated, info = env.step(action_cmd)
                episode_over = terminated or truncated
                frame_idx += 1

                q = np.asarray(observation["joint_qpos"], dtype=float)
                motion["left_arm"] += float(np.linalg.norm(q[15:22] - prev[15:22]))
                motion["right_arm"] += float(np.linalg.norm(q[22:29] - prev[22:29]))
                motion["waist"] += float(np.linalg.norm(q[12:15] - prev[12:15]))
                ra_lo = q[22:29].copy() if ra_lo is None else np.minimum(ra_lo, q[22:29])
                ra_hi = q[22:29].copy() if ra_hi is None else np.maximum(ra_hi, q[22:29])
                la_lo = q[15:22].copy() if la_lo is None else np.minimum(la_lo, q[15:22])
                la_hi = q[15:22].copy() if la_hi is None else np.maximum(la_hi, q[15:22])
                prev = q.copy()
                if clamp.latched and clamp.idx.get("right_arm"):
                    hold_r = clamp.hold_body_q[clamp.idx["right_arm"]]
                    cur_r = np.asarray(robot.prepare_obs()["body_q"])[clamp.idx["right_arm"]]
                    clamp_leak = max(clamp_leak, float(np.max(np.abs(cur_r - hold_r))))
                cmd_delta = max(cmd_delta, getattr(clamp, "last_commanded_delta", 0.0))

                door_q = float(data.joint(spec["effect_joint"]).qpos[0])
                door_trace.append(door_q)
                touching = probe.parts_touching_door()
                parts_ever |= touching
                # The task predicate is per-task (`< -0.16` for the door, `|q| > 0.7`
                # for the faucet); do not hardcode one of them here.
                if first_closed_step is None and task_effect_success(spec["short_name"], door_q):
                    first_closed_step = frame_idx
                    parts_at_close |= recent_contacts
                if first_closed_step is not None and frame_idx - first_closed_step <= 5:
                    parts_at_close |= touching
                recent_contacts = set()
                for t in list(contact_window)[-CLOSE_WINDOW:]:
                    recent_contacts |= t
                contact_window.append(touching)

                contact_hist.append(set(touching))
                dq_hist.append(abs(door_q - door_trace[-2]) if len(door_trace) > 1 else 0.0)
                if touching:
                    last_contact_step = frame_idx
                    last_contact_parts = set(touching)

                base_xy = np.asarray(data.qpos[:2], dtype=float)
                base_path += float(np.linalg.norm(base_xy - base_prev))
                base_prev = base_xy.copy()
                eff_xpos = probe.effect_body_xpos()
                for side, bid in palm_bid.items():
                    p_world = np.asarray(data.xpos[bid]) + np.asarray(
                        data.xmat[bid]
                    ).reshape(3, 3) @ palm_off[side]
                    palm_min[side] = min(
                        palm_min[side], float(np.linalg.norm(p_world - eff_xpos))
                    )

            # Attribute the effect to the contact present when the object moved fastest.
            cut = first_closed_step if first_closed_step is not None else len(dq_hist)
            parts_at_peak: set[str] = set()
            peak_step = None
            if cut > 1:
                peak_step = int(np.argmax(dq_hist[:cut]))
                for k in range(max(0, peak_step - 5), min(len(contact_hist), peak_step + 6)):
                    parts_at_peak |= contact_hist[k]

            official_success = bool(raw_env.unwrapped._success)
            effect = read_effect_state(sonic_env, args.env_id)

            attribution = parts_at_peak or parts_at_close or last_contact_parts
            right_used = bool({"right_arm", "right_hand"} & attribution)
            other_used = bool(attribution - {"right_arm", "right_hand"})
            row = make_record(
                env_id=args.env_id, config_id=config_id, condition=condition,
                effect=effect, official_success=official_success,
                route_verified=(other_used and not right_used) if condition != "canonical" else None,
                left_arm_motion_l2=motion["left_arm"],
                torso_motion_l2=motion["waist"],
            )
            row.update(
                right_arm_motion_l2=motion["right_arm"],
                contact_parts_ever=sorted(parts_ever),
                contact_parts_at_close=sorted(parts_at_close),
                contact_parts_at_peak_motion=sorted(parts_at_peak),
                route_attribution=sorted(attribution),
                peak_motion_step=peak_step,
                last_contact_step=last_contact_step,
                last_contact_parts=sorted(last_contact_parts),
                canonical_right_route=right_used,
                first_closed_step=first_closed_step,
                door_q_init=door_trace[0] if door_trace else None,
                door_q_min=min(door_trace) if door_trace else None,
                door_q_max=max(door_trace) if door_trace else None,
                steps=frame_idx,
                # Did the robot even get to the object? Separates "approached and
                # then did nothing" from "never approached".
                base_path_m=round(base_path, 3),
                base_net_displacement_m=round(float(np.linalg.norm(base_prev - base_xy0)), 3),
                min_dist_left_palm_m=(
                    round(palm_min["left"], 3) if np.isfinite(palm_min["left"]) else None
                ),
                min_dist_right_palm_m=(
                    round(palm_min["right"], 3) if np.isfinite(palm_min["right"]) else None
                ),
                right_arm_excursion_rad=round(float(np.max(ra_hi - ra_lo)), 4) if ra_lo is not None else None,
                left_arm_excursion_rad=round(float(np.max(la_hi - la_lo)), 4) if la_lo is not None else None,
                right_arm_clamp_leak_rad=round(clamp_leak, 4),
                right_arm_precmd_delta_rad=round(cmd_delta, 4),
                dr_level=dr_level,
                oracle_phase=oracle_log[-1]["phase"] if oracle_log else None,
                oracle_min_palm_dist=(
                    round(min(o["palm_dist"] for o in oracle_log), 3) if oracle_log else None
                ),
                oracle_reached_sweep=(
                    any(o["phase"] == "sweep" for o in oracle_log) if oracle_log else None
                ),
                seconds=round(time.perf_counter() - t0, 1),
            )
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{config_id}/{condition}] success={official_success} door=[{row['door_q_min']:.3f},{row['door_q_max']:.3f}] "
                  f"route={row['route_attribution']} rarm_l2={motion['right_arm']:.3f} "
                  f"larm_l2={motion['left_arm']:.3f} leak={clamp_leak:.3f} "
                  f"base={base_path:.2f}m palmL={row['min_dist_left_palm_m']} palmR={row['min_dist_right_palm_m']} "
                  f"({row['seconds']}s)", flush=True)

            if isinstance(env, VideoRecorder):
                env.release()

    raw_env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="simple/G1WholebodyCloseDoorTeleop-v0")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--video-dir", default="videos")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=22085)
    p.add_argument("--sim-mode", default="mujoco_isaac")
    p.add_argument("--conditions", nargs="+", default=list(CLAMP_SPEC))
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--success-criteria", type=float, default=0.5)
    p.add_argument("--config-seed-base", type=int, default=20260824)
    p.add_argument("--save-video", action="store_true", default=True)
    p.add_argument("--no-save-video", dest="save_video", action="store_false")
    p.add_argument("--resume", action="store_true", default=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
