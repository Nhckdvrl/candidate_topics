"""Scripted left-arm oracle: is the task still solvable with the right side slung?

This is the feasibility prerequisite for Topic 23. A `right_disabled` failure is
uninterpretable without it: the policy failing and the task being impossible look
identical in the success column.

The oracle is deliberately *not* a policy. It runs the same environment, the same
config, and the exact same right-side clamp as the `right_disabled` condition, and
solves the task with the left arm using MuJoCo's own Jacobian. It needs no policy
server.

Geometry it relies on (measured on OpenFaucet dr-level-0, 2026-08-24):

    effect joint  articulate_joint_0, hinge, world axis ~ [0.196, 0, 0.981]
    lever body    link_0, box geom, half-extent [0.034, 0.030, 0.102]
    lever centre  ~0.062 m above the hinge anchor along the body z axis
    handle world  x ~ -0.68, y in [-0.01, +0.10], z ~ 0.86
    left palm     [-0.91, +0.152, 0.888] at spawn, 0.22-0.30 m from the handle

The robot spawns within arm's reach, so the oracle does not walk: `navigate_cmd`
stays zero throughout. That makes a success unambiguous -- it is a left-arm
solution, not a locomotion solution.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
PALM_OFFSET_LEFT = np.array([0.0415, 0.003, 0.0])

# The palm is commanded to a standoff point on the near side of the lever rather
# than to the lever centre: aiming at the centre makes the IK push the palm into
# the geom, the contact blocks it, and the loop winds up thrashing the whole body.
STANDOFF = 0.070         # m, radius of the approach point around the lever centre
APPROACH_TOL = 0.105     # m, palm-to-lever-centre distance at which pushing starts
SETTLE_STEPS = 40        # let the WBC settle before commanding the arm
SWEEP_RATE = 0.008       # rad per control step of commanded lever angle
SWEEP_TARGET = 0.95      # rad, comfortably past the |q| > 0.7 success predicate
# Deliberately slow. A fast IK loop makes the balance controller fight the arm and
# the base walks away even with a zero navigate command, which would make an
# oracle success unattributable.
IK_GAIN = 0.8
IK_DAMPING = 0.10
MAX_DQ = 0.05            # rad per step per joint, relative to the *measured* pose


def _rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / max(np.linalg.norm(axis), 1e-9)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


class LeftArmOracle:
    """Drives the left palm to the lever and sweeps it past the success angle."""

    def __init__(self, robot, model, data, effect_joint: str, wbc_agent) -> None:
        import mujoco

        self.mj = mujoco
        self.robot = robot
        self.model = model
        self.data = data
        self.wbc = wbc_agent

        self.jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, effect_joint)
        if self.jid < 0:
            raise RuntimeError(f"effect joint {effect_joint!r} missing")
        self.lever_bid = int(model.jnt_bodyid[self.jid])
        self.qadr = int(model.jnt_qposadr[self.jid])

        self.palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        if self.palm_bid < 0:
            raise RuntimeError("left_wrist_yaw_link missing")

        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_ARM_JOINTS]
        if min(jids) < 0:
            raise RuntimeError("left arm joint contract changed")
        self.arm_jids = jids
        self.arm_dofs = [int(model.jnt_dofadr[j]) for j in jids]
        self.arm_qadr = [int(model.jnt_qposadr[j]) for j in jids]
        self.arm_limits = np.array([model.jnt_range[j] for j in jids], dtype=float)

        # Lever centre expressed in the lever body frame, so it can be rotated to a
        # commanded angle instead of only tracked where it currently is.
        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == self.lever_bid]
        if not gids:
            raise RuntimeError("lever body has no geoms")
        g = max(gids, key=lambda g: float(np.prod(model.geom_size[g])))
        self.lever_local = np.asarray(model.geom_pos[g], dtype=float).copy()

        self.q_cmd = np.array([data.qpos[a] for a in self.arm_qadr], dtype=float)
        self.commanded_angle: float | None = None
        self.last_ik_err = float("nan")
        self.sign = 0.0
        self.phase = "settle"
        self.steps_in_phase = 0

    # -- geometry ---------------------------------------------------------
    def _palm_world(self) -> np.ndarray:
        R = np.asarray(self.data.xmat[self.palm_bid]).reshape(3, 3)
        return np.asarray(self.data.xpos[self.palm_bid]) + R @ PALM_OFFSET_LEFT

    def _lever_world(self) -> np.ndarray:
        R = np.asarray(self.data.xmat[self.lever_bid]).reshape(3, 3)
        return np.asarray(self.data.xpos[self.lever_bid]) + R @ self.lever_local

    def _anchor_axis(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.data.xanchor[self.jid], dtype=float),
            np.asarray(self.data.xaxis[self.jid], dtype=float),
        )

    def _angle(self) -> float:
        return float(self.data.qpos[self.qadr])

    def _standoff(self, lever: np.ndarray) -> np.ndarray:
        """A point STANDOFF metres from the lever centre, on the palm's side."""
        v = self._palm_world() - lever
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return lever
        return lever + v / n * STANDOFF

    def _target_point(self) -> np.ndarray:
        """Where the palm should be next, given the commanded sweep angle."""
        lever = self._lever_world()
        if self.phase != "sweep":
            return self._standoff(lever)
        anchor, axis = self._anchor_axis()
        delta = self.commanded_angle - self._angle()
        R = _rot_about_axis(axis, delta)
        # Rotate both the lever and the standoff point, so the palm leads the
        # lever around the hinge instead of driving through it.
        lever_next = anchor + R @ (lever - anchor)
        return lever_next + R @ (self._standoff(lever) - lever)

    # -- control ----------------------------------------------------------
    def _ik_step(self, target: np.ndarray) -> None:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        palm = self._palm_world()
        self.mj.mj_jac(self.model, self.data, jacp, jacr, palm, self.palm_bid)
        J = jacp[:, self.arm_dofs]
        err = target - palm
        lam2 = IK_DAMPING ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), err)
        dq = np.clip(IK_GAIN * dq, -MAX_DQ, MAX_DQ)
        # Close the loop on the *measured* joint state. Integrating an open-loop
        # command drifts away from reality whenever the WBC does not track it, and
        # the Jacobian is evaluated at the real configuration, so the two disagree
        # and the arm converges to a standing offset instead of the target.
        q_meas = np.array([self.data.qpos[a] for a in self.arm_qadr], dtype=float)
        self.q_cmd = np.clip(
            q_meas + dq, self.arm_limits[:, 0] + 1e-3, self.arm_limits[:, 1] - 1e-3
        )
        self.last_ik_err = float(np.linalg.norm(err))

    def step(self) -> dict[str, Any]:
        q = self._angle()
        palm = self._palm_world()
        dist = float(np.linalg.norm(palm - self._lever_world()))
        self.steps_in_phase += 1

        if self.phase == "settle" and self.steps_in_phase >= SETTLE_STEPS:
            self.phase = "approach"
            self.steps_in_phase = 0

        if self.phase == "settle":
            return {"phase": self.phase, "lever_q": q, "palm_dist": dist,
                    "commanded_angle": self.commanded_angle}

        if self.phase == "approach" and dist < APPROACH_TOL:
            self.phase = "sweep"
            self.steps_in_phase = 0
            self.commanded_angle = q
            # Sweep whichever way the lever is already free to move. Both
            # directions satisfy the upstream predicate (|q| > 0.7).
            lo, hi = self.model.jnt_range[self.jid]
            self.sign = 1.0 if (hi - q) >= (q - lo) else -1.0

        if self.phase == "sweep":
            self.commanded_angle = float(
                np.clip(
                    self.commanded_angle + self.sign * SWEEP_RATE,
                    -SWEEP_TARGET, SWEEP_TARGET,
                )
            )
            # Do not let the command run arbitrarily far ahead of the realised
            # angle; that would just drive the palm through the lever.
            self.commanded_angle = float(np.clip(self.commanded_angle, q - 0.25, q + 0.25))

        self._ik_step(self._target_point())
        return {"phase": self.phase, "lever_q": q, "palm_dist": dist,
                "ik_err": self.last_ik_err, "commanded_angle": self.commanded_angle}

    def upper_body_pose(self, base_pose: np.ndarray, name_index: dict[str, int]) -> np.ndarray:
        pose = np.asarray(base_pose, dtype=float).copy()
        for name, v in zip(LEFT_ARM_JOINTS, self.q_cmd):
            pose[name_index[name]] = v
        return pose


def build_oracle_action(agent, oracle: LeftArmOracle, name_index, base_pose):
    """One WBC step driven by the oracle: stand still, left arm to the IK target."""
    from decoupled_wbc.control.main.constants import DEFAULT_BASE_HEIGHT
    from simple.core.action import ActionCmd

    t_now = time.monotonic()
    proprio = agent.robot.prepare_obs()
    agent._wbc_policy.set_observation(agent._build_wbc_observation(proprio))
    goal = {
        "target_upper_body_pose": oracle.upper_body_pose(base_pose, name_index),
        # The oracle never walks: a success is a left-arm solution, not a gait.
        "navigate_cmd": np.zeros(4, dtype=np.float32),
        "base_height_command": np.atleast_1d(np.asarray(DEFAULT_BASE_HEIGHT)),
        "target_time": t_now + 1 / agent._control_frequency,
        "interpolation_garbage_collection_time": t_now - 2 / agent._control_frequency,
        "timestamp": t_now,
    }
    agent._wbc_policy.set_goal(goal)
    wbc_action = agent._wbc_policy.get_action(time=t_now)
    rm = agent._dwbc_robot_model
    return ActionCmd(
        "decoupled_wbc",
        target_q=rm.get_body_actuated_joints(wbc_action["q"]),
        left_hand_q=rm.get_hand_actuated_joints(wbc_action["q"], side="left"),
        right_hand_q=rm.get_hand_actuated_joints(wbc_action["q"], side="right"),
    )
