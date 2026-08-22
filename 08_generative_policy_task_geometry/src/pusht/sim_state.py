"""Exact save/restore of the PushT pymunk simulator state.

`gym_pusht.envs.PushTEnv._set_state` is not usable for counterfactual replay: it sets
only positions/angle, leaves both bodies' velocities untouched, and then advances the
space by one `dt`. For this experiment we need the *complete* dynamic state so that the
same action sequence replayed from a saved state is bit-identical to the original.

The PushT space contains exactly two dynamic bodies (the circular agent and the T block);
everything else is static geometry rebuilt identically by `_setup`. So the full dynamic
state is the position/velocity/angle/angular-velocity of those two bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PushTSimState:
    agent_pos: tuple[float, float]
    agent_vel: tuple[float, float]
    agent_angle: float
    agent_angvel: float
    block_pos: tuple[float, float]
    block_vel: tuple[float, float]
    block_angle: float
    block_angvel: float
    goal_pose: tuple[float, float, float]

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                *self.agent_pos, *self.agent_vel, self.agent_angle, self.agent_angvel,
                *self.block_pos, *self.block_vel, self.block_angle, self.block_angvel,
            ],
            dtype=np.float64,
        )


def _unwrap(env):
    """Return the raw PushTEnv underneath gymnasium wrappers."""
    while hasattr(env, "env") and not hasattr(env, "space"):
        env = env.env
    if not hasattr(env, "space"):
        raise TypeError("could not find the raw PushTEnv (no .space attribute)")
    return env


def save_sim_state(env) -> PushTSimState:
    e = _unwrap(env)
    return PushTSimState(
        agent_pos=tuple(map(float, e.agent.position)),
        agent_vel=tuple(map(float, e.agent.velocity)),
        agent_angle=float(e.agent.angle),
        agent_angvel=float(e.agent.angular_velocity),
        block_pos=tuple(map(float, e.block.position)),
        block_vel=tuple(map(float, e.block.velocity)),
        block_angle=float(e.block.angle),
        block_angvel=float(e.block.angular_velocity),
        goal_pose=tuple(map(float, e.goal_pose)),
    )


def restore_sim_state(env, s: PushTSimState, rebuild: bool = True) -> None:
    """Restore the simulator so that a replay is bit-identical to the original.

    Two things beyond `PushTEnv._set_state` (which sets only positions/angle, leaves
    velocities untouched, and then advances physics by one `dt`):

    * velocities and angular velocities are restored;
    * the pymunk space is **rebuilt** by default.

    The rebuild is not paranoia. Chipmunk caches contact arbiters between steps to warm-
    start impulses, and that cache is not part of any body's position/velocity. Restoring
    only body state leaves the previous rollout's contact impulses in place, and the same
    action sequence from the same restored state then produces a different outcome.
    Measured over 30 contact-active states: up to **3.67 px** of T-block keypoint drift
    with `rebuild=False`, versus exactly 0 px with the rebuild. That is the same order as
    the genuine outcome differences this experiment sets out to detect, so leaving it in
    would mean partly measuring the simulator's own history.
    """
    e = _unwrap(env)
    if rebuild:
        e._setup()  # fresh pymunk.Space => empty arbiter cache
    # Angle before position, always. The T block's centre of gravity is offset from its
    # body origin ((0, 45) in local coords), and pymunk rotates a body about its CoG, so
    # assigning `angle` *moves* `position`. Restoring position first and angle second --
    # the order `PushTEnv._set_state` uses, kept there only for legacy data compatibility
    # -- teleports the block by up to ~59 px. `preflight.py` P0 pins this down.
    e.agent.angle = s.agent_angle
    e.agent.position = list(s.agent_pos)
    e.agent.velocity = list(s.agent_vel)
    e.agent.angular_velocity = s.agent_angvel
    e.block.angle = s.block_angle
    e.block.position = list(s.block_pos)
    e.block.velocity = list(s.block_vel)
    e.block.angular_velocity = s.block_angvel
    e.goal_pose = np.array(s.goal_pose, dtype=np.float64)
    e.space.reindex_shapes_for_body(e.agent)
    e.space.reindex_shapes_for_body(e.block)
    e.n_contact_points = 0


def block_keypoints(env) -> np.ndarray:
    """8 T-block keypoints in world pixels, shape [8, 2].

    A rigid-body-aware outcome descriptor: it folds position and orientation into one
    metric space (pixels) without us having to invent a weighting between them.
    """
    e = _unwrap(env)
    # Computed here rather than via PushTEnv.get_keypoints, which uses np.row_stack
    # (removed in numpy 2). Same definition: shape vertices rotated into world frame.
    pts = []
    for shape in e._block_shapes:
        for v in shape.get_vertices():
            v = v.rotated(shape.body.angle) + shape.body.position
            pts.append((float(v.x), float(v.y)))
    return np.asarray(pts, dtype=np.float64)


def coverage(env) -> float:
    return float(_unwrap(env)._get_coverage())
