"""PushT helpers for counterfactual replay.

Two things the stock env does not give us:

1. a physics step that does *not* render. `PushTEnv.step` calls `get_obs`, which for
   image observation types renders a pygame surface. Counterfactual rollouts only need
   the final block pose, and there are B of them per probe state, so rendering would
   dominate the cost. `step_physics` reproduces `PushTEnv.step`'s integration loop
   exactly and returns nothing.

2. exact state restore (see `sim_state.py`).
"""

from __future__ import annotations

import numpy as np
from pymunk.vec2d import Vec2d

from .sim_state import _unwrap, block_keypoints, coverage


def make_env(obs_type: str = "pixels_agent_pos", render_mode: str = "rgb_array"):
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the env)

    return gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode=render_mode)


def step_physics(env, action) -> None:
    """Byte-for-byte the integration loop of `PushTEnv.step`, minus observation/render."""
    e = _unwrap(env)
    e.n_contact_points = 0
    n_steps = int(1 / (e.dt * e.control_hz))
    e._last_action = action
    act = Vec2d(float(action[0]), float(action[1]))
    for _ in range(n_steps):
        acceleration = e.k_p * (act - e.agent.position) + e.k_v * (Vec2d(0, 0) - e.agent.velocity)
        e.agent.velocity += acceleration * e.dt
        e.space.step(e.dt)


def execute_chunk(env, chunk: np.ndarray) -> dict:
    """Execute an action chunk open-loop from the *current* sim state.

    Returns the task outcome only: the T-block's keypoints (the full SE(2) pose in a
    single pixel-valued metric space) and the environment's own coverage reward.
    Nothing here touches the action space, so the outcome cannot be an algebraic
    restatement of action variability.
    """
    e = _unwrap(env)
    contacts = 0
    for a in chunk:
        step_physics(env, a)
        contacts += e.n_contact_points
    return {
        "keypoints": block_keypoints(env),
        "coverage": coverage(env),
        "block_pose": np.array([*e.block.position, e.block.angle], dtype=np.float64),
        "agent_pos": np.array([*e.agent.position], dtype=np.float64),
        "n_contacts": int(contacts),
    }


def agent_block_gap(env) -> float:
    """Distance from the pusher centre to the nearest point of the T, in pixels.

    Recorded as a descriptive covariate only — it is the obvious mundane explanation for
    goal-equivalent diversity (a pusher in free space cannot move the block), and the
    analysis has to be able to check whether that is the whole story.
    """
    e = _unwrap(env)
    best = float("inf")
    for shape in e._block_shapes:
        info = shape.point_query(e.agent.position)
        best = min(best, float(info.distance))
    return best
