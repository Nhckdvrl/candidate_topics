from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .geometry import task_null_projectors


@dataclass
class ArmConfig:
    link_lengths: Tuple[float, ...] = (0.45, 0.35, 0.25, 0.20)
    dt: float = 0.08
    max_speed: float = 1.6
    kp_task: float = 3.5
    kp_null: float = 1.2
    damping: float = 0.03
    success_tol: float = 0.03
    max_steps: int = 90


class PlanarArm:
    """Minimal redundant robot: 4 joints, 2-D end-effector position task."""

    def __init__(self, cfg: ArmConfig | None = None):
        self.cfg = cfg or ArmConfig()
        self.link_lengths = np.asarray(self.cfg.link_lengths, dtype=np.float64)
        self.n_joints = len(self.link_lengths)

    def fk(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        theta = np.cumsum(q)
        return np.array([
            np.sum(self.link_lengths * np.cos(theta)),
            np.sum(self.link_lengths * np.sin(theta)),
        ], dtype=np.float64)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        theta = np.cumsum(q)
        j = np.zeros((2, self.n_joints), dtype=np.float64)
        for k in range(self.n_joints):
            j[0, k] = -np.sum(self.link_lengths[k:] * np.sin(theta[k:]))
            j[1, k] = np.sum(self.link_lengths[k:] * np.cos(theta[k:]))
        return j

    def expert_action(self, q: np.ndarray, target: np.ndarray, q_pref: np.ndarray, null_gain: float = 1.0) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        j = self.jacobian(q)
        err = target - self.fk(q)
        v = self.cfg.kp_task * err
        jj = j @ j.T + (self.cfg.damping ** 2) * np.eye(2)
        j_dls = j.T @ np.linalg.inv(jj)
        task = j_dls @ v
        _, p_null, _, _ = task_null_projectors(j)
        null = null_gain * self.cfg.kp_null * (p_null @ (q_pref - q))
        u = task + null
        return np.clip(u, -self.cfg.max_speed, self.cfg.max_speed)

    def step(self, q: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q) + self.cfg.dt * np.asarray(action), -2.8, 2.8)

    def distance(self, q: np.ndarray, target: np.ndarray) -> float:
        return float(np.linalg.norm(self.fk(q) - target))


PREF_LIBRARY = np.array([
    [0.9, -1.1, 0.9, -0.5],
    [-0.8, 1.1, -0.9, 0.6],
    [0.6, 0.6, -1.0, -0.5],
    [-0.6, -0.6, 1.0, 0.5],
    [1.0, -0.4, -0.8, 0.7],
    [-1.0, 0.4, 0.8, -0.7],
], dtype=np.float64)


def sample_reachable_problem(arm: PlanarArm, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        q0 = rng.uniform(-1.35, 1.35, size=arm.n_joints)
        q_goal = rng.uniform(-1.35, 1.35, size=arm.n_joints)
        target = arm.fk(q_goal)
        d = arm.distance(q0, target)
        if 0.20 <= d <= 1.10:
            return q0, target
    raise RuntimeError("failed to sample a nontrivial reachable problem")


def generate_episode(
    arm: PlanarArm,
    rng: np.random.Generator,
    null_gain: float = 1.0,
    q_pref: np.ndarray | None = None,
    q0: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> dict:
    if q0 is None or target is None:
        q, target = sample_reachable_problem(arm, rng)
    else:
        q = np.asarray(q0, dtype=np.float64).copy()
        target = np.asarray(target, dtype=np.float64).copy()
    if q_pref is None:
        q_pref = PREF_LIBRARY[rng.integers(0, len(PREF_LIBRARY))].copy()
        q_pref += rng.normal(scale=0.10, size=arm.n_joints)

    qs: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    dists: List[float] = []
    for _ in range(arm.cfg.max_steps):
        d = arm.distance(q, target)
        qs.append(q.copy())
        dists.append(d)
        if d < arm.cfg.success_tol:
            break
        a = arm.expert_action(q, target, q_pref, null_gain=null_gain)
        actions.append(a.copy())
        q = arm.step(q, a)

    qs = qs[: len(actions)]
    dists = dists[: len(actions)]
    return {
        "q": np.asarray(qs, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "target": np.asarray(target, dtype=np.float32),
        "q_pref": np.asarray(q_pref, dtype=np.float32),
        "distance": np.asarray(dists, dtype=np.float32),
        "success": bool(arm.distance(q, target) < arm.cfg.success_tol),
    }


def build_windows(episodes: List[dict], horizon: int = 8) -> dict:
    obs, actions, episode_id, step_id = [], [], [], []
    for eid, ep in enumerate(episodes):
        q = ep["q"]
        a = ep["action"]
        target = ep["target"]
        if len(a) == 0:
            continue
        for t in range(len(a)):
            idx = np.minimum(np.arange(t, t + horizon), len(a) - 1)
            obs.append(np.concatenate([q[t], target], axis=0))
            actions.append(a[idx])
            episode_id.append(eid)
            step_id.append(t)
    return {
        "obs": np.asarray(obs, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "step_id": np.asarray(step_id, dtype=np.int64),
    }


def generate_dataset(
    n_base_tasks: int,
    modes_per_task: int,
    horizon: int,
    seed: int,
    null_gain: float = 1.0,
) -> Tuple[dict, dict]:
    """Generate repeated identical tasks with different hidden posture preferences."""
    arm = PlanarArm()
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n_base_tasks):
        q0, target = sample_reachable_problem(arm, rng)
        mode_ids = rng.choice(len(PREF_LIBRARY), size=modes_per_task, replace=modes_per_task > len(PREF_LIBRARY))
        for mid in mode_ids:
            q_pref = PREF_LIBRARY[int(mid)].copy() + rng.normal(scale=0.05, size=arm.n_joints)
            episodes.append(generate_episode(
                arm, rng, null_gain=null_gain, q_pref=q_pref, q0=q0, target=target
            ))
    data = build_windows(episodes, horizon=horizon)
    summary = {
        "n_base_tasks": n_base_tasks,
        "modes_per_task": modes_per_task,
        "n_episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "n_windows": int(len(data["obs"])),
        "mean_episode_steps": float(np.mean([len(e["action"]) for e in episodes])),
    }
    return data, summary
