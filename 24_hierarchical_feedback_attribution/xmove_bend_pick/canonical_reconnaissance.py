"""Canonical reconnaissance for XMoveBendPickTeleop, before any G3 preregistration.

No perturbation. This script observes only: canonical success, episode length,
control cadence, and the first tick of right-hand <-> target physical contact.
It exists to let G3's timing anchor and eligible-config panel be frozen from
real data before a single push-condition row is collected -- not to explore
"what would look good."

Target body identity is read from the same accessor the engine itself built
it with, not re-derived by name lookup:

    target_body_id = env.unwrapped.mujoco.mj_objects["target"].id

with a fallback to `mj_name2id(model, BODY, task.target.asset.label)` only if
that accessor is unavailable, asserted to agree with it when both exist
(`simple/engines/mujoco.py::_build_object` is the source of both).

Right-hand identity is the full kinematic subtree rooted at
`right_wrist_roll_link` (confirmed against
`data/robots/g1/g1_29dof_with_dex3.xml`: elbow -> wrist_roll -> wrist_pitch ->
wrist_yaw -> {thumb,middle,index} fingers), not a body-name substring match.
A contact counts only if one of its two geoms belongs to that exact subtree
and the other belongs to the exact target body id above.

Frozen timing rule, decided here before any push data exists:

    push_tick = first_contact_tick - round(1.0 / control_dt)

`control_dt` is read from the live `sonic_config`, not assumed to be
CloseDoor's 50 Hz. If `push_tick < 0` or no contact occurs at all in the
canonical rollout, the config is `timing_ineligible` and is excluded from
every G3 force/condition cell -- never replaced with a different config or
a different anchor rule for that one case. Whether the canonical rollout
ultimately succeeded is recorded but never used to exclude a config: a
canonical contact that fails to complete the task is still eligible, because
excluding it would silently turn the evaluation population into "scenes
where the released checkpoint already succeeds."
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time as _real_time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import numpy as np

ENV_ID = "simple/G1WholebodyXMoveBendPickTeleop-v0"
RIGHT_HAND_ROOT = "right_wrist_roll_link"


def _make_sonic_config() -> dict[str, Any]:
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def resolve_target_body_id(env, task, model) -> tuple[int, dict[str, Any]]:
    """Prefer the accessor the engine itself used to build the body."""
    import mujoco

    diag: dict[str, Any] = {}
    accessor_id = None
    mujoco_sim = getattr(env.unwrapped, "mujoco", None)
    if mujoco_sim is not None and "target" in getattr(mujoco_sim, "mj_objects", {}):
        accessor_id = int(mujoco_sim.mj_objects["target"].id)
    diag["accessor_id"] = accessor_id

    label = getattr(task.target.asset, "label", None) or task.target.asset.uid
    diag["asset_label"] = label
    name_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, label)
    diag["name2id"] = int(name_id) if name_id >= 0 else None

    if accessor_id is not None:
        if name_id >= 0 and name_id != accessor_id:
            raise RuntimeError(
                f"target body id mismatch: mj_objects['target'].id={accessor_id} "
                f"but mj_name2id({label!r})={name_id}"
            )
        return accessor_id, diag
    if name_id < 0:
        raise RuntimeError(f"target body {label!r} not found by either accessor")
    return int(name_id), diag


def right_hand_body_set(model, root_name: str) -> set[int]:
    import mujoco

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root < 0:
        raise RuntimeError(f"right-hand root body {root_name!r} not in model")
    out = set()
    for b in range(model.nbody):
        x = b
        while x > 0:
            if x == root:
                out.add(b)
                break
            x = int(model.body_parentid[x])
    return out


def hand_target_contact(model, data, target_body_id: int, hand_bodies: set[int]) -> bool:
    geom_body = np.asarray(model.geom_bodyid)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(geom_body[c.geom1]), int(geom_body[c.geom2])
        if (b1 == target_body_id and b2 in hand_bodies) or (b2 == target_body_id and b1 in hand_bodies):
            return True
    return False


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch
    from gymnasium.wrappers import TimeLimit
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import simple.envs as _  # noqa: F401
    from simple.datasets.lerobot import get_episode_lerobot
    from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent

    sonic_config = _make_sonic_config()
    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt
    lead_ticks = int(round(1.0 / control_dt))
    print(f"[cadence] sim_dt={sim_dt} control_dt={control_dt} (={1/control_dt:.1f} Hz) "
          f"lead_ticks={lead_ticks}", flush=True)

    ds = LeRobotDataset(repo_id=ENV_ID, root=args.data_dir, video_backend="pyav")
    raw_env = gym.make(
        ENV_ID, sim_mode=args.sim_mode, render_hz=ds.meta.fps,
        headless=True, sonic_config=sonic_config,
    )
    task = raw_env.unwrapped.task
    max_steps = args.max_episode_steps or task.metadata.get("max_episode_steps")
    print(f"[task] max_episode_steps={max_steps}", flush=True)
    env = TimeLimit(raw_env, max_episode_steps=max_steps)
    robot = task.robot
    agent = Psi0DecoupledWbcAgent(robot, args.host, args.port, sonic_config=sonic_config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists() and args.resume:
        for line in out_path.read_text().splitlines():
            if line.strip():
                done_keys.add(json.loads(line)["config_id"])
        print(f"[resume] {len(done_keys)} configs already present", flush=True)

    dr_level = Path(args.data_dir).name
    hand_bodies_cache: set[int] | None = None

    for eps_idx in range(args.episode_start, min(ds.num_episodes, args.episode_start + args.num_episodes)):
        config_id = f"{dr_level}:{eps_idx}"
        if config_id in done_keys:
            print(f"[skip] {config_id}", flush=True)
            continue
        t0 = _real_time.perf_counter()

        random.seed(args.config_seed_base + eps_idx)
        np.random.seed(args.config_seed_base + eps_idx)
        env_conf, _ = get_episode_lerobot(ds, eps_idx)
        observation, info = env.reset(options={"state_dict": env_conf})
        sonic_env = raw_env.unwrapped
        model, data = sonic_env.mjModel, sonic_env.mjData

        target_body_id, target_diag = resolve_target_body_id(raw_env, task, model)
        if hand_bodies_cache is None:
            hand_bodies_cache = right_hand_body_set(model, RIGHT_HAND_ROOT)
        hand_bodies = hand_bodies_cache

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

        first_contact_tick = None
        last_contact_tick = None
        n_contact_ticks = 0
        frame_idx = 0
        episode_over = False
        instruction = task.instruction
        while not episode_over:
            try:
                action_cmd = agent.get_action(observation, info=info, instruction=instruction)
            except StopIteration:
                break
            observation, _r, terminated, truncated, info = env.step(action_cmd)
            episode_over = terminated or truncated
            frame_idx += 1

            if hand_target_contact(model, data, target_body_id, hand_bodies):
                if first_contact_tick is None:
                    first_contact_tick = frame_idx
                last_contact_tick = frame_idx
                n_contact_ticks += 1

        official_success = bool(raw_env.unwrapped._success)
        push_tick = (first_contact_tick - lead_ticks) if first_contact_tick is not None else None
        timing_eligible = push_tick is not None and push_tick >= 0

        row = {
            "config_id": config_id, "dr_level": dr_level,
            "success": official_success,
            "steps": frame_idx,
            "first_contact_tick": first_contact_tick,
            "last_contact_tick": last_contact_tick,
            "n_contact_ticks": n_contact_ticks,
            "push_tick": push_tick,
            "lead_ticks": lead_ticks,
            "timing_eligible": timing_eligible,
            "target_body_diag": target_diag,
            "control_dt": control_dt,
            "seconds": round(_real_time.perf_counter() - t0, 1),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[{config_id}] success={official_success} steps={frame_idx} "
            f"first_contact={first_contact_tick} push_tick={push_tick} "
            f"eligible={timing_eligible} n_contact_ticks={n_contact_ticks} "
            f"({row['seconds']}s)", flush=True,
        )

    raw_env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=26085)
    p.add_argument("--sim-mode", default="mujoco_isaac")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--config-seed-base", type=int, default=20260824)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    run(p.parse_args())


if __name__ == "__main__":
    main()
