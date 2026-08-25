"""Topic 24 G1: hybrid VLA-seam replay to factor the right-push reversal.

Only collects the two new conditions. `RR` (both channels replayed) and `LL`
(both channels live) are not re-run: they are read directly from the frozen
G0 records (`records/g0_closedoor.jsonl`) at force=100N, since G0's
`vla_replay` and `fresh` conditions are exactly RR and LL respectively.

    LR   navigation/base channel live, upper-body channel replayed
    RL   navigation/base channel replayed, upper-body channel live

The hybrid is built by hooking the same seam already proven live in P0b:
`SonicDecoupledWbcAgent.get_action` is the point where a `vla_cmd` popped from
the action queue is about to be consumed by the whole-body controller. This
hook overwrites the requested fields on that popped command with the tape's
recorded values *before* it reaches the WBC step, so the WBC always runs on
the (possibly hybrid) command using live proprioception -- nothing about the
WBC itself is touched.

Frozen upstream, same as G0:
    SIMPLE b49c1aea2dd57309bb533219d0d34d6020f3d943
    Psi0   9ad917526394c1cacc72dba08562629936505987

Frozen operating point: force=100N, both directions, the same 30 matched
configs as G0, using G0's own recorded push tick per config so the disturbance
is identical to the one G0 already measured a sign flip under.
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

from topic24_runner import (  # noqa: E402
    TASKS, VirtualClock, extend, _make_sonic_config, Pusher, PUSH_HORIZON,
)

FORCE_N = 100.0


def find_tape(tape_root: Path, dr_level: str, eps_idx: int) -> Path:
    for sub in sorted(tape_root.iterdir()):
        cand = sub / f"{dr_level}_cfg{eps_idx}.json"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no G0 tape found for {dr_level}:{eps_idx} under {tape_root}")


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

    _replay_state: dict[str, Any] = {
        "row": None, "replay_nav": False, "replay_upper": False,
        # Structural proof the intervention fired, in the spirit of P0b: a
        # condition that claims to replay a channel must be shown to have
        # actually overwritten it on every control tick.
        "nav_overwrites": 0, "upper_overwrites": 0,
    }

    # Capture the original before rebinding, or the hook would call itself.
    _orig_sonic_get_action = SonicDecoupledWbcAgent.get_action

    def _hybrid_hook(self, observation, instruction=None, **kwargs):
        cmd = _orig_sonic_get_action(self, observation, instruction, **kwargs)
        if cmd is not None and cmd.type == "vla_cmd" and _replay_state["row"] is not None:
            row = _replay_state["row"]
            if _replay_state["replay_nav"]:
                cmd.parameters["navigate_cmd"] = np.asarray(row["navigate_cmd"], dtype=np.float32)
                cmd.parameters["base_height_command"] = np.asarray(
                    row["base_height_command"], dtype=np.float32
                )
                _replay_state["nav_overwrites"] += 1
            if _replay_state["replay_upper"]:
                cmd.parameters["target_upper_body_pose"] = dict(row["target_upper_body_pose"])
                _replay_state["upper_overwrites"] += 1
        return cmd

    SonicDecoupledWbcAgent.get_action = _hybrid_hook

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str, str]] = set()
    if out_path.exists() and args.resume:
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["config_id"], r["direction"], r["condition"]))
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
    tape_root = Path(args.tape_root)

    # condition -> (replay_nav, replay_upper), matching g1_core.py's CONDITIONS
    COND_SPEC = {"LR": (False, True), "RL": (True, False)}

    for eps_idx in range(args.episode_start, min(ds.num_episodes, args.episode_start + args.num_episodes)):
        config_id = f"{dr_level}:{eps_idx}"
        tape = json.loads(find_tape(tape_root, dr_level, eps_idx).read_text())
        push_tick = int(tape["push_tick"])

        for direction in args.directions:
            for condition in args.conditions:
                key = (config_id, direction, condition)
                if key in done:
                    continue
                replay_nav, replay_upper = COND_SPEC[condition]
                t0 = _real_time.perf_counter()

                random.seed(args.config_seed_base + eps_idx)
                np.random.seed(args.config_seed_base + eps_idx)
                env_conf, _ = get_episode_lerobot(ds, eps_idx)
                observation, info = env.reset(options={"state_dict": env_conf})
                sonic_env = raw_env.unwrapped
                model, data = sonic_env.mjModel, sonic_env.mjData

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

                pusher = Pusher(model, data, FORCE_N, direction, push_tick, control_dt)
                vla_tape = extend(tape["vla"], horizon)
                _replay_state["replay_nav"] = replay_nav
                _replay_state["replay_upper"] = replay_upper
                _replay_state["nav_overwrites"] = 0
                _replay_state["upper_overwrites"] = 0

                door_trace: list[float] = []
                frame_idx = 0
                server_queries = 0
                exhausted = False
                episode_over = False
                while not episode_over and frame_idx < horizon:
                    pusher.step(frame_idx)
                    _replay_state["row"] = (
                        vla_tape[frame_idx] if frame_idx < len(vla_tape) else vla_tape[-1]
                    )
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
                    observation, _r, terminated, truncated, info = env.step(action_cmd)
                    episode_over = terminated or truncated
                    frame_idx += 1
                    door_trace.append(float(data.joint(spec["effect_joint"]).qpos[0]))

                _replay_state["row"] = None
                if pusher is not None:
                    data.xfrc_applied[pusher.body] = np.zeros(6)
                official_success = bool(raw_env.unwrapped._success)

                row = {
                    "task": spec["short_name"], "env_id": args.env_id,
                    "config_id": config_id, "dr_level": dr_level,
                    "condition": condition, "force_n": FORCE_N, "direction": direction,
                    "nav_replayed": replay_nav, "upper_replayed": replay_upper,
                    "success": official_success,
                    "steps": frame_idx, "horizon": horizon,
                    "tape_exhausted_early": exhausted,
                    "server_queries": server_queries,
                    "nav_overwrites": _replay_state["nav_overwrites"],
                    "upper_overwrites": _replay_state["upper_overwrites"],
                    "push_tick": push_tick,
                    "push_applied": bool(pusher.applied_ticks > 0),
                    "push_applied_ticks": pusher.applied_ticks,
                    "effect_qpos": door_trace[-1] if door_trace else None,
                    "effect_predicate_reached": any(spec["predicate"](q) for q in door_trace),
                    "clock": "virtual",
                    "seconds": round(_real_time.perf_counter() - t0, 1),
                }
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"[{config_id}/{direction}/{condition}] success={official_success} "
                    f"steps={frame_idx} push@{push_tick} applied={pusher.applied_ticks} "
                    f"q={server_queries} navovr={row['nav_overwrites']} "
                    f"upovr={row['upper_overwrites']} ({row['seconds']}s)", flush=True,
                )

    raw_env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="simple/G1WholebodyCloseDoorTeleop-v0")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tape-root", required=True, help="root dir containing G0's per-worker tape subdirs")
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=22085)
    p.add_argument("--sim-mode", default="mujoco_isaac")
    p.add_argument("--conditions", nargs="+", default=["LR", "RL"], choices=["LR", "RL"])
    p.add_argument("--directions", nargs="+", default=["left", "right"], choices=["left", "right"])
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
