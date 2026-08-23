"""SIMPLE/Psi0 integration helpers for Topic 23 G0.

Frozen upstream references:
  SIMPLE b49c1aea2dd57309bb533219d0d34d6020f3d943
  Psi0   9ad917526394c1cacc72dba08562629936505987

This module does not reimplement the upstream evaluator. It supplies the pieces
that should be inserted into a local SIMPLE rollout:
  1) read the task-defining object effect directly from MuJoCo;
  2) record the *official upstream episode success* separately from the raw effect.

The motor intervention itself lives in `topic23_runner.MotorClamp`, applied at the
actuator boundary after the whole-body controller. Keep the policy observation
untouched: on later closed-loop steps it naturally sees the consequences of the
constrained body through vision and proprioception.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from g0_core import Condition, task_effect_success


SIMPLE_COMMIT = "b49c1aea2dd57309bb533219d0d34d6020f3d943"
PSI0_COMMIT = "9ad917526394c1cacc72dba08562629936505987"

# Source-level frozen contracts checked on 2026-08-24.
TASKS = {
    "simple/G1WholebodyCloseDoorTeleop-v0": {
        "short_name": "close_door",
        "effect_joint": "articulate_joint_1",
        "demo_hand": "dex3_right",
        "demo_locked_link": "left_hand_palm_link",
    },
    "simple/G1WholebodyOpenFaucetTeleop-v0": {
        "short_name": "open_faucet",
        "effect_joint": "articulate_joint_0",
        "demo_hand": "dex3_right",
        "demo_locked_link": "left_hand_palm_link",
    },
}


@dataclass(frozen=True)
class EffectState:
    task: str
    qpos: float
    predicate_reached: bool


def read_effect_state(mujoco_env: Any, env_id: str) -> EffectState:
    """Read the raw object-state predicate underlying SIMPLE's task reward.

    This is deliberately *not* called official success. In the audited tasks,
    SIMPLE's `check_success` requires the object predicate to remain true long
    enough for an internal reward accumulator to reach `success_criteria`.
    The rollout runner must therefore report upstream episode success separately.
    """
    if env_id not in TASKS:
        raise KeyError(f"env_id not in frozen Topic 23 panel: {env_id}")
    spec = TASKS[env_id]
    qpos = float(mujoco_env.mjData.joint(spec["effect_joint"]).qpos[0])
    return EffectState(
        task=spec["short_name"],
        qpos=qpos,
        predicate_reached=task_effect_success(spec["short_name"], qpos),
    )


def make_record(
    *,
    env_id: str,
    config_id: str | int,
    condition: Condition | str,
    effect: EffectState,
    official_success: bool,
    route_verified: bool | None = None,
    left_arm_motion_l2: float | None = None,
    torso_motion_l2: float | None = None,
) -> dict[str, Any]:
    """Create one terminal row consumable by g0_core.py.

    `official_success` must come from the unmodified upstream SIMPLE evaluator,
    not from `effect.predicate_reached`.
    """
    c = Condition(condition)
    row = {
        "task": TASKS[env_id]["short_name"],
        "env_id": env_id,
        "config_id": str(config_id),
        "condition": c.value,
        "success": bool(official_success),
        "effect_qpos": float(effect.qpos),
        "effect_predicate_reached": bool(effect.predicate_reached),
        "route_verified": route_verified,
    }
    if left_arm_motion_l2 is not None:
        row["left_arm_motion_l2"] = float(left_arm_motion_l2)
    if torso_motion_l2 is not None:
        row["torso_motion_l2"] = float(torso_motion_l2)
    return row
