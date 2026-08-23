"""SIMPLE/Psi0 integration primitives for Topic 19 G0.

This file intentionally keeps the scientific core separate from the exact rollout
plumbing. It is meant to be copied/imported inside a checkout containing SIMPLE
and Psi0. The frozen upstream contracts audited for this topic are:
  SIMPLE b49c1aea2dd57309bb533219d0d34d6020f3d943
  Psi0   9ad917526394c1cacc72dba08562629936505987
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from g0_core import (
    aggregate_by_episode,
    bootstrap_mean_ci,
    construct_perturbations,
    finite_geometry_gate,
    response_metrics,
    verdict,
)

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
RIGHT_EE_BODY = "right_hand_palm_link"


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Run this integration inside the SIMPLE MuJoCo environment") from exc
    return mujoco


def right_arm_addresses(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return qpos and qvel/dof addresses for the seven named G1 right-arm joints."""
    mujoco = _require_mujoco()
    qpos, dof = [], []
    for name in RIGHT_ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"missing SIMPLE G1 joint: {name}")
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"expected hinge joint for {name}")
        qpos.append(int(model.jnt_qposadr[jid]))
        dof.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qpos, dtype=int), np.asarray(dof, dtype=int)


def right_wrist_jacobian(model: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    """3x7 translational and rotational Jacobian for right_hand_palm_link."""
    mujoco = _require_mujoco()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY)
    if bid < 0:
        raise KeyError(f"missing EE body: {RIGHT_EE_BODY}")
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, bid)
    _, dof = right_arm_addresses(model)
    return jacp[:, dof].copy(), jacr[:, dof].copy()


def snapshot_mujoco(data: Any) -> dict[str, Any]:
    """Snapshot fields needed for no-integration counterfactual rendering/query."""
    fields = [
        "qpos", "qvel", "act", "ctrl", "qacc_warmstart",
        "mocap_pos", "mocap_quat", "userdata",
    ]
    snap: dict[str, Any] = {"time": float(data.time)}
    for name in fields:
        if hasattr(data, name):
            snap[name] = np.array(getattr(data, name), copy=True)
    return snap


def restore_mujoco(model: Any, data: Any, snap: dict[str, Any]) -> None:
    mujoco = _require_mujoco()
    data.time = snap["time"]
    for name, value in snap.items():
        if name == "time" or not hasattr(data, name):
            continue
        dst = getattr(data, name)
        if np.size(dst):
            dst[...] = value
    mujoco.mj_forward(model, data)


def apply_right_arm_delta(model: Any, data: Any, delta: np.ndarray) -> None:
    """Apply a physical qpos perturbation and forward MuJoCo without stepping time."""
    mujoco = _require_mujoco()
    d = np.asarray(delta, dtype=np.float64).reshape(7)
    qpos, dof = right_arm_addresses(model)
    data.qpos[qpos] += d
    data.qvel[dof] = 0.0
    mujoco.mj_forward(model, data)


def ee_pose(model: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return EE position and 3x3 rotation matrix."""
    mujoco = _require_mujoco()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY)
    if bid < 0:
        raise KeyError(RIGHT_EE_BODY)
    return np.array(data.xpos[bid], copy=True), np.array(data.xmat[bid], copy=True).reshape(3, 3)


def rotation_distance(r0: np.ndarray, r1: np.ndarray) -> float:
    rel = np.asarray(r0).T @ np.asarray(r1)
    x = (np.trace(rel) - 1.0) / 2.0
    return float(np.arccos(np.clip(x, -1.0, 1.0)))


def build_pair_from_sim(model: Any, data: Any, epsilon: float = 0.08):
    """Construct deltas and verify their finite FK geometry before querying Psi0."""
    jp, jr = right_wrist_jacobian(model, data)
    pair = construct_perturbations(jp, jr, epsilon=epsilon)
    snap = snapshot_mujoco(data)
    p0, r0 = ee_pose(model, data)

    restore_mujoco(model, data, snap)
    apply_right_arm_delta(model, data, pair.task)
    pt, _ = ee_pose(model, data)

    restore_mujoco(model, data, snap)
    apply_right_arm_delta(model, data, pair.null)
    pn, rn = ee_pose(model, data)

    restore_mujoco(model, data, snap)
    ok, diag = finite_geometry_gate(
        task_translation_m=float(np.linalg.norm(pt - p0)),
        null_translation_m=float(np.linalg.norm(pn - p0)),
        null_rotation_rad=rotation_distance(r0, rn),
    )
    return pair, ok, diag


def analyze_jsonl(path: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    """Score paired raw targets collected by the environment-specific runner."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("geometry_valid", False):
            continue
        m = response_metrics(
            row["base_right_arm_target"],
            row["task_right_arm_target"],
            row["null_right_arm_target"],
            row["delta_task"],
            row["delta_null"],
        )
        row.update({
            "accommodation_task": m.accommodation_task,
            "accommodation_null": m.accommodation_null,
            "correction_task": m.correction_task,
            "correction_null": m.correction_null,
            "delta_correction": m.delta_correction,
        })
        rows.append(row)
    if not rows:
        raise RuntimeError("no geometry-valid paired rows")

    episode_values = aggregate_by_episode(rows)
    mean, lo, hi = bootstrap_mean_ci(episode_values)
    report = {
        "n_rows": len(rows),
        "n_episodes": int(episode_values.size),
        "mean_delta_correction": mean,
        "bootstrap_95_ci": [lo, hi],
        "mean_correction_task": float(np.mean([r["correction_task"] for r in rows])),
        "mean_correction_null": float(np.mean([r["correction_null"] for r in rows])),
        "mean_accommodation_task": float(np.mean([r["accommodation_task"] for r in rows])),
        "mean_accommodation_null": float(np.mean([r["accommodation_null"] for r in rows])),
        "verdict": verdict(mean, lo, hi),
    }
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("records", help="paired G0 JSONL")
    p.add_argument("--out", default="g0_result.json")
    args = p.parse_args()
    print(json.dumps(analyze_jsonl(args.records, args.out), indent=2))


if __name__ == "__main__":
    main()
