# RUN_LOCAL_AGENT — Topic 23 G0

You are running the first behavioral identification test for:

> **Do Robot Foundation Policies Learn Motor Equivalence Classes?**

Do not redesign the scientific question while running it.

> **Revision 2, 2026-08-24.** The condition panel and the intervention seam changed
> after the first implementation pass. See
> [VALIDATION_AUDIT.md](VALIDATION_AUDIT.md#revision-2-2026-08-24). The runner is
> now checked in as `topic23_runner.py`; you should not need to write rollout code.

## Frozen upstream

```text
SIMPLE b49c1aea2dd57309bb533219d0d34d6020f3d943
Psi0   9ad917526394c1cacc72dba08562629936505987
```

Frozen task panel (both, decided before any outcome was seen):

```text
simple/G1WholebodyCloseDoorTeleop-v0
simple/G1WholebodyOpenFaucetTeleop-v0
```

Released Psi0 SIMPLE checkpoints, `ckpt_40000`, one per task.

`*Teleop` tasks run through the **decoupled-WBC** entry point
(`simple/cli/eval_decoupled_wbc.py`, agent `psi0_decoupled_wbc`). Do not use
`eval.py` / `psi0`; that is the path for `*MP` tasks.

## Serve the policy

```bash
cd $PSI0 && ./.venv-psi/bin/python serve_psi0_g0.py --host 0.0.0.0 --port 22085 \
  --policy psi0 --run-dir=$RUN_DIR --ckpt-step=40000 --action-exec-horizon=24 --rtc
```

One client per server. `psi0_decoupled_wbc` sends no session id, so two concurrent
SIMPLE clients would corrupt each other's RTC state. Use a separate port and GPU
per task.

## Run the panel

```bash
cd $SIMPLE && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python \
  /path/to/23_motor_equivalence_classes/topic23_runner.py \
  --env-id simple/G1WholebodyCloseDoorTeleop-v0 \
  --data-dir $EVAL/G1WholebodyCloseDoorTeleop-v0/dr-level-0 \
  --out records/close_door_panel.jsonl \
  --port 22085 --sim-mode mujoco_isaac \
  --conditions canonical right_frozen right_disabled left_disabled both_arms_disabled full_hold \
  --num-episodes 10 --resume
```

Repeat for `dr-level-1` and `dr-level-2` into the same JSONL. Config ids are
namespaced by DR level, so the three levels pool into one matched-config table of
30 per task.

## Conditions

| condition | what it does | what it rules out |
| --- | --- | --- |
| `canonical` | nothing | — |
| `right_frozen` | right arm+hand held at the pose they already had | the arm has no motor program to substitute for |
| `right_disabled` | right arm+hand retracted to the neutral at-side pose and held | — (this is the scientific condition) |
| `left_disabled` | same, left side | any-arm-loss degradation |
| `both_arms_disabled` | both arms retracted, locomotion free | body/base-only route |
| `full_hold` | both arms retracted, waist frozen, base motion zeroed | accidental / environment-only success |
| `oracle_right_disabled` | scripted alternative under `right_disabled`'s constraint | the intervention made the task impossible |

The clamp is applied **after** the whole-body controller, on `target_q` /
`left_hand_q` / `right_hand_q`, and ramps in during stabilization so the policy's
first observation already contains the constraint. Only `full_hold`'s base freeze
is applied pre-WBC on the queued `vla_cmd`.

Do not alter the image/language observation. The physical consequence of the held
limb must reach the policy naturally through vision and proprioception.

## Record format

One JSON object per config/condition. `success` is always
`env.unwrapped._success`, the unmodified upstream evaluator. It is **not** derived
from a terminal qpos sample: these tasks require the object predicate to persist
while the upstream reward accumulator reaches `success_criteria`.

Route attribution is measured, not asserted: the runner scans MuJoCo contact pairs
each step and records which robot part touched the task object in the window
around the moment the task predicate is first satisfied.

```json
{
  "task": "close_door", "config_id": "dr-level-0:3", "condition": "right_disabled",
  "success": true, "effect_qpos": -0.177, "effect_predicate_reached": true,
  "route_verified": true, "canonical_right_route": false,
  "contact_parts_at_close": ["left_hand"], "contact_parts_ever": ["left_hand", "torso"],
  "right_arm_clamp_leak_rad": 0.03, "right_arm_excursion_rad": 0.02,
  "left_arm_motion_l2": 1.73, "torso_motion_l2": 0.44, "first_closed_step": 269
}
```

## Analyze

```bash
pytest -q tests/test_g0_core.py
python g0_core.py records/close_door_panel.jsonl --out g0_result.json
```

## Stop rules

Stop and report without tuning on any prerequisite failure:

```text
INSUFFICIENT_MATCHED_CONFIGS
PREREQUISITE_FAIL_CANONICAL
PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM
PREREQUISITE_FAIL_ROUTE_NOT_RIGHT_SIDE
PREREQUISITE_FAIL_INTERVENTION_LEAK
PREREQUISITE_FAIL_BODY_ONLY_ROUTE
PREREQUISITE_FAIL_ALTERNATIVE_FEASIBILITY
PREREQUISITE_FAIL_NEGATIVE_CONTROL
```

A prerequisite failure on one task is a result about **that task**, not about the
hypothesis. Run the other frozen panel task and report both. Do not add a third.

If the result is `NO_EVIDENCE_IN_PSI0_G0`, do not change thresholds, select
configs, weaken the clamp, or look for a new task after seeing the number.

## What counts as the first real signal

```text
canonical succeeds
right_frozen fails            <- the arm was actually doing something
oracle_right_disabled succeeds
right_disabled succeeds
both_arms_disabled fails
full_hold fails
```

plus a verified non-right-side contact route on the constrained successes.
