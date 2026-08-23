# RUN_LOCAL_AGENT — Topic 23 G0

You are running the first behavioral identification test for:

> **Do Robot Foundation Policies Learn Motor Equivalence Classes?**

Do not redesign the scientific question while running it.

## Frozen upstream

Use:

```text
SIMPLE b49c1aea2dd57309bb533219d0d34d6020f3d943
Psi0   9ad917526394c1cacc72dba08562629936505987
```

Start with:

```text
simple/G1WholebodyCloseDoorTeleop-v0
```

and the released Psi0 checkpoint that already passes the canonical SIMPLE evaluation.

## What to implement

Reuse the official Psi0 SIMPLE server/evaluator path as much as possible.

The only required custom hook is between:

```text
decoded policy action
        ↓
[Topic 23 motor intervention]
        ↓
SIMPLE controller/env step
```

Import:

```python
from g0_core import Condition, realized_motion_l2
from g0_simple_psi0 import (
    apply_motor_condition,
    assert_g1_locomanip_contract,
    make_record,
    read_effect_state,
)
```

Do not alter the image/language observation to make the policy "know" the condition. The physical/proprioceptive consequence of the held right arm must appear naturally on subsequent closed-loop observations.

## Conditions

For every fixed config/episode seed, run all of:

```text
canonical
oracle_right_disabled
right_disabled
full_hold
```

Use identical environment initialization for the four conditions.

### canonical

No action intervention.

### oracle_right_disabled

A separately constructed left-hand / alternative-body oracle is run under the exact same right-side hold as `right_disabled`.

This is a feasibility prerequisite. Do not substitute a verbal argument that "the left hand should reach."

### right_disabled

After policy inference:

```python
action = apply_motor_condition(action, state, "right_disabled")
```

This holds `right_arm` and `right_hand` at current state but leaves the rest of the body controllable.

### full_hold

After inference:

```python
action = apply_motor_condition(action, state, "full_hold")
```

This removes intentional body motion and estimates accidental/environment-only task completion.

## Record format

Write one terminal JSON object per condition/config:

```json
{
  "task": "close_door",
  "env_id": "simple/G1WholebodyCloseDoorTeleop-v0",
  "config_id": "0",
  "condition": "right_disabled",
  "success": true,
  "effect_qpos": -0.21,
  "effect_predicate_reached": true,
  "route_verified": true,
  "left_arm_motion_l2": 1.73,
  "torso_motion_l2": 0.44
}
```

There are **two distinct task signals** and they must not be conflated:

1. `success` = the official unmodified SIMPLE evaluator/check-success result for the episode;
2. `effect_qpos` / `effect_predicate_reached` = the raw door/faucet object-state predicate read from MuJoCo.

The audited SIMPLE tasks accumulate reward while the object predicate remains satisfied before declaring official success. Therefore **do not** derive `success` from one terminal qpos sample.

Use:

```python
effect = read_effect_state(mujoco_env, env_id)
row = make_record(
    env_id=env_id,
    config_id=config_id,
    condition=condition,
    effect=effect,
    official_success=official_upstream_success,
    ...
)
```

For successful `right_disabled` rollouts, save video/contact information and set `route_verified` only after confirming a non-canonical physical route actually caused the object effect.

## Sample

Use at least **20 matched configs** that have all four conditions.

Do not rerun only failed constrained episodes until they become successes.

## Analyze

```bash
python g0_core.py records.jsonl --out g0_result.json
```

Run unit tests first:

```bash
pytest -q tests/test_g0_core.py
```

## Stop rules

Stop and report without tuning if any prerequisite fails:

```text
PREREQUISITE_FAIL_CANONICAL
PREREQUISITE_FAIL_ALTERNATIVE_FEASIBILITY
PREREQUISITE_FAIL_NEGATIVE_CONTROL
INSUFFICIENT_MATCHED_CONFIGS
```

If the result is:

```text
NO_EVIDENCE_IN_PSI0_G0
```

do not change thresholds, choose convenient configs, alter the right-side hold, or search for a better task after seeing the result.

Return the raw JSONL, aggregate report, exact commit/checkpoint, and videos/contact traces for any constrained successes.

## What counts as the first real signal

The cleanest event is:

```text
canonical succeeds
oracle_right_disabled succeeds
right_disabled succeeds
full_hold fails
```

plus verified non-canonical realized motion/contact.

That is a behavior-level motor-equivalence event. Everything else is secondary until this exists.
