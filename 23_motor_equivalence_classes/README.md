# 23 — Do Robot Foundation Policies Learn Motor Equivalence Classes?

> **ACTIVE / REGISTERED 2026-08-24**
>
> Source search candidate: [`embodied_topic_search/candidates/do_robot_foundation_policies_learn_motor_equivalence_classes.md`](../embodied_topic_search/candidates/do_robot_foundation_policies_learn_motor_equivalence_classes.md)
>
> First target: **Psi0 + SIMPLE**, using task/outcome-space measurements rather than a joint-space proxy.

## Natural question

> **Do robot foundation policies learn the task constraint, or the particular motor realization chosen by the demonstrator?**

If demonstrations always solve a task with the right hand, but the same task still has a physically valid left-hand / whole-body solution, what happens when the canonical right-hand route is removed at test time?

A task-level policy should be able, at least sometimes, to preserve the environmental effect while changing the motor realization. A trajectory-bound policy should keep trying the demonstrator's unavailable route or fail without discovering the alternative.

This is not generic visual robustness, cross-embodiment transfer, or fault-tolerant-control training. The first experiment holds fixed:

```text
same robot
same task
same world
same language
```

and changes only the **available motor solution set**.

## Why this was promoted

Among the current embodied search candidates, this one has the cleanest first identification experiment.

The important open-source asymmetry is already present in SIMPLE:

- `G1WholebodyCloseDoorTeleop`: the reward/success logic is grounded in an **object-state predicate** (`articulate_joint_1 < -0.16`), which must remain satisfied long enough for the reward accumulator to reach the official success criterion;
- `G1WholebodyOpenFaucetTeleop`: likewise, its task effect is grounded in `|articulate_joint_0| > 0.7`, again with persistence through the official reward/success logic;
- yet both automated demonstration decompositions explicitly use `hand_uid="dex3_right"` and `lock_links=["left_hand_palm_link"]`.

Thus the benchmark itself separates:

```text
what environmental effect defines the task
from
which motor realization generated the demonstrations
```

That gives a direct counterfactual instead of inventing a latent metric.

## Relation to Topic 19

Topic 19 was archived because a joint-axis projection did **not** identify task-space correction in a redundant arm.

Topic 23 starts from that lesson:

> **The primary dependent variable lives in task/outcome space.**

We do not infer task abstraction from whether an action points along a particular joint-space vector. If the claim is that the policy preserves the task while changing the body solution, we directly record:

1. the unmodified upstream SIMPLE episode success;
2. the raw task-defining door/faucet object coordinate and predicate;
3. whether the canonical right-side route was physically unavailable;
4. whether a non-canonical route actually occurred.

This is a new registered topic, not a post-hoc repair of Topic 19.

## Frozen upstream contracts

Audited on 2026-08-24:

- SIMPLE: `b49c1aea2dd57309bb533219d0d34d6020f3d943`
- Psi0: `9ad917526394c1cacc72dba08562629936505987`

Psi0's G1 loco-manip modality exposes separate:

```text
state:
left_hand, right_hand, left_arm, right_arm, rpy, height

action:
left_hand, right_hand, left_arm, right_arm,
rpy, height, torso_vx, torso_vy, torso_vyaw, target_yaw
```

The arm/hand/body-pose groups are absolute action targets in the audited modality config.

## G0: canonical-route removal

Start with **CloseDoor**, because Topic 19 already established that the released Psi0 checkpoint can solve the official-path task reliably.

Use a fixed panel of at least **20 matched SIMPLE configurations**. Every configuration is evaluated under all four conditions below.

### A. `canonical`

Unmodified policy rollout.

Purpose: verify that the task is alive.

### B. `oracle_right_disabled`

A scripted / teleoperated **alternative** solution under the exact same right-side intervention used in C.

Purpose: prove that the intervention did not make the task physically impossible.

This prerequisite must be demonstrated before interpreting a policy failure.

### C. `right_disabled`

Run the policy normally, then **after inference and before controller execution**, hold:

```text
right_arm  := current right_arm state
right_hand := current right_hand state
```

The policy still receives the real observation on the next step, including the consequences of the blocked right side.

Left arm/hand, torso and locomotion remain available.

This is the scientific condition.

### D. `full_hold`

Remove intentional whole-body motion:

- hold both arm/hand groups at current state;
- hold `rpy` and `height`;
- set torso velocities to zero;
- hold target yaw.

Purpose: estimate accidental/environment-only success and catch broken interventions.

## Primary endpoint

For each matched configuration:

```text
Y = official unmodified SIMPLE episode success
```

Alongside it, always log:

```text
effect_qpos
raw object-state predicate reached or not
```

Do **not** infer official episode success from one terminal qpos sample: in these tasks the object predicate must persist while the upstream reward accumulator reaches `success_criteria`.

The first aggregate contrast is:

```text
P(success | right_disabled)
-
P(success | full_hold)
```

and the most interpretable event is:

```text
right_disabled succeeds
AND
full_hold fails
```

among configurations that also pass canonical and alternative-oracle prerequisites.

## Secondary route verification

A task success under `right_disabled` is not by itself enough to call the event motor substitution.

For successful constrained episodes, verify the route using at least one of:

- contact links / contact pairs;
- realized left-arm / torso proprioceptive motion;
- video inspection on the small number of successful events.

The code records `route_verified` separately so this cannot silently become part of a tuned success metric.

## Frozen G0 gates

Defaults in `g0_core.py`:

- matched configurations: `>= 20`;
- canonical success: `>= 0.70`;
- oracle-right-disabled success: `>= 0.70`;
- full-hold success: `<= 0.10`;
- right-disabled substitution rate: `>= 0.20`;
- at least `5` paired substitution events.

A positive first-pass verdict is:

```text
PROMISING_MOTOR_SUBSTITUTION
```

only when the route is also verified for at least 80% of constrained successes.

If task-level constrained success exists but route evidence has not yet been checked:

```text
PROMISING_NEEDS_ROUTE_VERIFICATION
```

If the released Psi0 model shows no effect:

```text
NO_EVIDENCE_IN_PSI0_G0
```

This does **not** trigger metric/model fishing. The next decision should be made at the topic level: either run one predeclared second foundation-policy family on the same identification test, or stop.

## Why both positive and negative results matter

### Positive

The pretrained policy preserves the object-level goal while abandoning the demonstration's canonical effector.

That supports a real abstraction claim:

> physical pretraining can induce motor solutions organized around task effects rather than only demonstrated trajectories.

Natural method openings:

- motor-equivalence augmentation;
- effector/DoF dropout;
- same-effect contrastive objectives;
- effect-space supervision;
- deliberate collection of multiple body solutions per task.

### Negative

The policy repeatedly fails or persists on the unavailable right-side route despite a verified alternative solution.

That supports the opposite but still important conclusion:

> current robot foundation policies may remain more demonstrator-route-bound than their task-level success claims suggest.

The method opening is equally clear: train across goal-equivalent motor realizations rather than merely scaling demonstrations of one canonical solution.

## Files

- `g0_core.py` — frozen intervention/scoring logic; independent of SIMPLE.
- `g0_simple_psi0.py` — audited SIMPLE/Psi0 contract helpers.
- `tests/test_g0_core.py` — unit tests for the intervention and gates.
- `RUN_LOCAL_AGENT.md` — exact hand-off instructions for the server-side runner.
- `VALIDATION_AUDIT.md` — collision and identification audit.

## Immediate next step

Do **not** add hidden-state analysis.

First wire `apply_motor_condition()` into one local SIMPLE rollout loop and obtain four-condition JSONL records for CloseDoor. The earliest useful result is behavioral and task-level.
