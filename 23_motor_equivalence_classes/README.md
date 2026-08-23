# 23 — Do Robot Foundation Policies Learn Motor Equivalence Classes?

> **G0 COMPLETE — BOTH FROZEN PANEL TASKS FAIL PREREQUISITES.**
> Registered 2026-08-24; condition panel revised the same day (revision 2); G0 run
> and reported the same day. See [G0_RESULTS.md](G0_RESULTS.md).
>
> CloseDoor: `PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM` (canonical 30/30, but
> freezing the right arm costs 0.033 and removing both arms costs nothing).
> OpenFaucet: `PREREQUISITE_FAIL_CANONICAL` (canonical 11/30 measured, 10/30
> published, against a 0.70 bar).
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
- yet the policy's own canonical solution is strongly right-lateralized.

> **Corrected 2026-08-24.** The registration originally cited
> `Task.decompose()`'s `hand_uid="dex3_right"` / `lock_links=["left_hand_palm_link"]`
> as evidence of a right-handed demonstrator. That is wrong for these two tasks:
> `decompose()` drives the CuRobo datagen path used by the `*MP` tasks, while both
> of these are `*Teleop` tasks whose data was human teleoperated. The laterality
> evidence is now the measured `right_frozen` vs `left_disabled` contrast instead.
> See [VALIDATION_AUDIT.md](VALIDATION_AUDIT.md#r28--decompose-is-not-evidence-about-the-teleop-demonstrations).

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

> **Revised 2026-08-24 (revision 2).** The original four-condition panel
> (`canonical` / `oracle_right_disabled` / `right_disabled` / `full_hold`) was
> replaced after the intervention was implemented against the real upstream stack
> and a contact-level route probe was run. See
> [VALIDATION_AUDIT.md](VALIDATION_AUDIT.md#revision-2-2026-08-24) for the evidence.
> In short: a `right_disabled` success is consistent with three different worlds,
> and the original panel could not tell them apart.

```text
W1  the policy re-planned the task onto another effector        <- the claim
W2  the arm was never articulating; the hand was a passive
    bumper carried into the object by locomotion
W3  no arm was needed at all; the torso/base does the work
```

Every configuration is evaluated under all seven conditions.

### A. `canonical`

Unmodified policy rollout. Verifies the task is alive.

### B. `right_frozen` — is there a right-arm motor program at all?

The right arm and hand are held at the configuration they already had, so the limb
loses its **articulation** but stays where it is. This is the locked-joint fault
model used in the VLA fault literature.

If freezing the arm in place costs the policy nothing, the canonical solution
contains no right-arm motor program and there is nothing for an equivalent route
to substitute for. This kills W2.

### C. `right_disabled` — effector removal

The right arm and hand are ramped to the robot's neutral at-side pose during
stabilization and PD-held there for the whole episode, so the limb is unavailable
as an effector before the policy is ever engaged. Left arm/hand, torso and
locomotion remain available. This is the scientific condition.

### D. `left_disabled` — laterality control

The same retract-and-hold applied to the *left* side. Separates "removing the
demonstrator's hand hurts" from "removing any arm hurts".

### E. `both_arms_disabled` — body-only route probe

Both arms retracted, locomotion free. If the task survives losing both arms, any
`right_disabled` success is a body/base route and says nothing about one arm
standing in for the other. This kills W3.

### F. `full_hold` — accidental success control

Both arms retracted, waist frozen, `(vx, vy, vyaw)` zeroed and heading/height held
at their stabilized values. Estimates environment-only success and catches broken
interventions.

### G. `oracle_right_disabled` — feasibility prerequisite

A scripted / teleoperated **alternative** solution under exactly condition C's
constraint. Proves the intervention did not make the task physically impossible.
This must be demonstrated before any policy failure is interpreted.

### Where the intervention is applied

At the **actuator boundary**, after the GR00T whole-body controller, on
`target_q` / `left_hand_q` / `right_hand_q`. The `*Teleop` tasks run through
`eval_decoupled_wbc`, where a WBC sits between the policy action and the
simulator; editing the policy's action groups before the WBC lets the controller
re-solve around the constraint. Only `full_hold`'s base freeze is applied
pre-WBC, on the queued `vla_cmd`, because the lower-body RL policy consumes it.

The clamp is verified per episode: `right_arm_clamp_leak_rad` is the largest
realized deviation from the held target, and a leaking clamp is a prerequisite
failure rather than a result.

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

Defaults in `g0_core.py`, evaluated in this order:

| # | gate | verdict on failure |
| --- | --- | --- |
| 0 | matched configurations `>= 20` | `INSUFFICIENT_MATCHED_CONFIGS` |
| 1 | canonical success `>= 0.70` | `PREREQUISITE_FAIL_CANONICAL` |
| 2 | `canonical - right_frozen >= 0.20` | `PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM` |
| 3 | canonical right-side contact rate `>= 0.70` | `PREREQUISITE_FAIL_ROUTE_NOT_RIGHT_SIDE` |
| 4 | clamp leak `<= 0.20 rad` | `PREREQUISITE_FAIL_INTERVENTION_LEAK` |
| 5 | `both_arms_disabled <= 0.10` | `PREREQUISITE_FAIL_BODY_ONLY_ROUTE` |
| 6 | `oracle_right_disabled >= 0.70` | `PREREQUISITE_FAIL_ALTERNATIVE_FEASIBILITY` |
| 7 | `full_hold <= 0.10` | `PREREQUISITE_FAIL_NEGATIVE_CONTROL` |

Gate 2 is deliberately a **behavioural** criterion, not a kinematic threshold: it
asks whether removing the arm's articulation costs the policy task success, so it
needs no tuned cut on joint excursion.

Only if every gate passes is the substitution test evaluated:

- right-disabled substitution rate `>= 0.20`;
- at least `5` paired substitution events;
- paired `right_disabled - full_hold` bootstrap mean `>= 0.20`.

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
