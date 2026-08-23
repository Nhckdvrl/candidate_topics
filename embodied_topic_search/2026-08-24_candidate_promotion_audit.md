# 2026-08-24 — B/C/D promotion audit

This note records the pre-G0 promotion decision and the same-day post-G0 outcome.

## Pre-G0 decision

```text
B — KEEP PROVISIONAL
C — DOWNGRADE / REMOVE FROM ACTIVE SHORTLIST
D — PROMOTE -> root Topic 23
```

## B — How Do Robot Foundation Policies Generalize Actions?

**Keep provisional.**

The question remains interesting, but the first clean identification experiment is not ready.

ICLR 2026 `Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative` already establishes a substantial part of the proposed mechanism spectrum on matched data:

```text
Diffusion Policy -> strong retrieval / memorization
ACT              -> interpolation
GR00T            -> interpolation + stronger OOD robustness
```

The remaining foundation-scale question — whether behavior moves toward composition or genuine synthesis/extrapolation — still needs a behavior-level definition that does not depend on a tunable trajectory-similarity decomposition.

Per repository policy, do not register until that identification problem is solved.

## C — Why Does Task Decomposition Help Robot Foundation Policies?

**Downgrade from active shortlist.**

The original mechanism split was:

```text
planner reasoning
vs
controller-support matching
vs
temporal handoff/reset
```

This was initially promising, but 2026 work has moved directly into the planner/executor interface:

- `What Matters in Orchestrating Robot Policies` systematically studies hierarchical VLA components and low-level steerability;
- `Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation` explicitly attacks the high-level semantic-plan / low-level executable-kinematics gap with canonical executable skill primitives and tractability constraints;
- `Beyond Flat Policies: Hierarchical Post-Training for Embodied Agents in Robotic Manipulation` explicitly identifies planner-generated-subgoal / executor distribution misalignment and aligns the executor to those subgoals.

A cleaner causal decomposition could still be publishable in isolation, but the broad mechanism angle is no longer empty enough to justify occupying a root Topic. Keeping it alive would increasingly require narrowing scope to escape collision.

## D — Do Robot Foundation Policies Learn Motor Equivalence Classes?

**Promoted to [`../23_motor_equivalence_classes/`](../23_motor_equivalence_classes/) before G0, then archived after G0.**

The pre-run reason for promotion was the apparent identification simplicity: SIMPLE exposes task success in environment/object state, so the experiment could measure the task effect directly rather than through a joint-space proxy.

### Correction to the original promotion rationale

The original audit incorrectly treated the `Task.decompose()` entries

```text
hand_uid="dex3_right"
lock_links=["left_hand_palm_link"]
```

as evidence that the CloseDoor/OpenFaucet `*Teleop` demonstrations were generated with that right-hand decomposition. They were not: `decompose()` belongs to the CuRobo datagen path used by the `*MP` tasks, while these panel tasks use human teleoperation data.

The laterality/canonical-route premise therefore had to be established behaviorally from the released policy itself, not inherited from `decompose()`.

## What G0 discovered

The first registered four-condition panel was:

```text
canonical
oracle_right_disabled
right_disabled
full_hold
```

On CloseDoor, the observed policy conditions looked like an overwhelming positive result:

```text
canonical       30/30
right_disabled  29/30
full_hold        0/30
paired diff      0.967, 95% CI [0.90, 1.00]
substitution events 29
```

But contact/kinematic inspection showed this was a false positive. Psi0 closes the door mainly by locomotion, carrying the already-low right hand into the door; shoulder and elbow articulation are tiny. The original intervention had not removed a causal right-arm motor program, so no substitution was required.

Revision 2 added:

```text
right_frozen
left_disabled
both_arms_disabled
```

before the substitution claim could be evaluated.

### CloseDoor final gate

```text
canonical                 30/30
right_frozen              29/30
canonical-right_frozen    0.033   < 0.20  FAIL
both_arms_disabled        30/30   > 0.10  FAIL
full_hold                  0/30
```

Verdict: `PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM`.

### OpenFaucet final gate

OpenFaucet does contain a genuine arm program, but released Psi0 canonical success was only:

```text
5/10 + 2/10 + 4/10 = 11/30 = 0.367
```

which reproduces the published `10/30` regime and fails the frozen `>=0.70` competence prerequisite.

Verdict: `PREREQUISITE_FAIL_CANONICAL`.

## Final status

```text
B — KEEP PROVISIONAL
C — DOWNGRADED / OUT OF ACTIVE SHORTLIST
D — PROMOTED AS TOPIC 23, THEN ARCHIVED
```

Topic 23's broad scientific question is not falsified. The frozen Psi0 + SIMPLE panel simply contains no task that is simultaneously:

```text
policy competent
AND
causally dependent on the motor program being removed
```

Searching a third task after these two frozen failures would be post-hoc task shopping.

The most important reusable lesson is:

> **Before interpreting preserved task success as motor substitution, verify that the intervention actually removes a causal motor program used by the canonical behavior. Statistical confidence cannot rescue a non-identifying intervention.**

See [`../23_motor_equivalence_classes/ARCHIVE_SUMMARY.md`](../23_motor_equivalence_classes/ARCHIVE_SUMMARY.md) and [`../23_motor_equivalence_classes/G0_RESULTS.md`](../23_motor_equivalence_classes/G0_RESULTS.md).