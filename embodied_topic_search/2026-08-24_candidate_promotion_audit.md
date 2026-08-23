# 2026-08-24 — B/C/D promotion audit

This note supersedes the 2026-08-23 provisional-shortlist status in this search directory. It records the decision taken **before** running the new experiment.

## Decision

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

**Promoted to [`../23_motor_equivalence_classes/`](../23_motor_equivalence_classes/).**

The decisive reason is identification simplicity.

SIMPLE provides tasks where success is defined by an environment effect while demonstrations use one canonical motor realization:

```text
CloseDoor:
  success -> door joint state
  demo    -> dex3_right, left hand locked

OpenFaucet:
  success -> faucet joint state
  demo    -> dex3_right, left hand locked
```

Therefore we can keep robot/task/world/language fixed, remove the canonical right-side route, first verify an alternative solution with an oracle, and then directly ask whether the frozen policy preserves the task effect using a different body solution.

This avoids the Topic 19 failure mode because the dependent variable is **task/outcome space from the start**, not a joint-space projection.

## New registered experiment

Topic 23 freezes four matched conditions:

```text
canonical
oracle_right_disabled
right_disabled
full_hold
```

and scores the exact task-defined object effect.

No hidden-state analysis, action-manifold metric, or post-hoc trajectory-similarity search is part of G0.

## Status bookkeeping

The older D candidate file is preserved as a historical search artifact even though its header says provisional. **This promotion note is the newer status record**, and the root Topic 23 directory is now authoritative.
