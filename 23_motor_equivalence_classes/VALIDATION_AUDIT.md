# Validation Audit — Topic 23

Date: 2026-08-24

## Decision

**REGISTER.**

This audit compared the three active embodied search candidates B/C/D against the repository's current selection rules and the newest adjacent work.

## B — How Do Robot Foundation Policies Generalize Actions?

**Keep provisional; do not register yet.**

The broad question is good, but its first identification object is not yet clean enough.

ICLR 2026 `Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative` already gives a direct action-generation account for three representative families:

- Diffusion Policy: strong action memorization / retrieval;
- ACT: interpolation;
- GR00T: interpolation plus stronger OOD robustness.

The remaining proposed axis — composition vs genuine extrapolation/synthesis and how it changes with foundation scaling — is interesting, but still needs a behavior-level definition that does not collapse into a tunable trajectory-similarity statistic.

Repository rule triggered: **conceptual identifiability before scaling**.

## C — Why Does Task Decomposition Help Robot Foundation Policies?

**Downgrade from active shortlist; do not register.**

The question was initially attractive because oracle atomic decomposition could separate planner intelligence from low-level steerability.

However, collision increased substantially in July–August 2026:

- `Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation` explicitly centers the semantic-planning ↔ executable-low-level gap and constructs canonical executable skill primitives / tractability constraints.
- `Beyond Flat Policies: Hierarchical Post-Training for Embodied Agents in Robotic Manipulation` explicitly identifies planner/executor subgoal distribution misalignment and aligns the executor to planner-generated subgoals.
- `What Matters in Orchestrating Robot Policies` already performs a broad hierarchical-VLA component study.

A fresh paper could still run cleaner causal controls, but novelty would increasingly rely on a narrower decomposition of an already-active mechanism.

Repository rule triggered: **do not keep shrinking scope to escape collision**.

## D — Do Robot Foundation Policies Learn Motor Equivalence Classes?

**Promote to Topic 23.**

### Scientific contrast

```text
demonstrator-route binding
vs
task-effect / motor-equivalence abstraction
```

### Why identification is unusually clean

SIMPLE already provides the separation needed by the hypothesis:

1. success is defined in the environment/object state;
2. automated demonstrations explicitly privilege the right hand and lock the left.

No learned probe, SAE, latent distance, or trajectory similarity is required for the primary endpoint.

### Exact source facts checked

At SIMPLE commit:

`b49c1aea2dd57309bb533219d0d34d6020f3d943`

`g1_wholebody_close_door_teleop.py`:

- success requires `articulate_joint_1 < -0.16`;
- demonstration decomposition uses `hand_uid="dex3_right"`;
- demonstration decomposition locks `left_hand_palm_link`.

`g1_wholebody_open_faucet_teleop.py`:

- success requires `articulate_joint_0 > 0.7 or < -0.7`;
- same right-hand / left-lock demonstration asymmetry.

At Psi0 commit:

`9ad917526394c1cacc72dba08562629936505987`

the G1 loco-manip modality exposes distinct left/right arm and hand state/action groups, allowing a transparent post-policy intervention.

### Topic 19 failure check

Topic 19 failed because joint-axis restoration was used as a proxy for task-space correction.

Topic 23 does not reuse that proxy.

The primary endpoint is the exact object state that defines task success. The actuator intervention only removes one motor route; it is not itself the dependent variable.

### Remaining confound and how it is handled

A right-arm block can make a task physically impossible even if we intuitively expect a left-hand solution.

Therefore the alternative-solution oracle is a **prerequisite**, not a control added after seeing failures.

If the oracle cannot solve the exact constrained environment, that configuration is invalid for the scientific test.

### Positive-result excitement test

If a pretrained whole-body policy spontaneously changes effector/body strategy while preserving a task effect that was demonstrated with one canonical route, that is a qualitatively stronger statement than ordinary OOD robustness.

### Then-what test

Both outcomes expose a concrete training target:

- positive: identify what pretraining diversity induces motor-equivalence abstraction;
- negative: deliberately train across goal-equivalent motor realizations.

## Collision conclusion

No direct 2025–2026 work found in this audit tests the same fixed-robot, fixed-task, fixed-world counterfactual on an already-trained robot foundation policy:

> remove the demonstrator's canonical effector solution while preserving a verified alternative solution, then measure whether the policy preserves the task effect through a different body realization.

Related work on whole-body redundancy, fault-tolerant control, cross-embodiment transfer and general VLA robustness is adjacent but does not answer this mechanism question.

## Registration consequence

Topic 23 should start with **one clean task-level behavioral G0**. Do not add representation analysis unless the constrained substitution event first exists at useful density.
