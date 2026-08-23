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

1. reward/success is grounded in an environment/object-state predicate and its persistence;
2. automated demonstrations explicitly privilege the right hand and lock the left.

No learned probe, SAE, latent distance, or trajectory similarity is required for the primary endpoint.

### Exact source facts checked

At SIMPLE commit:

`b49c1aea2dd57309bb533219d0d34d6020f3d943`

`g1_wholebody_close_door_teleop.py`:

- the raw task predicate is `articulate_joint_1 < -0.16`; while it remains true, reward accumulates until the official success criterion is reached;
- demonstration decomposition uses `hand_uid="dex3_right"`;
- demonstration decomposition locks `left_hand_palm_link`.

`g1_wholebody_open_faucet_teleop.py`:

- the raw task predicate is `articulate_joint_0 > 0.7 or < -0.7`, likewise feeding the persistent reward/success logic;
- same right-hand / left-lock demonstration asymmetry.

At Psi0 commit:

`9ad917526394c1cacc72dba08562629936505987`

the G1 loco-manip modality exposes distinct left/right arm and hand state/action groups, allowing a transparent post-policy intervention.

### Topic 19 failure check

Topic 19 failed because joint-axis restoration was used as a proxy for task-space correction.

Topic 23 does not reuse that proxy.

The primary endpoint is the official upstream episode success, with the exact object coordinate/predicate logged alongside it. The actuator intervention only removes one motor route; it is not itself the dependent variable.

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

---

## Revision 2 (2026-08-24)

The G0 design registered earlier the same day was revised after it was implemented
against the real upstream stack and a contact-level route probe was run on
`simple/G1WholebodyCloseDoorTeleop-v0` with the released Psi0 `ckpt_40000`.

Nothing here was learned from an outcome comparison. All of it came from reading
upstream source and from a single canonical rollout instrumented at the MuJoCo
contact level.

### R2.1 — The intervention was applied above the whole-body controller

`*Teleop` tasks do **not** run through `simple/cli/eval.py`. The upstream entry
point is `eval_decoupled_wbc.py` with agent `psi0_decoupled_wbc`, in which the
decoded Psi0 action becomes an `ActionCmd("vla_cmd", ...)` that a GR00T whole-body
controller then re-solves into joint targets.

The registered design held the arm by rewriting the policy's action groups, i.e.
*before* the WBC. The WBC is free to re-solve around that, so the limb is not
reliably held. Measured on one config: commanded right-arm deviation from the hold
target was `0.32 rad` while the realized deviation was `0.15 rad` — the clamp was
partially, not fully, effective.

The clamp now runs at the actuator boundary (`target_q`, `left_hand_q`,
`right_hand_q`) and every episode records `right_arm_clamp_leak_rad`. A leaking
clamp is a prerequisite failure, not a result.

Only `full_hold`'s base freeze stays pre-WBC, on the queued `vla_cmd`, because the
lower-body RL policy is what consumes `navigate_cmd` / `base_height_command`. With
that fixed, `full_hold` behaves as a negative control should: door stays at its
initial `0.792 rad`, no robot–door contact, total arm path length `0.003 rad`.

### R2.2 — `navigate_cmd` semantics

`navigate_cmd = pred_action[32:36]` is `(vx, vy, vyaw_flag, target_yaw)`, where the
fourth element is an **absolute world-frame heading**, integrated upstream, and the
third is a raw turning flag rather than a yaw rate
(`decoupled_wbc/control/policy/g1_gear_wbc_policy.py`). The registered `full_hold`
held `target_yaw` at the *waist joint* yaw, which is a different frame. It is now
held at the measured base yaw captured after stabilization.

### R2.3 — Matched configs are genuinely matched

`Task.reset` reassigns `articulate_init_joint_qpos` from an unseeded
`random.uniform` on every reset, which looked like it would desynchronise the
conditions. It does not: the eval path passes `options["state_dict"]` and
`DRManager.load_state_dict` is called with `dr_level=None`, restoring every
randomizer. Two resets on the same eval episode produced bit-identical `qpos`
across all 80 DoF and an identical `ngeom`. The runner additionally seeds
`random` / `numpy` per config, which costs nothing.

### R2.4 — The killer: CloseDoor's canonical solution has no right-arm motor program

A canonical rollout was instrumented for per-step MuJoCo contact attribution
between robot bodies and the door subtree. Under the official `mujoco_isaac`
sim mode:

```text
right_shoulder_pitch  range 0.053 rad
right_shoulder_roll   range 0.046 rad
right_elbow           range 0.077 rad
right_wrist_yaw       range 0.309 rad
left arm              ranges <= 0.24 rad
base xy travel        0.734 m
only contact part ever touching the door: right_hand
contact ends at step 258; the door coasts from -0.062 to -0.166 unaided
official success at step 269
```

Psi0 solves CloseDoor by **walking into the door with the hand that already hangs
at its side**. The shoulder and elbow move less than 5 degrees. The right arm is
not executing a motor program; it is a passive bumper transported by locomotion.

The same probe under `sim_mode=mujoco` gives the same picture (wrist ranges
`0.231 / 0.126`, base travel `0.720 m`, `right_hand` the only contact), so this is
not an artifact of degraded rendering.

Consequences for the registered design:

1. `right_disabled` as originally specified (hold the arm where it is) removes
   nothing, because the arm was not moving. Measured: `canonical`,
   `right_frozen` and `right_disabled` all succeed, all with `right_hand` as the
   door contact.
2. Retracting the arm to the neutral at-side pose does not help either — that pose
   *is* where the arm already is.
3. `both_arms_disabled` also succeeds, with `right_hand` still the contact.

So the registered "cleanest event" —

```text
canonical succeeds / oracle succeeds / right_disabled succeeds / full_hold fails
```

— would have been observed on CloseDoor **for reasons that have nothing to do with
motor equivalence**. The original panel could not have detected this.

### R2.5 — New conditions and gates

Two conditions were added to separate the three worlds a `right_disabled` success
is consistent with:

- `right_frozen` — locked-joint fault; tests whether the arm's *articulation*
  matters at all;
- `both_arms_disabled` — tests whether any arm matters at all.

Plus `left_disabled` as a laterality control. The gate order is in
[README.md](README.md#frozen-g0-gates). Gate 2
(`canonical - right_frozen >= 0.20`) is behavioural, so it needs no tuned cut on
joint excursion.

### R2.6 — Dead code removed

`intervene_absolute_action` and `apply_motor_condition` implemented the pre-WBC
action-group intervention and are no longer on any execution path. They were
deleted rather than left in place, so nothing in the repository looks like a wired
intervention that is not.

### R2.7 — Sample plan

Published SIMPLE Table 7 reports Psi0 at `10/10/10` on CloseDoor across DR levels
0/1/2, and each level ships 10 eval episodes. The frozen panel is therefore all
three levels, 30 matched configs per task, decided before any panel outcome was
seen. Config ids are namespaced by level so they cannot collide.

The frozen task panel was two tasks from the start (`close_door`, `open_faucet`),
so running OpenFaucet after CloseDoor fails a prerequisite is completing the
preregistered panel, not shopping for a better task.

### R2.8 — `decompose()` is not evidence about the *Teleop* demonstrations

The registration argued that the benchmark separates task effect from motor
realization because both tasks' `Task.decompose()` use
`hand_uid="dex3_right", lock_links=["left_hand_palm_link"]`.

That argument does not hold for these two tasks. `decompose()` drives the CuRobo
motion-planning datagen path used by the `*MP` tasks. `G1WholebodyCloseDoorTeleop`
and `G1WholebodyOpenFaucetTeleop` are `*Teleop` tasks: their data was human
teleoperated (`pico_decoupled_agent`), so `decompose()` describes a code path that
did not generate the demonstrations.

Nor can the claim be checked against the shipped eval data. The
`simple-eval/*.zip` LeRobot datasets contain **one frame per episode** (10 frames
for 10 episodes, action dim 43) — they are reset configurations, not trajectories.
Measuring demonstrator laterality would require the full training split.

This does not damage the experiment, because the panel already measures the
relevant quantity in outcome space and on the policy itself:

```text
right_frozen fails  AND  left_disabled succeeds
```

is a direct behavioural demonstration that the policy's solution depends on the
right arm specifically rather than on having *an* arm. That contrast replaces the
`decompose()` argument as the laterality evidence, and it is stronger, because it
is about the object under study rather than about how the data was collected.

### R2.9 — `min_dist_*_palm_m` is measured to the articulated root body

The palm-distance diagnostics are distances to the origin of the body that owns
the effect joint, which for CloseDoor is the hinge root rather than the panel
surface (observed canonical values cluster near `1.18 m` while the hand is in
contact). Use it as a within-task relative signal — did this condition approach
the object as closely as canonical did — not as an absolute clearance.
