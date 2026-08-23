# 19 — Do Robot Foundation Policies Learn Task-Structured Feedback?

> **ARCHIVED / PRIMARY METRIC IDENTIFICATION FAILURE (2026-08-24).**
>
> Read [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) and [`G0_RESULTS.md`](./G0_RESULTS.md) first. The original scientific question remains interesting, but the frozen G0 scalar did not identify task-space correction once Psi0 responded through joint-space directions orthogonal to the injected perturbation.

## Final status

The frozen scalar on the observed configs was a very tight numerical null:

| quantity | result |
| --- | ---: |
| `R_task` | `0.9632` |
| `R_null` | `0.9671` |
| `DeltaR` | **`-0.0038`** |
| 8-config bootstrap 95% CI | **`[-0.0275,+0.0178]`** |

This lies inside the preregistered numerical KILL region. **It is not interpreted as a clean falsification of task-structured feedback.**

The reason is empirical, not philosophical: most of Psi0's action-target response was orthogonal to the injected joint-space direction. The frozen score only measured how much of the new target moved *along that exact direction*.

A redundant arm can correct an end-effector error with a different joint coordination. Therefore the same score can correspond to opposite task-space behaviors.

## Original natural question

A robot can reach the same task goal with many body configurations. Classical optimal feedback control motivates the minimal-intervention question:

> **When the robot body is perturbed, does a foundation policy selectively correct deviations that threaten the task while tolerating task-equivalent redundant variation?**

This remains a natural question. Topic 19 is archived because its first measurement did not identify that question cleanly enough, not because the question was shown false.

## Frozen G0 design

- **Policy:** released Psi0 `ckpt_40000`.
- **Simulator/evaluation:** SIMPLE.
- **Task:** `G1WholebodyCloseDoorTeleop-v0`.
- **Robot:** G1 Sonic, seven-DoF right arm.
- **Physical intervention:** equal joint-norm (`epsilon=0.08 rad`) right-arm perturbations from the same MuJoCo state.
- **Task branch:** top singular direction of the 3x7 wrist-position Jacobian.
- **Null branch:** one-dimensional null direction of the full 6x7 wrist geometric Jacobian.
- **Observation:** physically perturb qpos, preserve the rest of the snapshot, `mj_forward`, re-render and rebuild proprio.
- **Inference:** common-random-number base/task/null queries, using the first absolute right-arm target.

The source and identifiability audit is in [`VALIDATION_AUDIT.md`](./VALIDATION_AUDIT.md). The pre-run config-cluster amendment is in [`PROTOCOL_AMENDMENT_2026-08-23.md`](./PROTOCOL_AMENDMENT_2026-08-23.md).

## Prerequisites that passed

The experimental object itself was alive:

- official-path CloseDoor P0: **10/10**;
- paired stochastic inference: same state + same seed reproduced exactly (`diff=0.0`), while a different seed changed the action;
- finite geometry: **48/48** selected states passed;
- typical task wrist translation about **31 mm**;
- typical null wrist translation about **0.29 mm**;
- typical null wrist rotation about **0.09 deg**;
- task/null translation ratio about **109**.

Thus the stop is not explained by a dead policy, weak perturbation, stochastic noise, or failed kinematics.

## Frozen primary metric

Psi0 emits absolute right-arm joint targets, so the G0 defined

```text
A(d) = <a(q+d)-a(q), d> / ||d||^2
R(d) = 1 - A(d)
DeltaR = R_task - R_null
```

The intended interpretation was:

- `A≈1`: the target follows the body perturbation, so the deviation is accommodated;
- `A≈0`: the target stays near its original value, so downstream WBC tends to restore the body;
- task-structured feedback should give `DeltaR > 0`.

## Why that interpretation failed

The raw target response was not small:

| branch | `||Delta a||` | along `d` | orthogonal to `d` | alignment |
| --- | ---: | ---: | ---: | ---: |
| task | `0.0334 rad` | `0.0049` | `0.0328` | `15.3%` |
| null | `0.0198 rad` | `0.0069` | `0.0177` | `36.0%` |

Most of the response therefore lived outside the injected joint-space axis.

Let `J` be the end-effector Jacobian. It is possible to have

```text
Delta a perpendicular to d
J Delta a = -J d
```

which represents task-space correction through a different joint coordination while the frozen metric still gives `A=0, R=1`.

An orthogonal response with a different task-space consequence can receive the same scalar. So `R` is not a generic correction fraction once this response geometry appears.

The result that is actually supported is narrower:

> **On the observed CloseDoor states, Psi0 did not show differential restoration along the exact injected joint-space perturbation axes.**

It does not establish that Psi0 lacks a task-space minimal-intervention response.

## Second construct problem

The G0 used the wrist-position Jacobian's largest singular direction as the `task` perturbation. That guarantees a large wrist displacement, but CloseDoor task relevance depends on hand-door relative geometry, contact, hinge direction, and phase.

Therefore:

```text
end-effector-changing != automatically task-relevant
```

A fresh future formulation should define perturbation relevance and correction directly in task/contact/outcome space.

## Sample deviation

The final G0a sample contained:

- **16 successful rollouts**;
- **8 distinct level-0 configs**;
- **48 selected states**;
- four frozen pair seeds per state.

Configs `1` and `7` failed both collector attempts although official-path P0 was 10/10. The remaining eight configs succeeded in both repeats. This indicates a residual collector-vs-official-eval difference and systematic missingness.

The pre-run amendment had intended 10 config-level bootstrap units. The reported 8-config CI is preserved as the data actually obtained, but the intended 10-config primary analysis was not fully realized.

No rerun-until-success procedure was used to force the missing configs into the sample.

## Important implementation correction

`right_hand_palm_link` is authored through a fixed joint and is folded in the MuJoCo body representation. The collector therefore used `mj_jac` at `right_wrist_yaw_link` plus the authored palm offset `[0.0415,-0.003,0]`, with a runtime frame verification check.

This preserved the intended physical palm point rather than silently measuring a point about 4.15 cm away.

## No post-hoc rescue

After the frozen result, Topic 19 did **not**:

- tune epsilon;
- choose another joint subset or EE;
- select time points by response;
- replace the metric with a task-space/Jacobian-output metric;
- fit nonlinear response models;
- probe hidden states;
- run G0b to rescue G0a.

Those would be redesigns after observing the failure.

## Reusable lesson

> **For redundant control systems, joint-axis restoration is not the same thing as task-space correction. If the scientific claim is about task-space feedback, define the dependent variable in task/outcome space before collecting data.**

And:

> **A kinematically large end-effector perturbation is not automatically a task-relevant perturbation. Manipulation relevance should be grounded in contact/object/outcome geometry.**

If this scientific question is revisited, register it as a new topic with those definitions frozen from the start rather than repairing Topic 19.

## Files

- [`G0_RESULTS.md`](./G0_RESULTS.md) — complete result, diagnostics, and deviations.
- [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) — concise final interpretation and transferable lesson.
- [`PROTOCOL_AMENDMENT_2026-08-23.md`](./PROTOCOL_AMENDMENT_2026-08-23.md) — pre-score config-cluster amendment.
- [`VALIDATION_AUDIT.md`](./VALIDATION_AUDIT.md) — source/collision/identifiability audit before the run.
- `g0_core.py` — frozen original metric and perturbation logic.
- `g0_simple_psi0.py` — SIMPLE/Psi0 integration primitives.
- `tests/test_g0_core.py` — original frozen logic tests.
