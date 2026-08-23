# Topic 19 Archive Summary

## Final decision

**ARCHIVED / PRIMARY METRIC IDENTIFICATION FAILURE**

Topic 19 asked whether a robot foundation policy selectively corrects task-threatening body deviations while tolerating task-equivalent redundant deviations, in the spirit of the minimal-intervention principle.

The experiment itself ran cleanly enough to produce a tight frozen scalar null on the observed configs:

- `R_task = 0.9632`
- `R_null = 0.9671`
- `DeltaR = -0.0038`
- 8-config bootstrap 95% CI `[-0.0275, +0.0178]`

That numerical result lies deep inside the preregistered KILL region. However, post-run diagnostics showed that the scalar was not an identifying measurement of the scientific construct.

## Why the numerical KILL is not a hypothesis KILL

The frozen score projected the change in Psi0's absolute right-arm target onto the exact injected joint-space perturbation direction:

```text
A(d) = <Delta a, d> / ||d||^2
R(d) = 1 - A(d)
```

This only behaves like a correction/accommodation fraction if the policy response is approximately aligned with the injected joint-space axis.

The actual response violated that assumption strongly:

- task branch: only about `15.3%` of the target response was aligned with `d`;
- null branch: only about `36.0%` was aligned with `d`.

Most of the response lived in orthogonal joint-space directions.

That matters because a redundant arm can correct the same end-effector error through a different joint coordination. It is entirely possible to have

```text
Delta a perpendicular to d
J Delta a = -J d
```

which is a task-space correction, while the frozen scalar still returns `A=0, R=1`. A different orthogonal response could instead preserve the task-space displacement and receive the same scalar value.

Thus the same primary score can correspond to opposite task-space behavior.

The clean conclusion is therefore:

> Psi0 did not differentially restore the exact injected joint-space axes under this test.

The experiment cannot support the stronger claim:

> Psi0 lacks task-structured feedback or a minimal-intervention-like response.

## Additional construct problem

The `task` perturbation was defined as the wrist-position Jacobian direction that maximizes wrist translation for a fixed joint norm. This is kinematically clean, but `wrist-moving` is not synonymous with `CloseDoor-task-relevant`.

Task relevance in this environment depends on hand-door relative geometry, contact, hinge direction, phase, and realized outcome. So even before changing the response metric, a future study should define both perturbation relevance and correction directly in task/outcome space.

## What worked well

Several important prerequisites were strong:

- official-path P0 competence: `10/10`;
- common-random-number paired inference: exact same-seed repeat (`diff=0.0`), while a different seed changed the action;
- finite geometry: `48/48` selected states passed;
- typical task wrist displacement about `31 mm` versus null displacement about `0.29 mm`, with about `0.09 deg` null rotation;
- no epsilon search, time-point search, joint search, latent probing, or G0b rescue was used after the negative result.

The experiment therefore taught a useful identification lesson rather than merely failing from weak implementation.

## Sample limitation

The final G0a set had 16 successful rollouts from 8 distinct level-0 configs. Configs `1` and `7` failed both collector attempts despite P0 reaching 10/10 under the official CLI. This indicates a residual collector-vs-official-eval difference and means the pre-run amendment's intended 10-config bootstrap was not fully realized.

The 8-config CI is preserved because it is the result actually obtained, but the systematic missingness is another reason not to promote the frozen numerical KILL into a broad scientific falsification.

## Reusable lesson

> **For redundant control systems, projecting a policy's action response back onto the exact injected joint-space perturbation is not a generic measure of task-space correction. A policy can correct through a different joint coordination. Define the dependent variable in the same task/outcome space as the scientific claim.**

A second reusable rule follows:

> **`end-effector-changing` is not automatically `task-relevant`. For manipulation, task relevance should be tied to contact geometry, object state, or realized outcome rather than generic Cartesian displacement.**

## Stop rule

Do not repair Topic 19 by swapping in a task-space metric after seeing these data. That would be a post-hoc redesign of the primary measurement. If this question is revisited, it should be registered as a new topic with task/outcome-space perturbation and correction definitions frozen before data collection.

See [`G0_RESULTS.md`](./G0_RESULTS.md) for the complete run result and deviations.
