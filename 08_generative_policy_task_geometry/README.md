# 08 — Generative Policy Diversity Has Task Geometry

## Question

Generative robot policies can produce many different action trajectories from the same observation. When that distribution is broad, does it mean the policy is uncertain about what to do, or can the diversity mostly lie in directions that do not change the task outcome?

The concrete hypothesis is:

> **Two policy states can have similar scalar action entropy but different functional risk because their variability points in different task-relative directions.**

This is motivated by the uncontrolled-manifold / motor-redundancy literature, but the object measured here is the sampled conditional action distribution of a generative policy.

## Why this folder starts with an analytic robot

The first experiment intentionally avoids images, perception, and a large simulator. We need to know whether the proposed phenomenon exists before paying for a full Franka/ManiSkill replication.

The G0 robot is a 4-DoF planar arm with a 2-D end-effector position task. Its Jacobian gives an exact local decomposition of joint velocity into:

- task-sensitive row-space directions;
- task-null directions that leave end-effector position unchanged to first order.

The policy is a conditional DDPM over 8-step action chunks.

## Crucial data design

A naive dataset would give every episode a different initial state and target. That is not enough: the current joint configuration may reveal which posture mode generated the trajectory, so the conditional distribution `p(a | s)` need not actually be multimodal.

Instead, each base task `(q0, target)` is repeated with several different hidden posture preferences. At the identical initial observation, all demonstrations pursue the same task outcome but have different projected null-space components. The preference itself is **not** observed by the policy.

A second policy is trained on the same repeated tasks with `null_gain=0`. This negative control tests whether apparent null-space diversity is merely isotropic DDPM sampling noise.

## Measurements

For action covariance `Sigma` and task Jacobian `J`, the code computes orthogonal action-space projectors from an SVD:

```text
P_task = projector onto row(J)
P_null = projector onto null(J)
```

and records both total and dimension-normalized variance:

```text
V_task = tr(P_task Sigma P_task) / rank(P_task)
V_null = tr(P_null Sigma P_null) / rank(P_null)
```

The dimension normalization is mandatory: otherwise a larger null space mechanically has more total variance.

Scalar action uncertainty is measured with the FIPER-style Action-Chunk Entropy (ACE) estimator: fixed calibration ranges, dimension-wise cell widths, joint D-dimensional occupied cells, entropy per prediction step, summed over the chunk.

Functional risk is measured by cloning the same state, executing each sampled chunk open-loop for a short horizon, and asking whether it makes at least 15% relative progress toward the task target.

The key contrast is then **matched ACE, different task geometry**. High-task-fraction states are matched to low-task-fraction states with nearly equal standardized ACE, and their empirical execution risk is compared.

## G0 decision

G0 is an existence screen, not the paper experiment. It asks four things:

1. can the learned generative policy actually solve the task (`rollout_success >= 0.80`)?
2. does its sampled diversity contain substantial task-null structure after per-dimension normalization?
3. can we find enough high-vs-low task-fraction state pairs at nearly matched ACE?
4. inside those matched pairs, is higher task-sensitive variability associated with materially higher execution risk?

If these fail, there is no reason to build a larger robot experiment around the story.

If they pass, the next experiment must move beyond the easy objection “just map joints to Cartesian space”: use a Cartesian 6-D end-effector policy on a task that constrains only a lower-dimensional functional variable (for example position but not orientation), and test the same decomposition relative to the **task**, not merely the robot kinematics.

## Run

```bash
pip install -r requirements.txt
GPU=0 SEED=0 ./run_g0.sh
```

For parallel seeds on separate GPUs/nodes:

```bash
GPU=0 SEED=0 ./run_g0.sh
GPU=1 SEED=1 ./run_g0.sh
GPU=2 SEED=2 ./run_g0.sh
```

See `SERVER_HANDOFF.md` for the exact server workflow and `VALIDATION.md` for the frozen measurements.

## Layout

```text
src/geometry.py       exact UCM/task-null decomposition + FIPER ACE
src/planar_arm.py     redundant robot, resolved-rate expert, repeated-task dataset
src/diffusion.py      low-dimensional conditional DDPM
src/train_g0.py       training + checkpoints
src/evaluate_g0.py    sampled action geometry, ACE, rollout success, empirical risk
src/analyze_g0.py     matched-entropy analysis, bootstrap, G0 decision
run_g0.sh             main condition + no-null negative control
tests/                numerical and analysis unit tests
```

## What a positive G0 does *not* prove

A positive planar-arm result does not establish that current VLA uncertainty methods fail in realistic manipulation. It establishes only that a learned generative robot policy can encode large goal-equivalent diversity which scalar action entropy collapses together with task-sensitive variability. The realistic Cartesian/task-manifold replication is required before making the broader robotics claim.
