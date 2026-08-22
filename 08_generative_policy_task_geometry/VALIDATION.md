> **SUPERSEDED / ARCHIVED.** This is the validation design for the planar-arm G0
> prototype, which was killed before it ran: its primary contrast was an algebraic
> identity satisfied by any action distribution. See [`AUDIT.md`](./AUDIT.md) finding A1
> and [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md). Kept unedited for the record.

# Validation design

## Primary scientific object

For a fixed observation `s`, sample `B` action chunks from the same conditional generative policy:

```text
A^(1), ..., A^(B) ~ pi(A | s)
```

The primary question is not whether the distribution is broad. It is whether the **directions of variability relative to the task** contain information that a scalar entropy discards.

## G0 system

Robot: 4-link planar arm, 4 joint-velocity action dimensions.

Task variable: 2-D end-effector position.

At current configuration `q`:

```text
J(q) = d x_ee / d q
rank(J) = 2 in regular configurations
```

Thus the action space has two local task-sensitive dimensions and two task-null dimensions.

### Expert

The demonstration controller is:

```text
qdot = J_DLS^+ K(x* - x) + lambda_N P_null (q_pref - q)
```

where `q_pref` is a latent posture preference. All posture preferences target the same Cartesian goal.

### Conditional multimodality requirement

Each sampled `(q0, target)` base task is repeated `modes_per_task` times using different `q_pref` values. `q_pref` is hidden from the learner. Therefore the exact same initial observation has multiple correct first actions whose differences are designed to lie primarily in the task-null subspace.

This is necessary. Merely collecting different trajectories from different states would not establish a multimodal `p(a|s)`.

### Negative control

Repeat the entire dataset/training pipeline with:

```text
null_gain = 0
```

The repeated observations remain, but the expert no longer injects a posture-dependent null component. If the main and control policies show indistinguishable null-space diversity, DDPM sampling noise/model error is a more plausible explanation than learned functional redundancy.

## Policy

The existence pilot uses a small conditional DDPM rather than the full image Diffusion Policy architecture.

Input:

```text
[q1, q2, q3, q4, target_x, target_y]
```

Output:

```text
8 x 4 joint-velocity action chunk
```

Training predicts diffusion noise. Default diffusion steps: 50. Default training steps: 30,000.

This is sufficient for G0 because the scientific measurement is on the sampled conditional action distribution, not on a visual representation.

## Action-geometry measurement

For each evaluation state, sample `B=128` chunks.

For each predicted step, estimate action covariance `Sigma_h`. Use SVD of `J(q)` to define exact orthogonal projectors:

```text
P_task = V_task V_task^T
P_null = V_null V_null^T
```

Report:

```text
TaskTotal_h = tr(P_task Sigma_h P_task)
NullTotal_h = tr(P_null Sigma_h P_null)

TaskPerDim_h = TaskTotal_h / rank(P_task)
NullPerDim_h = NullTotal_h / rank(P_null)
```

The chunk scores sum over prediction steps. The main null/task ratio uses the **per-dimension** values.

The local Jacobian is frozen at the current state across the short predicted chunk. This makes the measurement explicitly a local first-order UCM test; it does not silently mix task geometry changes along the rollout.

## Scalar entropy measurement

`src/geometry.py` implements the FIPER Action-Chunk Entropy construction:

1. compute each action dimension's calibration range `R_d` over training chunks;
2. set fixed cell width `alpha R_d` (`alpha=0.1` by default);
3. for each predicted timestep, bin the `B` D-dimensional sampled actions into joint cells;
4. compute occupied-cell entropy in bits;
5. sum over the action-chunk horizon.

This matters because a Gaussian log-det entropy would itself impose a unimodal approximation and would not reproduce the type of scalar entropy currently used for generative policy failure prediction.

## Functional risk

For each sampled chunk:

1. clone the same robot state;
2. execute the first four actions open-loop;
3. measure relative end-effector progress toward the fixed target.

A sample is labeled bad when:

```text
(initial_distance - final_distance) / initial_distance < 0.15
```

State risk is the fraction of bad sampled chunks.

The outcome label therefore comes from task execution, not from action magnitude, Jacobian projection, or entropy.

## Evaluation-state distribution

Fresh reachable problems are generated with a held-out seed. States mix:

```text
sigma = 0.00  ID
sigma = 0.15  mild joint perturbation
sigma = 0.30  stronger perturbation
```

The perturbation changes the robot state but not the task target. Its purpose is to create a useful range of policy competence/risk without manufacturing a new reward function or failure definition.

Results must also be reported separately by perturbation tag so a pooled effect cannot hide a pure OOD-level confound.

## Primary matched-entropy contrast

Scalar ACE is z-scored across evaluation states.

Define:

```text
low geometry  = bottom quartile of task_fraction
high geometry = top quartile of task_fraction
```

Greedily pair high and low states subject to:

```text
abs(z_ACE_high - z_ACE_low) <= 0.15
```

with no state reused.

The primary effect is:

```text
DeltaRisk_matched = mean(risk_high_task_fraction - risk_low_task_fraction)
```

Bootstrap evaluation states 2,000 times.

## G0 screen

The automated analysis uses the following deliberately small set of conditions:

```text
G0: rollout_success >= 0.80
G1: median(NullPerDim / TaskPerDim) >= 0.75
G2: >= 30 matched-ACE high/low geometry pairs
G3: matched DeltaRisk >= 0.10 and bootstrap CI95 lower bound > 0
```

`G1` is intentionally permissive: the point is not to require null variance to dominate every state, only to reject a policy whose diversity is almost entirely task-sensitive.

The no-null condition is interpreted as a diagnostic rather than another hard gate. A compelling main result should show substantially more structured null variability than the no-null policy.

## Secondary measurements

Reported but not used to rescue a failed primary contrast:

```text
Spearman(ACE, risk)
Spearman(TaskPerDim, risk)
Spearman(NullPerDim, risk)
AUC for high-risk state ranking
raw total action variance
task_fraction
tag-stratified effects
```

Training checkpoints are saved at 1k, 3k, 10k, 30k steps so the proposed acquisition trajectory can be examined after the final-policy existence result is known:

```text
success up
TaskPerDim down
NullPerDim stays high / falls more slowly
```

This trajectory is secondary. The topic should not be saved by a pretty checkpoint curve if the final-policy matched-entropy contrast is absent.

## Interpretation

### Strong G0

- policy solves the task;
- repeated identical tasks lead the generative policy to retain substantial goal-equivalent diversity;
- scalar ACE can be nearly identical for states with different task-sensitive fractions;
- the high-task-sensitive member of those pairs has higher empirical execution risk;
- the no-null control shows much weaker/null structure.

Proceed to a Cartesian/task-manifold robot experiment.

### Stop

Stop or redesign the premise if any of the following dominates:

- final policy cannot solve the simple task;
- sampled diversity is effectively isotropic DDPM noise and looks the same in the no-null control;
- task-null variability collapses during learning;
- matched ACE states do not differ in functional risk;
- any apparent effect exists only after mixing perturbation levels and disappears within each level.

## Required next experiment after G0

The planar arm only addresses kinematic redundancy. A broader paper claim requires a task-level null space in Cartesian action coordinates.

The clean next design is a 6-D Cartesian end-effector policy on a position-only task:

```text
a = [dx, dy, dz, dRx, dRy, dRz]
task = end-effector position only
```

Translation is task-sensitive and orientation is goal-equivalent by task definition. If the same matched-entropy/risk result survives there, the explanation cannot be dismissed as merely “compute entropy in Cartesian space.”
