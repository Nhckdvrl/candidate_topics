# Topic 08 audit — 2026-08-22

Auditor pass over the pre-existing planar-arm G0 prototype (`src/planar_arm.py`,
`src/geometry.py`, `src/evaluate_g0.py`, `src/analyze_g0.py`) plus a re-check of the
baseline literature. The prototype had **never been run** (no `results/` on `main`), so
this is a design audit, not a results audit.

Verdict: **the planar-arm G0 cannot decide this topic.** Its primary contrast is
circular by construction. It is retained as an illustrative toy but demoted out of the
decision path; the decision is moved to PushT + a real generative policy (see
`README.md`, `PUSHT_EXISTENCE_TEST.md`).

---

## A1 (blocking). The G0 "functional risk" outcome is a linear function of the same
projection used to define task-sensitive variance.

`evaluate_g0.py:state_metrics` defines risk from

```text
progress = (d0 - ||fk(q_T) - target||) / d0 ,   q_T = q + dt * sum_h a_h
```

with `dt = 0.08` and 4 executed steps, and defines `task_var` as
`tr(P_task Sigma P_task)` with `P_task` the row-space projector of `J(q)` — the *same*
`J(q)`, frozen at the *same* `q`.

To first order `fk(q + dq) - fk(q) = J(q) dq`, so `progress` is an affine function of
`P_task dq`. Variance of the executed action along `row(J)` therefore *mechanically*
produces variance in `progress`, and — with a one-sided threshold at 15% — mechanically
produces a higher fraction of below-threshold samples. Variance along `null(J)` is
first-order invisible to `progress` by definition.

So gate `G3` ("matched ACE, higher task-sensitive variance ⇒ higher risk") is close to a
tautology of the linearization, not an empirical claim about a policy. It would fire for
*any* action distribution, including pure isotropic Gaussian noise, and tells us nothing
about whether a learned generative policy actually allocates its diversity this way.

The fix is not a better threshold. The outcome has to be produced by a dynamics model
that is **not** the first-order map used to build the projectors. In PushT the outcome is
the T-block pose after contact-rich pymunk physics; the action is the pusher's target
position. There is no linear identity connecting them, so the same measurement becomes a
real test.

## A2 (blocking for the stated claim). The conditional multimodality only exists at t=0.

`planar_arm.generate_dataset` repeats each base task `(q0, target)` with several hidden
posture preferences `q_pref`, which is the right idea — at `t = 0` the observation
`[q0, target]` is identical across modes while the demonstrated action differs.

But `build_windows` then emits a training window at *every* timestep `t`, with observation
`[q_t, target]`. For `t >= 1` the trajectories have already separated in joint space, so
`q_t` identifies the mode almost perfectly and `p(a | s)` is effectively unimodal. With a
mean episode length of ~30 steps, roughly 3% of the training windows carry the designed
multimodality; ~97% actively teach the policy that the posture preference is *inferable*.

The README explicitly names this failure mode ("the current joint configuration may reveal
which posture mode generated the trajectory") but the code does not defend against it.

## A3. The ACE implementation does not match released FIPER code.

Checked against the official repository (`utiasDSL/fiper`,
`evaluation/method_eval_classes/entropy_eval.py`, `configs/eval/entropy.yaml`,
`configs/task/push_t.yaml`), not against the paper text.

| | released FIPER | `src/geometry.py` |
|---|---|---|
| cell width | `cellsize_factor * R_d`, **`cellsize_factor = 0.03`** (`entropy.yaml`; base default `0.01`) | `alpha * R_d`, `alpha = 0.1` |
| horizon aggregation | **mean** over prediction steps | **sum** over prediction steps |
| grid origin | per-state `min - 1% * (max-min)`, `np.digitize` | per-state `min`, `floor` |
| dims scored | the `position` action mapping only (3 dims for push_t) | all action dims |
| calibration | ranges over a held-out *calibration rollout set* | ranges over the training chunks |

Sum-vs-mean is a fixed horizon-length rescale and does not change rankings, so it is
cosmetic. `alpha = 0.1` vs `0.03` is not cosmetic: it is a 3.3x coarser cell and directly
changes how many distinct cells a given sample spread occupies. Using the wrong constant
against a named baseline is exactly the strawman risk to avoid, so the new code uses
`0.03` and reproduces the released grid construction (dynamic limits + 1% buffer +
`digitize`) rather than a paraphrase.

Also worth recording, because it bears on our novelty claim: FIPER's ACE is computed on
the **Cartesian `position` component of the predicted action chunk**, not on raw joint
commands. FIPER already maps actions into a task-relevant space before taking entropy.
Any version of this topic that only says "compute entropy in end-effector coordinates
instead of joint coordinates" is already covered by the baseline.

## A4. Bootstrap treats states as i.i.d.

`analyze_g0.py:bootstrap` resamples rows of the state table. For the planar-arm G0 the
evaluation states were independently sampled, so this was defensible there. It is not
defensible for the PushT design, where probe states come from the same rollouts and are
strongly dependent within a rollout. The new analysis bootstraps **rollouts**, not states.

## A5. `G1` (`median(NullPerDim / TaskPerDim) >= 0.75`) is not evidence for the hypothesis.

For an isotropic action distribution the per-dimension normalization makes this ratio
exactly 1.0. The gate is therefore passed most easily by a policy with *no* task
structure at all. It cannot distinguish "learned goal-equivalent diversity" from "the
sampler is noisy". The no-null control was supposed to catch this but is only a
diagnostic, not a gate.

## A6 (minor). `_joint_hist_entropy` anchors the grid to the sample minimum.

This turns out to match released FIPER (which also uses per-state dynamic limits), so it
is *not* a bug relative to the baseline. Noting it only because the docstring calls the
ranges "fixed calibration ranges", which describes the cell *width*, not the grid origin.
Kept, with the comment corrected.

---

## What survives

The scientific question is unchanged and still worth one clean shot:

> Does scalar action entropy conflate task-sensitive uncertainty with goal-equivalent
> (task-null) diversity?

What does not survive is the plan to answer it with an analytic arm whose outcome
function is the linearization of its own measurement. The decision experiment is moved to
a setting where the outcome is produced by a simulator that knows nothing about our
projectors.
