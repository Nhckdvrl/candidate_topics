# Validation audit — Topic 19

Date: 2026-08-23

## 1. Natural construct audit

The scientific object predates modern robot learning: under the minimal intervention principle, feedback should correct deviations only insofar as they interfere with the task. Redundant body motion should be tolerated.

This gives a direct foundation-policy question without requiring latent variables, probes, or an invented benchmark score.

**Pass.**

## 2. Platform competence audit

SIMPLE reports Ψ₀ CloseDoor success of 10/10 at each of its three tabulated DR levels. The task source makes success depend on the physical door joint, not on similarity to a demonstration posture.

Local run still has a frozen prerequisite: exact checkpoint >=8/10 on level 0. Published numbers do not excuse a broken local inference stack.

**Pass in upstream; local prerequisite required.**

## 3. Task/body separability audit

`G1WholebodyCloseDoorTaskTeleop.check_wether_door_is_closed` reads `articulate_joint_1` and success accumulates only after the door is physically closed. The demonstration decomposition uses `dex3_right` and locks the left hand palm.

Therefore a right-arm redundant configuration is not itself the success criterion.

**Pass.**

## 4. Kinematic identifiability audit

G1 Sonic defines seven right-arm joints and `RIGHT_ARM_EE_LINK = right_hand_palm_link`.

At a generic configuration the restricted geometric Jacobian is 6x7. A rank-6 state therefore has a one-dimensional local null direction. G0 rejects rank-deficient states rather than choosing a convenient alternative direction.

The task direction uses the top singular direction of the *positional* Jacobian, avoiding arbitrary hand selection and guaranteeing a strong local wrist-translation contrast for the fixed joint-norm budget.

Finite FK is checked after the perturbation; first-order nullness alone is not accepted.

**Pass, conditional on fixed finite-geometry gate.**

## 5. Action-semantics audit — corrected during design

An earlier draft proposed scoring the sign of `a(q+d)-a(q)` as if Ψ₀ emitted delta actions. Source audit falsified that assumption.

SIMPLE's Ψ₀ format uses action indices 21:28 as absolute right-arm joint targets. `Psi0DecoupledWbcAgent` passes those targets as `target_upper_body_pose` to the WBC.

Hence the correct local quantity is the change in *tracking error*:

`[a(q+d)-(q+d)] - [a(q)-q] = [a(q+d)-a(q)] - d`.

Projecting onto `d` yields the accommodation/correction decomposition used in `g0_core.py`:

- `A(d)=<a_d-a_0,d>/||d||^2`
- `R(d)=1-A(d)`.

This correction is mandatory. The old raw-action sign metric must not be run.

**Pass after redesign.**

## 6. Downstream-controller confound audit

If we physically perturb the robot, let WBC execute, and then measure recovery, WBC itself can restore posture even if the VLA never encoded task-structured feedback.

G0 therefore stops before execution and reads the first high-level absolute target. WBC semantics enter only to interpret what that target would ask the controller to do.

**Pass.**

## 7. Observation-consistency audit

Changing only proprio while holding the camera frame fixed would create an impossible multimodal observation. G0 physically changes MuJoCo qpos, calls `mj_forward`, and re-renders the observation.

**Pass by contract.**

## 8. Stochastic sampling audit

Ψ₀ uses iterative action generation. Unpaired samples could easily masquerade as local sensitivity.

Psi0's own `seed_everything` resets Python, NumPy, Torch, and CUDA RNG state. G0 uses the same pair seed immediately before every base/task/null model call. Exact same-state/same-seed repeatability is a hard preflight assertion.

**Pass with common random numbers.**

## 9. RTC/history audit

Official SIMPLE serving enables RTC and keeps `previous_action`. Sequential branch requests would therefore contaminate each other.

G0a deliberately tests the reset-mode ordinary policy mapping. A positive signal is then confirmed in G0b by explicitly freezing the identical RTC previous-action/history state across branches. G0b cannot rescue a weak G0a.

**Pass with two-stage contract.**

## 10. Own-repo collision: Topic 08

Topic 08's original Jacobian design was killed because its key gate was an algebraic identity: geometry participated in both the construction and the outcome measure.

Topic 19 does not reuse that logic. Jacobian geometry only chooses the interventions. The outcome is the frozen VLA's absolute action target. For example:

- policy ignores both perturbations -> `DeltaR=0`;
- policy accommodates both perturbations equally -> `DeltaR=0`;
- policy restores both equally -> `DeltaR=0`;
- only selective task-vs-null response -> nonzero `DeltaR`.

**No internal collision / no tautology.**

## 11. Recent-literature collision audit

Searched combinations of:

- minimal intervention + VLA / robot foundation policy;
- Jacobian null space + VLA / robot policy;
- task-relevant vs task-irrelevant perturbation + VLA;
- physical/body/proprio perturbation + VLA robustness;
- failure recovery / CBF / runtime intervention in recent VLA work.

Closest families:

1. Todorov & Jordan (NIPS 2002; Nature Neuroscience 2002): establishes the minimal-intervention control principle itself.
2. Classical null-space / redundancy-resolution robot control: explicitly designs null-space behavior rather than asking whether a learned foundation policy acquired it.
3. BYOVLA (`arXiv:2410.01971`): task-irrelevant *visual-region* sensitivity and runtime image intervention.
4. ProbeAct (`arXiv:2606.09740`): failure detection plus minimal CBF action filtering.
5. VLA feature/action steering and robustness benchmarks: intervene on representation, visual nuisance, or failures; they do not provide the same body-space task/null paired feedback test.

No exact collision was found for the scoped audit question. This is not an exhaustive novelty guarantee; repeat the collision search before paper submission.

**Pass for candidate registration.**

## 12. Positive-result excitement test

A positive result says a demonstration-trained generalist foundation policy has acquired a classical task-selective feedback property: it distinguishes physical deviations that alter the active end-effector geometry from equally large body deviations that do not.

That is stronger than "the model is robust to noise" and produces a concrete comparative axis across model/training families.

A negative result is also actionable if tight: a strong policy can complete the nominal task yet fails to respect body redundancy, exposing an intervention/training target.

**Pass.**

## 13. Method-opening test

The obvious follow-up is not a probe. It is a task-structured consistency objective/data augmentation:

- augment training states along locally task-null body directions;
- make the action target co-move with those task-equivalent deviations;
- retain corrective targets for task-space deviations.

Evaluate nominal task success, robustness to body perturbations, and the same `DeltaR` mechanism score.

**Pass.**

## 14. Frozen kill discipline

Do not rescue a weak G0 by:

- searching perturbation magnitude;
- selecting elbow/shoulder directions after seeing results;
- changing EE link;
- choosing favorable time points;
- nonlinear response fitting;
- latent probing/SAE/PCA/CCA;
- switching primary metric from first action target to a convenient chunk statistic;
- using G0b RTC history to overturn a failed G0a.

If G0a is tightly near zero, archive the topic.
