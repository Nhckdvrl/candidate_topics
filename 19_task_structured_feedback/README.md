# 19 — Do Robot Foundation Policies Learn Task-Structured Feedback?

**Status: CANDIDATE / FROZEN G0 READY**

## The natural question

A robot can reach the same task goal with many different body configurations. Classical optimal feedback control predicts a **minimal intervention principle**: correct deviations that threaten the task, but do not waste control effort restoring deviations that live in redundant/task-null dimensions.

That gives a simple question for modern robot foundation policies:

> **When the robot body is perturbed, does a foundation policy correct the perturbation according to task geometry, or does it simply pull the body back toward a familiar demonstrated configuration?**

The first experiment does not inspect hidden states, train probes, or add a learned metric. It changes the physical robot state in two matched ways and asks the frozen policy what absolute target it now commands.

## Why this is worth testing

The minimal-intervention principle is a classic property of competent feedback control, not an LLM/VLA-specific construct. A generalist VLA trained mainly from demonstrations is not explicitly told that many body configurations are equivalent for the task. If task-structured feedback emerges anyway, that is a meaningful control property of foundation policies. If it does not, the failure exposes a direct training target: teach policies invariance/accommodation along task-equivalent body directions while preserving correction in task-sensitive directions.

The method opening is therefore concrete: **task-structured feedback regularization / body-redundancy augmentation**, not just another diagnostic plot.

## Exact G0 platform

- **Policy:** released Ψ₀.
- **Simulator/evaluation:** SIMPLE.
- **Task:** `G1WholebodyCloseDoorTeleop-v0`.
- **Robot:** G1 Sonic, seven-DoF right arm.
- **End effector:** `right_hand_palm_link`.

This is not a competence gamble. SIMPLE reports Ψ₀ at **10/10, 10/10, 10/10** on CloseDoor across its three reported DR levels. The task source defines success from the door joint state, while the demonstration decomposition explicitly uses the right hand. Thus task completion and a particular whole-body realization are not the same variable.

Upstream contracts frozen during design audit:

- SIMPLE commit `b49c1aea2dd57309bb533219d0d34d6020f3d943`
- Ψ₀ commit `9ad917526394c1cacc72dba08562629936505987`

Relevant source:

- SIMPLE CloseDoor task: `src/simple/tasks/g1_wholebody_close_door_teleop.py`
- G1 Sonic joint/EE definitions: `src/simple/robots/g1_sonic.py`
- Ψ₀ SIMPLE agent: `src/simple/baselines/psi0_decoupled_wbc.py`
- Ψ₀ dataset/action layout: `scripts/postprocess_psi0.py`
- Ψ₀ server: `src/psi/deploy/psi0_serve_simple.py`

## The one clean contrast

At one real on-policy state with right-arm joint state `q`, compute the geometric Jacobian of `right_hand_palm_link` restricted to the seven right-arm joints.

Construct equal joint-norm perturbations:

### 1. Task-space perturbation

`δ_task` is the top right singular vector of the **3×7 wrist-position Jacobian**, scaled to joint-space norm `ε=0.08 rad`.

It is the parameter-free local direction that moves the wrist position most for that joint-space perturbation budget.

### 2. Full-pose null perturbation

`δ_null` is the one-dimensional null direction of the **6×7 wrist geometric Jacobian**, with the same joint-space norm.

To first order it changes the redundant body configuration while preserving both wrist position and orientation.

We do not trust the linearization blindly. After applying each perturbation to the real MuJoCo state and calling `mj_forward`, the state is accepted only if the finite-FK checks pass:

- task wrist translation ≥ 5 mm;
- null wrist translation ≤ 2 mm;
- null wrist rotation ≤ 1 degree;
- task/null translation ratio ≥ 5;
- joint limits and simulator validity must hold.

If too few ordinary on-policy states satisfy this fixed construction, the platform fails; we do not tune `ε`, pick a convenient joint, or search perturbation directions.

## Crucial action-semantics correction

Ψ₀ does **not** output a right-arm delta. Its SIMPLE action dimensions `21:28` are **absolute right-arm joint targets** that are passed to the downstream WBC.

Therefore raw `a(q+δ)-a(q)` is not itself a correction signal.

Let `a0 = a(q)` and `aδ = a(q+δ)` be the first returned right-arm absolute targets. Define the **accommodation fraction**

`A(δ) = <aδ-a0, δ> / ||δ||²`

and the corresponding **implied correction fraction**

`R(δ) = 1 - A(δ)`.

Interpretation:

- `A≈0, R≈1`: Ψ₀ leaves the absolute target near the original target; WBC will tend to erase the perturbation.
- `A≈1, R≈0`: Ψ₀ moves the target one-for-one with the perturbation; WBC will tend to preserve/accommodate it.

The primary paired statistic is

`ΔR = R_task - R_null = A_null - A_task`.

**Task-structured feedback predicts `ΔR > 0`: the policy asks for more correction of wrist-changing deviations than of wrist-pose-preserving redundant deviations.**

This statistic is not guaranteed by how the Jacobian constructs the perturbations. A policy whose absolute target ignores both perturbations gets `ΔR≈0`; a policy that accommodates both equally also gets `ΔR≈0`.

## Physical observation intervention, not proprio spoofing

For every branch:

1. restore the exact same MuJoCo snapshot;
2. physically change only the seven right-arm qpos values;
3. preserve qvel, controller state, time, and every other simulator field from the same snapshot;
4. call `mj_forward` without integrating time;
5. re-render the observation and rebuild proprio from that physical state;
6. query Ψ₀ while preserving the same deployed previous-height/context values across branches.

The camera image is **not** held fixed while proprio changes. The intervention is a consistent physical robot state, and the only intended state difference is right-arm configuration.

No branch rollout is needed for the first gate. We measure the next high-level target before downstream WBC can create its own recovery behavior.

## Stochastic inference and RTC

Official SIMPLE deployment runs Ψ₀ with RTC and `action_exec_horizon=24`. The server is stateful, so naively issuing three HTTP requests would contaminate the paired comparison through `previous_action` and action history.

The frozen G0 therefore has two stages:

### G0a — clean policy-map test

At a saved deployed on-policy state, query the **reset-mode ordinary `predict_action` mapping** for `base / task / null`. Before each of the three model calls, run Ψ₀'s own `seed_everything(pair_seed)` with the same pair seed. Use only the **first returned action target**.

This removes diffusion/flow sampling noise and RTC history as confounds while testing the learned mapping from the current physical observation to action target.

Before using G0a, an exact-repeat contract test must show that two identical state + identical pair-seed queries agree numerically. Failure is an engineering/identification stop.

### G0b — deployment confirmation, only after a positive G0a

Freeze the same `previous_action` / RTC history snapshot and perform the paired query through `predict_action_with_training_rtc_flow`, again with common random numbers and without mutating history between branches.

G0b is confirmation, not a rescue route. If G0a is weak, we stop.

## State sampling

First verify the exact released checkpoint succeeds on at least **8/10** level-0 CloseDoor evaluation episodes under the audited official stack. Otherwise this platform is not a valid object for the question.

For G0a:

- run **20 successful level-0 episodes**;
- use the **last three fresh-policy query states before success** from each episode;
- one episode remains the independent bootstrap unit;
- skip only states that fail the fixed kinematic validity checks above or violate joint limits/simulator validity;
- use four fixed common-random-number pair seeds per state: `20260823, 20260824, 20260825, 20260826`;
- average seeds/states within episode before inference.

This focuses the audit on actual door-manipulation decisions without choosing frames after seeing the response metric.

## Frozen G0 gate

Bootstrap the episode-level mean `ΔR` over episodes.

- **GO:** mean `ΔR >= 0.20` and 95% bootstrap CI lower bound `> 0`.
- **KILL:** 95% CI upper bound `<= 0.10`.
- **Otherwise:** `INCONCLUSIVE_DO_NOT_TUNE`.

A 0.20 effect means twenty percentage points more implied correction for task-space than null-space deviations. It is large enough to be a meaningful feedback-structure effect rather than microscopic sensitivity.

### What a positive G0 proves

At late successful CloseDoor decision states, the frozen foundation policy's immediate absolute target response is selectively structured by local end-effector task geometry: it accommodates wrist-pose-preserving body deviations more than wrist-moving deviations.

### What it does not prove

A positive result does **not** prove:

- optimal feedback control globally;
- that Ψ₀ represents an explicit Jacobian or task manifold internally;
- that every null perturbation is irrelevant over long horizons;
- that a negative result uniquely means "demonstration replay".

A negative result supports the narrower statement that this policy does not show a meaningful local minimal-intervention signature under the clean test.

## Why this avoids Topic 08's failure

Topic 08's original planar-arm gate became circular because the same Jacobian geometry helped define both the predictor and the measured "functional risk". Here the Jacobian is used **only to create two matched physical interventions**. The dependent variable is an independent frozen-model response: Ψ₀'s absolute target.

Nothing in the construction forces `A_null > A_task` or `R_task > R_null`.

## Collision audit

The closest classical line is Todorov & Jordan's minimal-intervention/optimal-feedback-control work: task-irrelevant deviations should be allowed to vary rather than actively corrected.

Recent adjacent VLA work found during the 2026-08-23 audit includes:

- **BYOVLA / Run-time Observation Interventions** — tests sensitivity to task-irrelevant **visual regions** and edits images at runtime (`arXiv:2410.01971`).
- **ProbeAct (2026)** — detects VLA failures and minimally filters actions with a CBF (`arXiv:2606.09740`).
- representation/action-steering work — intervenes on latent spatial features or action variables.
- classical robot redundancy/null-space control — explicitly engineers null-space behavior in controllers.

We did **not** find a recent paper that uses paired **same-state physical body perturbations, split into end-effector task-space vs Jacobian-null directions, to audit whether an off-the-shelf foundation VLA itself has learned a minimal-intervention feedback law**. This is a scoped collision statement, not a claim that no related paper exists anywhere.

## If G0 is positive: the paper-sized path

Do not immediately search hidden states. First establish external validity:

1. repeat the identical intervention on at least one additional foundation policy available in SIMPLE (e.g. π0.5 / GR00T-family if competent on the same task);
2. test another redundant manipulation task;
3. characterize whether training recipe/model family predicts the feedback signature.

Then the natural method contribution is to train **task-structured feedback consistency**:

- for task-null body augmentations, shift the target consistently with the null displacement rather than forcing the demonstration posture;
- for task-space perturbations, preserve corrective behavior.

The diagnostic gives a direct before/after mechanism metric, while task success under body perturbations gives the behavioral metric.

## Files

- `g0_core.py` — frozen perturbation construction, absolute-target response metric, episode bootstrap, verdict.
- `g0_simple_psi0.py` — audited SIMPLE/MuJoCo integration primitives and JSONL analyzer.
- `VALIDATION_AUDIT.md` — source, collision, identifiability, failure-mode audit.
- `tests/test_g0_core.py` — logic tests.

## Local engineering rule

Paths, launcher flags, device placement, rendering plumbing, and checkpoint locations may be adjusted to the actual machine. The scientific contrast may not:

**same physical state, equal-norm task/null right-arm perturbations, physically re-rendered observations, common-random-number Ψ₀ queries, first absolute right-arm target, frozen `ΔR` gate.**
