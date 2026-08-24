# Where Does Closed-Loop Robustness Actually Live in Hierarchical Robot Foundation Policies?

> Status: **PROMOTED — registered as [Topic 24](../../24_hierarchical_feedback_attribution/) on 2026-08-24.**
>
> The registration condition written here was "do not register until the unperturbed replay-fidelity P0 passes". It passed, and one more gate was added before registration after P0 exposed a gap this document had not anticipated.
>
> ```text
> P0   replay fidelity     PASS   10/10 in all three conditions, 0.000 rad divergence
> P0b  WBC seam liveness   PASS   D = 2.4e-02..4.6e-02 rad over an exactly zero floor
> ```
>
> P0's perfect result was also its limit: with nothing driving the system off the recorded trajectory, a purely feedforward whole-body controller would have passed it identically, and then `delta_low` would have been structurally zero rather than informative. P0b closed that. It also found a hard limit on what `delta_low` can mean — below the VLA seam the arms and hands are open-loop interpolation, so that term can only carry locomotion/balance state feedback.
>
> See [`../prototypes/feedback_source_attribution/P0_RESULTS.md`](../prototypes/feedback_source_attribution/P0_RESULTS.md) and [`P0B_RESULTS.md`](../prototypes/feedback_source_attribution/P0B_RESULTS.md).

## Natural question

> **When a hierarchical robot foundation policy recovers from a physical disturbance, which layer is actually doing the recovery?**

A modern humanoid VLA is rarely the whole controller. A typical deployed stack is:

```text
vision + language + proprio
        -> VLA / high-level policy
        -> reference command
        -> whole-body / RL controller
        -> actuator reference
        -> robot
```

When the robot is pushed and still completes the task, the usual paper-level statement is simply "the VLA/system is robust". But that observed robustness is compatible with very different computations:

```text
H1  high-level VLA sees the disturbed state and replans
H2  VLA keeps issuing essentially the old reference, while the low-level controller recovers
H3  neither learned feedback layer is necessary; servo tracking, passive mechanics, or task slack absorb the perturbation
```

These explanations imply very different conclusions about what foundation-model pretraining has learned.

## Why this candidate exists now

This is not a speculative gap invented from literature adjacency. It comes directly from Topic 23's implementation audit.

In Psi0 + SIMPLE, editing the nominal VLA action **before** the whole-body controller did not necessarily remove the intended physical behavior, because the WBC observes the live robot state and re-solves the command. Topic 23 therefore had to move its clamp to the post-WBC actuator boundary.

That engineering discovery is itself a scientific tension:

> **hierarchical feedback can absorb an upstream intervention, so system-level robustness cannot automatically be attributed to the foundation policy.**

The new question is therefore one step closer to the actual system than Topics 19/23: instead of inferring a latent control law or motor-equivalence class, directly cut feedback at known software interfaces and ask which cut removes recovery.

## Why this avoids the previous failure modes

- **Topic 08:** no entropy/uncertainty proxy. Primary endpoint is official task success under a real physical perturbation.
- **Topic 09:** no rare natural checkpoint crossover. The contrast is created by exact replay at explicit interfaces that always exist in the deployed hierarchy.
- **Topic 15:** no decodability/mediation argument. Each feedback source is removed causally by replay.
- **Topic 19:** no joint-space projection is interpreted as task-space correction. The main outcome is task success / object state.
- **Topic 23:** the first gate is intervention-semantic fidelity before any perturbation outcome is inspected. If replay cannot faithfully instantiate the intended causal cut on the unperturbed task, the scientific test never starts.

## The causal experiment: a three-level feedback ladder

First record one successful, unperturbed canonical rollout from a fixed realized initial state. Record both interface tapes:

```text
Tape A: VLA -> WBC
  target_upper_body_pose
  navigate_cmd
  base_height_command

Tape B: WBC -> actuators
  target_q
  left_hand_q
  right_hand_q
```

Then start the same physical initial state and apply the exact same external disturbance under three conditions.

### 1. `fresh`

Everything is live:

```text
live observation -> live VLA -> live WBC -> actuator
```

This is total closed-loop system robustness.

### 2. `vla_replay`

Replay the recorded nominal high-level command tape, but let the WBC continue reading the **current perturbed robot state**:

```text
recorded VLA command -> live WBC(current proprio) -> actuator
```

This removes high-level VLA feedback/replanning while preserving low-level feedback.

### 3. `actuator_replay`

Replay the recorded post-WBC actuator-reference tape:

```text
recorded post-WBC targets -> joint servos -> robot
```

This removes both VLA feedback and WBC state-feedback adaptation. Remaining robustness is task tolerance, passive mechanics, and actuator-level servo behavior.

The causal decomposition is direct:

```text
high-level contribution = success(fresh) - success(vla_replay)
low-level contribution  = success(vla_replay) - success(actuator_replay)
residual robustness     = success(actuator_replay)
```

No latent manifold is required to interpret these quantities.

## P0 — replay fidelity before registration

This is deliberately **unperturbed** and must be completed before registering a root Topic.

Target: Psi0 `ckpt_40000` + SIMPLE `G1WholebodyCloseDoorTeleop-v0`.

Why this task first: Topic 23 already established canonical `30/30` success on exactly this stack, matching the released benchmark. We are not searching for a task with a convenient outcome.

For 10 fixed configs, run:

```text
fresh, force=0
vla_replay, force=0
actuator_replay, force=0
```

Frozen technical pass:

```text
fresh >= 0.90
vla_replay >= 0.90
actuator_replay >= 0.90
fresh - vla_replay <= 0.10
fresh - actuator_replay <= 0.10
```

Also require:

- same realized settled simulator state hash across the three conditions;
- exact tape serialization round-trip;
- same command length / control cadence;
- no command interpolation or hidden queue state silently changed between record/replay conditions.

If P0 fails, **fix the harness or stop**. Do not inspect perturbation outcomes and do not change tasks.

## G0 — fixed physical disturbance

Only after P0 passes.

Use the same 30 CloseDoor configs already validated by Topic 23.

External perturbation panel is frozen as a full grid, not tuned to find a sweet spot:

```text
force magnitude: 50, 100, 150 N
lateral direction: left, right
duration: 0.2 s
```

The force range is externally motivated by recent whole-body humanoid robustness evaluations; using the entire grid prevents post-hoc force selection.

Per config, perturbation timing is determined from the **unperturbed canonical trajectory only**, before any perturbed outcome is seen: 1.0 s before first canonical task-object contact. The same timing is used for all three feedback conditions and every force cell for that config.

Primary outcome:

```text
Y = official SIMPLE episode success
```

Always also store the raw door-joint trajectory, base state, realized external-force application, VLA tape index and actuator tape index.

Bootstrap clusters by physical config, keeping all force/direction/condition cells together.

## Result interpretations

### High-level recovery

```text
fresh >> vla_replay
```

The foundation policy's observation feedback materially contributes to recovery.

### Low-level recovery

```text
fresh ~= vla_replay >> actuator_replay
```

The system looks like a robust VLA, but recovery is primarily supplied by the downstream controller.

### Distributed hierarchy

```text
fresh > vla_replay > actuator_replay
```

Both levels contribute causally.

### Task/servo tolerance

```text
fresh ~= vla_replay ~= actuator_replay
```

The perturbation did not require either learned feedback layer to recover. Do not call this VLA robustness.

### No robust phenomenon

If `fresh` itself collapses across the frozen force panel, there is no recovery phenomenon to attribute on this task; stop without changing force magnitudes or shopping for another task in the same registered experiment.

## Why a positive result would matter

Humanoid/VLA papers increasingly report robustness, recovery, disturbance tolerance, and closed-loop execution from multi-rate hierarchical stacks. But system success does not identify which learner supplies the feedback computation.

A clean result changes how such claims are interpreted:

- if low-level control explains most recovery, VLA robustness benchmarks over-credit foundation-model intelligence;
- if high-level feedback is necessary, that is direct behavioral evidence that VLA inference performs online corrective computation rather than only issuing nominal plans;
- if contributions shift across policy families or pretraining scale, this becomes a mechanism-level scaling result.

## Method opening

- controller-aware VLA training only where high-level feedback is actually needed;
- adaptive VLA replanning frequency based on measured low-level recoverability;
- interface design that allocates disturbances to the appropriate feedback layer;
- benchmarks that separately report semantic/high-level and low-level control robustness;
- co-training objectives that avoid duplicating or fighting downstream feedback.

## Collision audit summary

Several 2025–2026 works make the question timely but do not appear to close it:

- **APEX** shows a real policy-controller execution gap and uses low-level state feedback to adapt VLA/visuomotor references. It does not replay/cut hierarchy feedback to attribute a fixed disturbance-recovery event.
- **OpenHLM** systematically studies whole-body VLA/control interfaces and explicitly uses a high-level policy plus low-level controller; it asks which interface works better, not which feedback layer caused a recovery.
- **HuMI** explicitly trains the low-level controller against command/state mismatch and sudden corrections, establishing low-level correction capability rather than attributing deployed-system recovery.
- **WholeBodyVLA** trains an LMO RL controller for precision and stability under disturbances; its robustness is exactly the kind of system-level result whose source this candidate would separate.
- **SV-VLA** studies open-loop action chunks vs online verification/replanning inside the VLA execution loop; it does not separate VLA feedback from downstream whole-body-controller feedback.
- **PaCo-VLA** is the closest methodological neighbor: it uses counterfactual semantic corruption to show task success requires live semantic VLA inputs while a low-level passivity shield maintains safety. Its causal question is semantic guidance vs shielded geometry/safety in contact-rich insertion, not the contribution of high- vs low-level state feedback to recovery from the same physical disturbance.

I have not found a paper that performs the specific same-perturbation ladder:

```text
live VLA + live controller
vs recorded VLA commands + live controller
vs recorded post-controller references
```

with task outcome as the endpoint.

## Stop rules

Stop immediately if any of these occur:

1. P0 replay fidelity cannot reproduce unperturbed canonical behavior.
2. The two seams are not actually distinct in the deployed stack.
3. The external perturbation is not physically realized as logged.
4. `fresh` shows no meaningful recovery on the frozen force panel.
5. A direct 2025–2026 paper is found that already performs the same feedback-source replay attribution at comparable scale.

Do not rescue by selecting a different force, phase, task, controller or metric after seeing G0.
