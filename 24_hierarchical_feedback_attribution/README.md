# 24 — Where Does Closed-Loop Robustness Live in Hierarchical Robot Foundation Policies?

> **G0 + G1 + G2 RUN, 2026-08-24/25.**
>
> **G0 verdict: `WBC_LEVEL_DOMINATES`.** Of the recovery `fresh` shows over
> `actuator_replay`, the WBC/reference-generation seam accounts for
> essentially all of it (`delta_low = 0.300`, 95% CI `[0.244, 0.356]`); the
> pooled VLA-level online contribution is not statistically distinguishable
> from zero (`delta_high = 0.017`, 95% CI `[-0.050, 0.083]`). See
> [`G0_RESULTS.md`](./G0_RESULTS.md).
>
> **G1 verdict: `NAVIGATION_CHANNEL_CAUSES_REVERSAL`** (100N). G0's near-zero
> pooled `delta_high` hid a left/right sign flip. Factoring the VLA command
> shows that flip lives in the navigation/base channel: `N_100,left = +0.200`
> CI `[0.067,0.367]`, `N_100,right = -0.200` CI `[-0.367,-0.033]`, both
> independently significant and opposite-signed. A post-result audit tightened
> two overclaims about the upper-body channel and the evaluator's verdict
> predicate — see [`G1_RESULTS.md`](./G1_RESULTS.md).
>
> **G2 verdict: `REVERSAL_CONFIRMED_AT_SOME_FORCES_ONLY`.** Completing the
> force grid (50N/150N, 100N reused from G1) with the corrected predicate
> finds the reversal strictly established only at `100N`. `50N` is
> direction-consistent but short of significance; `150N`'s `left` effect is
> exactly `0.000` (CI `[0,0]`, zero on all 30 configs) — the navigation
> channel's `left`-push benefit decays monotonically to nothing as force
> increases (`+0.30 → +0.20 → 0.00`), while its `right`-push cost peaks at
> `100N` rather than trending. See [`G2_RESULTS.md`](./G2_RESULTS.md).

## Natural question

> **When a hierarchical robot foundation policy recovers from a physical
> disturbance, which feedback layer actually caused the recovery?**

A humanoid VLA gets shoved and still finishes the task. Did the VLA see the
change and re-plan, or did the whole-body controller underneath quietly put the
robot back on the nominal trajectory? Papers routinely show the recovery. Nothing
in the demonstration says who gets the credit.

## Why this topic was selected in this order

Topics 09, 19 and 23 all failed the same way: a second-order concept was proposed
first, then an experimental object was found to carry it, and whether that object
existed was only discovered late. Topic 23 is the sharpest case — the experiment
returned a paired difference of `0.967`, 95% CI `[0.90, 1.00]`, 29 apparent
substitution events, and the interpretation was still false, because the treatment
never removed the motor program the claim named.

This topic was built in the opposite order. The causal seam was found first, in
the released source, while debugging Topic 23:

```text
VLA  -> WBC       ActionCmd("vla_cmd",  target_upper_body_pose, navigate_cmd,
                            base_height_command)
WBC  -> actuator  ActionCmd("decoupled_wbc", target_q, left_hand_q, right_hand_q)
```

Both seams are real code paths in `simple/baselines/psi0_decoupled_wbc.py`. The
observation that the WBC re-solves from fresh proprioception and swallows an
upstream intervention was made by watching it happen, not by inference. The
scientific question is then one step away from something already verified to
exist.

## The experiment

Record both command tapes on a successful unperturbed rollout. Then, from the
same initial state and under the same physical push, run three conditions:

```text
fresh            live obs -> live VLA -> live WBC -> robot
vla_replay       recorded nominal vla_cmd tape -> live WBC (live proprio) -> robot
actuator_replay  recorded nominal post-WBC tape -> actuator servo -> robot
```

In `vla_replay` the VLA cannot react to the push at all, while the WBC still sees
that the robot was actually knocked off course. In `actuator_replay` the WBC's
online correction is gone too.

Three success rates give two differences:

```text
delta_high = S_fresh      - S_vla_replay
delta_low  = S_vla_replay - S_actuator_replay
residual   = S_actuator_replay
```

No SAE, no hidden state, no trajectory similarity, no task-null manifold, no
definition of an "alternative motor program". The interpretation of each outcome
is fixed in advance:

| observed | reading |
| --- | --- |
| `fresh >> vla_replay` | the VLA itself is doing closed-loop recovery |
| `fresh ≈ vla_replay >> actuator_replay` | behaviour that looks like a robust foundation VLA is mostly the controller saving it |
| `fresh > vla_replay > actuator_replay` | recovery is distributed across both layers |
| `fresh ≈ vla_replay ≈ actuator_replay` high | this perturbation never demanded learned feedback recovery — servo, mechanics or task slack absorbed it |
| `fresh` collapses | this task has no robustness phenomenon worth attributing at this force grid |

## What the three levels are called, fixed before any number exists

`actuator_replay` is still a closed loop below the seam it cuts. Joint servo/PD
feedback, actuator dynamics, passive mechanical stabilization and task tolerance
all survive it. So:

```text
delta_high  VLA-level online feedback contribution
delta_low   WBC / reference-generation feedback contribution
residual    servo + actuator dynamics + mechanics + task tolerance
```

`delta_low` is **not** "the low-level controller contribution". That phrasing
would claim the whole stack below the VLA when only one layer of it was cut.

P0b narrows `delta_low` further, and this is a real limit rather than a caveat:
below the VLA seam the arms and hands are open-loop interpolation of `vla_cmd`,
so `delta_low` can only ever carry locomotion/balance state feedback. A
disturbance that could only be absorbed by re-reaching with the arm has no
WBC-level route by construction, and any such recovery must appear in
`delta_high`.

## Pre-registration evidence: the instrument was proven before the topic was registered

This is the Topic 23 lesson turned into procedure. Both gates are in
[`../embodied_topic_search/prototypes/feedback_source_attribution/`](../embodied_topic_search/prototypes/feedback_source_attribution/).

**P0 — replay fidelity** (10 configs, no push anywhere). With the world
undisturbed, both replays must reproduce the live system.

```text
fresh 10/10   vla_replay 10/10   actuator_replay 10/10
both gaps 0.00      trajectory divergence 0.000 rad / 0.000 m
server_queries == 0 on every replay row
```

**P0b — WBC seam liveness.** P0's perfect result is also its limit: with nothing
driving the system off the recorded trajectory, a purely feedforward controller
would pass it identically, and then `delta_low` would be structurally zero rather
than informative. P0b holds the `vla_cmd`, the WBC internal state and the clock
identical and changes only the proprioceptive observation:

```text
repeatability floor   0.0 exactly     (same obs, same cmd -> bit-identical)
restore probe         0.0 exactly     (state restoration provably complete)
D                     2.4e-02 .. 4.6e-02 rad on 3/3 configs
```

See [`P0_RESULTS.md`](../embodied_topic_search/prototypes/feedback_source_attribution/P0_RESULTS.md)
and [`P0B_RESULTS.md`](../embodied_topic_search/prototypes/feedback_source_attribution/P0B_RESULTS.md).

## Frozen G0

```text
task        simple/G1WholebodyCloseDoorTeleop-v0
policy      Psi0 ckpt_40000, released SIMPLE checkpoint
configs     dr-level-0/1/2 x episodes 0-9  = 30 matched configs
conditions  fresh / vla_replay / actuator_replay
force grid  0 N control  +  {50, 100, 150} N  x  {left, right}
push        0.2 s lateral shove on torso_link, world-frame along the robot's own
            lateral axis, applied to the simulator and never to any command
timing      canonical first object contact tick - 50 ticks (1.0 s at 50 Hz)
horizon     450 control steps for all three conditions
clock       virtual (nominal 50 Hz), declared in every record
success     unmodified upstream env.unwrapped._success
```

**The whole grid is reported.** No force is selected after seeing an outcome.

**Push timing is derived only from the unperturbed canonical rollout** — the tick
at which the robot first touches the door, minus one second. No perturbed outcome
is inspected to choose it.

**All three conditions get the same 450-step budget.** The recorded tape is held
at its final command beyond its own length; otherwise `fresh` could win merely by
being allowed to run longer than the tape.

### Prerequisites, checked before any delta is read

```text
PREREQUISITE_FAIL_STRUCTURAL        a replay contacted the policy server, a tape
                                    exhausted early, a control cell got a push, a
                                    force cell got none, or the push tick differed
                                    across the three conditions
PREREQUISITE_FAIL_REPLAY_FIDELITY   the force=0 control column failed to reproduce
                                    P0 under the exact G0 code path
PREREQUISITE_FAIL_PUSH_INEFFECTIVE  at the largest force the median base
                                    displacement from the canonical trajectory is
                                    below 0.02 m, i.e. the push is not an
                                    intervention
INSUFFICIENT_MATCHED_CONFIGS        any grid cell has fewer than 24 of 30 complete
                                    condition triples
```

### Stop outcomes that are results, not failures

```text
NO_ROBUSTNESS_PHENOMENON                 fresh and actuator_replay both >= 0.90 at
                                         every force -> nothing required learned
                                         feedback recovery
FRESH_COLLAPSE_NOTHING_TO_ATTRIBUTE      fresh <= 0.10 at every force
NO_MEANINGFUL_LEARNED_FEEDBACK_CONTRIBUTION
                                         both deltas below the pre-registered
                                         minimum worthy effect of 0.10, or their
                                         clustered-bootstrap CI includes zero
```

Statistics: clustered bootstrap over the 30 physical configs, keeping each
config's whole force panel intact, 10 000 resamples, seed `20260824`. Configs are
the unit of independence, not cells.

Do not tune the force grid, the timing rule, the minimum worthy effect, or the
horizon after seeing a number. A prerequisite failure is a result about this
task and this force grid, not about the hypothesis.

## Method opening

The result points at a lever either way, which is why it passes the then-what
test:

- if most recovery is WBC-level, do not pay for a full VLA re-plan on every
  disturbance — learn low-level recoverability and wake the VLA only when the
  controller cannot absorb the error;
- if the VLA contribution is large, the question becomes which observations,
  memory and training make high-level corrective re-planning work.

Either way it opens compute allocation, replanning policy, the policy–controller
interface, and co-training, rather than ending at "we found a pattern".

## Cost

630 rollouts at roughly 220 s each is about 38 GPU-hours, or 8–10 hours wall
clock spread across the four local GPUs and the free `fvcrc` nodes. This is not a
cheap experiment and was not called one.

## Files

- `topic24_runner.py` — canonical recording, the push, and the three conditions.
- `g0_core.py` — the frozen decision procedure and the clustered bootstrap.
- `tests/test_g0_core.py` — pure-logic tests for that procedure (`19 passed`).
- `RUN_LOCAL_AGENT.md` — how to run it.
