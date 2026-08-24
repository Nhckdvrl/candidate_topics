# P0 replay-fidelity result — feedback-source attribution candidate

**Date:** 2026-08-24
**Status:** frozen P0 gate **PASSED**. Registration as a root Topic is licensed by
this result; the physical-disturbance G0 is not yet frozen and no attribution
quantity has been computed.

## What was tested

The candidate question — when a hierarchical robot foundation policy recovers
from a physical disturbance, which feedback layer caused the recovery? — cuts the
deployed stack at two seams that exist in the released source, not in a probe:

```text
VLA  -> WBC       ActionCmd("vla_cmd",        target_upper_body_pose, navigate_cmd,
                                              base_height_command)
WBC  -> actuator  ActionCmd("decoupled_wbc",  target_q, left_hand_q, right_hand_q)
```

P0 asks only whether replaying a recorded tape at either seam reproduces the live
system **when the world is not disturbed at all**. No push is applied anywhere in
this result, and no attribution gap is computed.

## Frozen gate, and the outcome

```text
                            required     observed
fresh                       >= 0.90      1.00   (10/10)
vla_replay                  >= 0.90      1.00   (10/10)
actuator_replay             >= 0.90      1.00   (10/10)
fresh - vla_replay          <= 0.10      0.00
fresh - actuator_replay     <= 0.10      0.00
```

Structural checks, which are contracts rather than thresholds, all held on all
30 cells:

- every replay consumed its whole tape at the recorded cadence (`steps == tape_len`,
  no early exhaustion, no leftover rows);
- every replay row has `server_queries == 0`, i.e. the policy server was never
  contacted once the tape was loaded;
- every row is unperturbed (`force_n == 0`).

Trajectory fidelity, reported alongside the success gate rather than as part of it:

```text
max |door_q(replay) - door_q(fresh)| over all steps and configs   0.000 rad
max terminal base displacement difference                          0.000 m
```

The replays are bit-exact, not merely outcome-equivalent.

**Panel:** `simple/G1WholebodyCloseDoorTeleop-v0`, Psi0 `ckpt_40000`,
`dr-level-0` episodes 0–9, `mujoco_isaac`, `--clock virtual`.
Records: [`records/p0_closedoor.jsonl`](records/p0_closedoor.jsonl),
gate output [`records/p0_gate.json`](records/p0_gate.json).

## The controller is a wall-clock interpolator, and this had to be handled

`Psi0DecoupledWbcAgent.get_action` stamps `target_time = time.monotonic() + 1/control_freq`
and samples the interpolation spline at the real time of the call. Policy-server
latency is therefore an input to the controller, and it is present in `fresh` and
absent in both replay conditions. The runner substitutes a monotonic surrogate
advancing exactly one control period per WBC invocation, in every condition
(`--clock virtual`).

This was measured, not assumed. `--clock real` was run twice on the same three
configs:

| run | machine state | fresh | vla_replay | actuator_replay |
| --- | --- | ---: | ---: | ---: |
| [`clockcheck_real.jsonl`](records/clockcheck_real.jsonl) | loaded (a second rollout + two policy servers) | 0/3 | 0/3 | 0/3 |
| [`clockcheck_real_idle.jsonl`](records/clockcheck_real_idle.jsonl) | lighter load | 1/3 | 1/3 | 1/3 |
| `p0_closedoor.jsonl` (virtual) | any | 10/10 | 10/10 | 10/10 |

The sharpest form of this is not the totals. Config `dr-level-0:0` — same seed,
same scene draw, same checkpoint, same tape — **fails in the loaded run and
succeeds in the lighter one**, in all three conditions alike. Under the virtual
clock the same config reproduces bit-identically across runs. Failure is also not
graceful degradation: the failing rollouts run to the 450-step limit with the door
never moving.

Two things follow, and they should not be conflated.

1. **Replay fidelity is not a clock artefact.** Under the real clock the three
   conditions agree with each other config-by-config, in both the loaded and the
   idle run — including on the configs where all three fail. The tapes are
   faithful either way.
2. **Task competence under the real clock is contaminated by machine load.** The
   same released stack that Topic 23 measured at 30/30 drops to 1/3 under light load and 0/3
   under contention. That is a property of a wall-clock-coupled controller running
   in non-real-time simulation, not of the policy. The virtual clock removes the
   dependence and is at least as competent (10/10).

The virtual clock is therefore the declared instrument. Every record carries a
`clock` field so the choice stays auditable.

## What P0 does *not* establish

This is the part worth stating plainly, because it is the Topic 23 failure mode in
a new costume.

With no disturbance, the state never leaves the recorded trajectory. A whole-body
controller that ignored proprioception entirely — a pure feedforward map from
`vla_cmd` to `target_q` — would replay exactly as well as the real one and would
pass this gate with the same 10/10. In that world the later
`vla_replay - actuator_replay` gap would be structurally zero and
`Δ_low` would be uninterpretable.

So a passing P0 licenses the claim *the tapes are lossless*, and nothing about
whether the lower seam can carry state feedback at all. That second claim needs
its own instrument check before the scientific G0 is frozen; see
[`P0B_RESULTS.md`](P0B_RESULTS.md), which has since been run and **passed**.

## Naming, fixed now rather than after seeing results

`actuator_replay` is still a closed loop below the seam it cuts: joint servo/PD
feedback, actuator dynamics, passive mechanical stabilization and task tolerance
all survive it. The three levels are therefore named

```text
fresh - vla_replay              VLA-level online feedback contribution
vla_replay - actuator_replay    WBC / reference-generation feedback contribution
actuator_replay residual        servo + actuator dynamics + mechanics + task tolerance
```

"low-level controller contribution" is not used for the middle quantity, because
it would claim the whole stack below the VLA when only one layer of it was cut.
P0b narrows that middle term further still: below the VLA seam the arms and hands
are open-loop interpolation, so it can only ever carry locomotion/balance state
feedback.

## Incidental finding: `fresh` is not reproducible run-to-run

The same config run twice under `fresh` produced 278 and 276 control steps. The
policy server's flow sampling is not seeded per query, so the canonical rollout
differs slightly between runs. This does not affect the design — each replay is
paired against the tape from *its own* recording pass — but it does mean a tape
must never be reused across a re-recorded `fresh`.
