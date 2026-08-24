# P0b — WBC seam liveness (CloseDoor, 2026-08-24)

**Verdict: `SEAM_LIVE` on all three configs.** Combined with the P0 pass, the
candidate may now be registered as a root Topic.

## Why P0 was not enough

P0 showed both replay instruments reproduce the live rollout exactly with no
disturbance — including `max door deviation 0.0 rad`. That "perfect" number is
also the reason it proves less than it looks. With nothing driving the system off
the recorded trajectory, no feedback of any kind had to act. A purely feedforward
whole-body controller would have produced the identical table, and in that world

```text
S_vla_replay - S_actuator_replay
```

would be structurally zero rather than informative.

## What was measured

A command-level paired test, not a behavioural one. Nothing was pushed and the
environment was not stepped after the measurement, so contact, dynamics,
controller history and trajectory phase cannot enter.

```text
same config, same tick, same fixed vla_cmd,
same WBC internal state, same clock
              |
   canonical proprio  -> WBC -> target_q^0
   perturbed proprio  -> WBC -> target_q^1
              |
        D = |target_q^1 - target_q^0|
```

The canonical side is re-evaluated live rather than read back from the recorded
tape, so interpolation state, clock position and previous-tick history are held
identical by construction and cannot contribute to `D`.

Frozen before the run: tick `= round(0.4 * len(tape))`, derived only from the
unperturbed canonical rollout; a single +0.05 rad body-frame roll offset on the
floating-base orientation and the torso IMU quaternion; one magnitude, no sweep,
no per-config choice.

## Result

| config | tick | repeat floor | restore probe | `D` (max over post-WBC command) | ratio | joints changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dr-level-0:0 | 110/276 | **0.0** | **0.0** | **4.56e-02 rad** | inf | 15 |
| dr-level-0:1 | 123/307 | **0.0** | **0.0** | **2.43e-02 rad** | inf | 15 |
| dr-level-0:2 | 112/281 | **0.0** | **0.0** | **3.93e-02 rad** | inf | 15 |

The repeatability floor is exactly zero: the same observation and the same
command, from the restored state, return a bit-identical command vector. The
restore probe — the canonical evaluation re-run *after* the perturbed one — also
returns bit-identical output, so state restoration was complete in practice and
the difference cannot be residue from the perturbed call. The separation is
therefore not "large relative to noise"; there is no noise.

Records: [`records/p0b_seam_liveness.jsonl`](records/p0b_seam_liveness.jsonl).

### Deviation from the frozen tick condition, reported not hidden

The tick rule required the approach phase, checked as "the door joint has not
moved". On `dr-level-0:2` the door had drifted **7.2e-04 rad** by tick 112, so the
implemented exact-equality check records `door_untouched_at_tick = false`. That is
passive settling, three orders of magnitude below the ~0.95 rad of travel the task
requires and far from any contact; configs 0 and 1 show exactly 0.0. The verdict
does not rest on config 2 — configs 0 and 1 satisfy the frozen condition as
written and both give `SEAM_LIVE`.

## The structural finding that constrains what Topic 24 may claim

The same 15 joints move on every config, and they are the same 15 every time:

```text
12 leg joints + 3 waist joints      changed
7 left-arm + 7 right-arm joints     0.0 exactly
left_hand_q, right_hand_q           0.0 exactly
```

This is not an empirical accident, it is what the source says. In
`G1DecoupledWholeBodyPolicy.set_observation`:

> Upper body policy is open loop (just interpolation), so we don't need to set
> the observation

Only `lower_body_policy` receives the observation. Below the VLA seam, the arms
and hands are pure feedforward interpolation of `vla_cmd`; the state-dependent
computation is the lower-body RL policy alone.

**Written down now, before the G0, so it cannot be discovered afterwards as a
convenient reading:**

```text
vla_replay - actuator_replay
    can only ever measure locomotion / balance-level state feedback.
    It cannot contain arm-level corrective reaching, because no arm or hand
    command below the VLA seam depends on the current state at all.
```

So a disturbance that could only be absorbed by re-reaching with the arm has no
WBC-level route by construction, and any such recovery must appear in
`fresh - vla_replay`. This sharpens the three levels rather than weakening them,
but it is a real limit on the middle quantity and must be stated in the G0.

## Ladder status

```text
P0   replay fidelity        PASS   10/10 all three conditions, 0.0 divergence
P0b  WBC seam liveness      PASS   D = 2.4e-02 to 4.6e-02 rad over a zero floor
                                   -> register the root Topic, then freeze G0
```
