# P0 — replay-fidelity result (CloseDoor, 2026-08-24)

**Verdict: `PASS`.** The two replay instruments are lossless on the unperturbed
task. This licenses P0b, not yet Topic 24.

## What was run

```text
task        simple/G1WholebodyCloseDoorTeleop-v0
policy      Psi0 ckpt_40000, released SIMPLE checkpoint
configs     dr-level-0, episodes 0-9 (10 configs)
conditions  fresh / vla_replay / actuator_replay
push        none anywhere (force_n = 0 on every row)
clock       virtual (see below)
sim-mode    mujoco_isaac
```

## Gate

The gate was written down before any rollout was run.

| quantity | frozen requirement | observed |
| --- | --- | ---: |
| `fresh` | >= 0.90 | **1.00** (10/10) |
| `vla_replay` | >= 0.90 | **1.00** (10/10) |
| `actuator_replay` | >= 0.90 | **1.00** (10/10) |
| `fresh - vla_replay` | <= 0.10 | **0.00** |
| `fresh - actuator_replay` | <= 0.10 | **0.00** |

Structural checks, which are not thresholds and cannot be tuned:

```text
server_queries == 0 on every replay row        pass  (VLA provably out of the loop)
steps == tape_len on every replay row           pass
no tape exhausted early                         pass
every row force_n == 0                          pass
```

Trajectory-level divergence from each config's own `fresh` rollout:

```text
vla_replay        max door deviation 0.0 rad   max terminal base deviation 0.0 m
actuator_replay   max door deviation 0.0 rad   max terminal base deviation 0.0 m
```

Records: [`records/p0_closedoor.jsonl`](records/p0_closedoor.jsonl),
[`records/p0_gate.json`](records/p0_gate.json).

## What this result does not establish

The zero divergence is exactly as strong and exactly as narrow as it looks.
With nothing disturbing the world, neither replay condition is ever driven off
the recorded trajectory, so no feedback of any kind had to act for the replays
to reproduce `fresh`. A purely feedforward whole-body controller would produce
the same table.

So P0 establishes that the plumbing is lossless. It is silent on whether the two
seams carry *different* feedback, which is what

```text
S_vla_replay - S_actuator_replay
```

would later have to mean. That question is P0b, and it must pass before the
candidate is registered as a root Topic. This is the Topic 23 lesson applied to
our own instrument rather than to the upstream policy: a clean intervention
number is not evidence that the intervention removed what it names.

## Naming, fixed now rather than after seeing results

`actuator_replay` is still a closed loop below the seam it cuts — joint servo/PD
feedback, actuator dynamics, passive mechanical stabilization and task tolerance
all survive it. The three levels are therefore:

```text
fresh - vla_replay                 VLA-level online feedback contribution
vla_replay - actuator_replay       WBC / reference-generation feedback contribution
actuator_replay residual           servo + mechanics + task tolerance
```

"low-level controller contribution" is not used for the middle quantity.

## Engineering confound found while building the instrument

`Psi0DecoupledWbcAgent.get_action` stamps `target_time = time.monotonic() + 1/control_freq`,
and `InterpolationPolicy` samples the spline at the wall-clock time of the call.
Model-inference latency is therefore an input to the controller. The deployed
stack is not

```text
a_t = f(o_t)
```

but closer to

```text
a_t = f(o_t, dt_compute)
```

That latency is present in `fresh` and absent in both replay conditions, so left
alone it would have moved the WBC output and been misread as a replay-fidelity
failure. `p0_runner.py --clock virtual` advances a monotonic surrogate by exactly
one control period per WBC invocation in all three conditions — the nominal 50 Hz
schedule the controller is written against.

The size of the effect was measured rather than assumed, on the same 
task, policy and configs:

| clock | machine | `fresh` |
| --- | --- | ---: |
| virtual | any | 10/10 |
| real | one concurrent simulator + two policy servers | **0/9, the door never moves** |
| real | low load | succeeds |

Under contention the released stack does not merely degrade, it fails the task
outright, in all three conditions alike
([`records/clockcheck_real.jsonl`](records/clockcheck_real.jsonl)).

**This is recorded as an engineering/system observation and is deliberately not
part of the candidate's hypothesis.** All causal replay experiments use the
frozen nominal virtual control clock; real-clock sensitivity is never used as
hypothesis evidence. Folding it in would turn a clean attribution question into
"VLA latency + WBC clock + system scheduling + recovery attribution" and regrow
exactly the control chain the candidate was chosen to avoid.

It may deserve to become its own question later — *are robot-policy benchmarks
accidentally benchmarking compute latency?* — but not here, and not now.
