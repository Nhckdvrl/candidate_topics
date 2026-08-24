# Feedback-Source Attribution Prototype

This directory is deliberately a **pre-registration instrument test**, not Topic 24 yet.

Candidate question:

> When a hierarchical robot foundation policy recovers from a physical disturbance, which feedback layer actually caused the recovery?

## Upstream target

- Psi0 + SIMPLE
- `simple/G1WholebodyCloseDoorTeleop-v0`
- same released checkpoint/config panel already validated by archived Topic 23

The deployed stack exposes two distinct seams:

```text
VLA -> WBC       : vla_cmd
WBC -> actuator  : target_q / left_hand_q / right_hand_q
```

## P0 only: no external perturbation

For 10 fixed CloseDoor configs:

1. run canonical live policy and record both command tapes;
2. reset to the same realised settled simulator state and replay the recorded VLA-level tape through the live WBC;
3. reset again and replay the recorded post-WBC actuator-reference tape;
4. write one result row per condition with `force_n=0`.

Frozen technical pass:

```text
fresh >= 0.90
vla_replay >= 0.90
actuator_replay >= 0.90
fresh - vla_replay <= 0.10
fresh - actuator_replay <= 0.10
```

Do not apply pushes, inspect attribution gaps, change tasks, or tune any scientific threshold before this passes.

## What to record

The canonical recorder must capture every executed control tick:

```text
VLA seam
  target_upper_body_pose
  navigate_cmd
  base_height_command

post-WBC seam
  target_q
  left_hand_q
  right_hand_q
```

Also persist:

- config id / DR level;
- exact settled MuJoCo-state hash before policy start;
- command count and control timestamps;
- official SIMPLE success;
- raw door joint trajectory.

A replay condition must consume the same number of tape rows at the same cadence. Exhausting early or leaving unused rows is a P0 failure.

## Files

- `p0_replay_contract.py` — lossless command-tape schema and serialization checks.
- `g0_core.py` — P0 fidelity scoring plus the later three-level feedback attribution statistics.
- `p0_runner.py` — the executable P0: records both seams on a live rollout and
  replays each seam from the same config. Runs inside the SIMPLE venv.
- `p0_analyze.py` — the frozen gate plus the structural checks that cannot be tuned.
- `p0b_seam_liveness.py` — the command-level paired seam test that gates registration.
- `tests/` — pure-logic tests for both the statistics and the gate.

## How the two seams are actually cut

Both interventions are the upstream data path, not a reimplementation of it.

`vla_replay` pre-loads the recorded `vla_cmd` tape into
`SonicDecoupledWbcAgent._action_queue`. `Psi0DecoupledWbcAgent.get_action` queries
the policy server only when that queue is empty, so a pre-loaded queue removes the
VLA from the loop while every downstream line — `_build_wbc_observation` on live
proprioception, `set_goal`, `_wbc_policy.get_action` — still executes unchanged.
`server_queries == 0` on every replay row is the recorded proof that this held.

`actuator_replay` bypasses the agent entirely and steps the environment with the
recorded `ActionCmd("decoupled_wbc", target_q, left_hand_q, right_hand_q)`.

## The controller is a wall-clock interpolator

`Psi0DecoupledWbcAgent.get_action` stamps `target_time = time.monotonic() + 1/control_freq`
and samples the spline at the real time of the call
(`third_party/decoupled_wbc/control/policy/interpolation_policy.py`). Policy-server
latency is therefore part of the controller's input, and it is present in `fresh`
and absent in both replay conditions. Left alone, that difference alone would move
the WBC output and be misread as a replay-fidelity failure.

`p0_runner.py --clock virtual` (default) substitutes a monotonic surrogate that
advances exactly one control period per WBC invocation, in all three conditions —
the nominal 50 Hz schedule the controller is written against. `--clock real` keeps
upstream behaviour and is retained so the choice stays measurable. The gate below
applies to whichever clock the run declares; the recorded rows carry the `clock`
field.

## Running P0

```bash
cd $SIMPLE && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python \
  /path/to/feedback_source_attribution/p0_runner.py \
  --env-id simple/G1WholebodyCloseDoorTeleop-v0 \
  --data-dir $EVAL/G1WholebodyCloseDoorTeleop-v0/dr-level-0 \
  --out records/p0_closedoor.jsonl --tape-dir tapes/p0_closedoor \
  --port 22085 --sim-mode mujoco_isaac --clock virtual --num-episodes 10
```

```bash
PYTHONPATH=. python p0_analyze.py records/p0_closedoor.jsonl --tape-dir tapes/p0_closedoor
```

## Local logic tests

```bash
cd embodied_topic_search/prototypes/feedback_source_attribution
PYTHONPATH=. pytest -q
```

Current result: `13 passed`.

## Status

```text
P0   replay fidelity     PASS   see P0_RESULTS.md
P0b  WBC seam liveness   PASS   see P0B_RESULTS.md
```

P0b also fixed a limit on what the middle quantity can mean: below the VLA seam
the arms and hands are open-loop interpolation, so `vla_replay - actuator_replay`
can only ever carry locomotion/balance state feedback. See
[`P0B_RESULTS.md`](P0B_RESULTS.md).

## Only after P0 passes

Register the question as a root Topic, then freeze the physical-disturbance G0. The provisional candidate document currently specifies a non-selected force grid `{50,100,150} N x {left,right} x 0.2 s`, with perturbation timing derived only from each config's unperturbed canonical trajectory.
